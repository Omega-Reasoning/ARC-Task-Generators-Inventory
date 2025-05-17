from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import retry, Contiguity

class IntersectingLinesTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of same sizes.", 
            "Each grid contains :", 
            "  * - Two-celled horizontal blocks with {{color(\"h_color\")}} color.",
            "  * - Two-celled vertical blocks with {{color(\"v_color\")}} color.", 
            "These horizontal and vertical blocks are positioned such that when extended in their respective directions, they intersect at one or more grid cells.."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "Each horizontal and vertical block is then extended fully:",
            "  * - Horizontal blocks extend left and right across the row, maintaining the {{color(\"h_color\")}} color.",
            "  * - Vertical blocks extend up and down across the column, maintaining the {{color(\"v_color\")}} color.",
            "Wherever a horizontal and vertical extension intersect, the cell at the intersection is assigned a distinct {{color(\"i_color\")}} color."
        ]
        
        taskvars_definitions = {
            "h_color": {
                "type": "int",
                "description": "Color of the horizontal blocks.",
                "range": [1, 9]
            },  
            "v_color": {
                "type": "int",
                "description": "Color of the vertical blocks.",
                "range": [1, 9]
            },
            "i_color": {
                "type": "int",
                "description": "Color of the intersection cells.",
                "range": [1, 9]
            }
        }
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, gridvars):
        # Random grid size (preferring even numbers and more than 5 rows/columns)
        size = random.choice([6, 8, 10, 12, 14, 16])
        
        # Colors for horizontal, vertical blocks and intersections
        h_color = self.gridvars.get("h_color", random.randint(1, 9))
        v_color = self.gridvars.get("v_color", random.choice([c for c in range(1, 10) if c != h_color]))
        i_color = self.gridvars.get("i_color", random.choice([c for c in range(1, 10) if c != h_color and c != v_color]))
        
        # Initialize empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # Create one horizontal block
        h_row = random.randint(0, size-1)
        h_col_start = random.randint(0, size-2)
        grid[h_row, h_col_start:h_col_start+2] = h_color
        
        # Create one vertical block
        # Decide whether the blocks should intersect in the input (randomly)
        should_intersect_in_input = random.choice([True, False])
        
        if should_intersect_in_input:
            # Place vertical block to overlap with horizontal block
            v_col = random.choice([h_col_start, h_col_start+1])
            v_row_start = random.randint(0, size-2)
            
            # Make sure at least one cell of the vertical block doesn't overlap with the horizontal
            # (we need two distinct blocks, not just one L-shape)
            if v_row_start <= h_row and v_row_start + 1 >= h_row:
                if h_row > 0:
                    v_row_start = h_row - 1
                else:
                    v_row_start = h_row + 1
            
            grid[v_row_start:v_row_start+2, v_col] = v_color
            
            # If they overlap in the input, we need to correct that cell
            if (h_row >= v_row_start and h_row < v_row_start+2 and 
                v_col >= h_col_start and v_col < h_col_start+2):
                # We'll prioritize the vertical block at the intersection
                grid[h_row, v_col] = v_color
        else:
            # Place vertical block to not overlap with horizontal block, but to create intersection when extended
            v_col = random.randint(0, size-1)
            v_row_start = random.randint(0, size-2)
            
            # Ensure they will intersect when extended
            while v_col < h_col_start or v_col > h_col_start+1:
                v_col = random.randint(0, size-1)
                
            # Ensure they don't overlap in the input
            while (h_row >= v_row_start and h_row < v_row_start+2):
                v_row_start = random.randint(0, size-2)
                
            grid[v_row_start:v_row_start+2, v_col] = v_color
        
        # Store colors as grid variables
        self.gridvars["h_color"] = h_color
        self.gridvars["v_color"] = v_color
        self.gridvars["i_color"] = i_color
        
        return grid
    
    def transform_input(self, input_grid, gridvars=None):
        # Extract colors from gridvars if provided, otherwise use self.gridvars
        if gridvars is None:
            gridvars = self.gridvars
            
        h_color = gridvars["h_color"]
        v_color = gridvars["v_color"]
        i_color = gridvars["i_color"]
        
        # Create a copy of the input grid
        output_grid = np.copy(input_grid)
        
        # Find the horizontal block
        horizontal_blocks = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        horizontal_blocks = horizontal_blocks.filter(lambda obj: 
                                                   h_color in obj.colors and 
                                                   obj.width == 2 and
                                                   obj.height == 1)
        
        # Find the vertical block
        vertical_blocks = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        vertical_blocks = vertical_blocks.filter(lambda obj: 
                                               v_color in obj.colors and 
                                               obj.width == 1 and
                                               obj.height == 2)
        
        # Extend horizontal block
        for h_block in horizontal_blocks:
            row = list(h_block.cells)[0][0]  # Get the row of the block
            output_grid[row, :] = h_color
        
        # Extend vertical block
        for v_block in vertical_blocks:
            col = list(v_block.cells)[0][1]  # Get the column of the block
            output_grid[:, col] = v_color
        
        # Mark intersections
        for h_block in horizontal_blocks:
            row = list(h_block.cells)[0][0]
            for v_block in vertical_blocks:
                col = list(v_block.cells)[0][1]
                output_grid[row, col] = i_color
        
        return output_grid
    
    def create_grids(self):
        # Random number of training examples
        num_train_pairs = random.randint(3, 5)
        
        gridvars = {
            "h_color": random.randint(1, 9),
            "v_color": None,
            "i_color": None
        }
        
        # Ensure different colors for vertical blocks and intersections
        while gridvars["v_color"] is None or gridvars["v_color"] == gridvars["h_color"]:
            gridvars["v_color"] = random.randint(1, 9)
            
        while gridvars["i_color"] is None or gridvars["i_color"] == gridvars["h_color"] or gridvars["i_color"] == gridvars["v_color"]:
            gridvars["i_color"] = random.randint(1, 9)
        
        self.gridvars = gridvars
        
        # Generate train and test pairs
        train_pairs = []
        for _ in range(num_train_pairs):
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_input = self.create_input(gridvars)
        test_output = self.transform_input(test_input, gridvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)
