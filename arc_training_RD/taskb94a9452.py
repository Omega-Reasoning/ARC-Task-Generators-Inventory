from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects, GridObject

class Taskb94a9452Generator(ARCTaskGenerator):
    def __init__(self):
        # Initialize reasoning chains
        self.input_reasoning_chain = [
            "Input grids are of different sizes.", 
            "The grid consists of exactly one square block.", 
            "The square block is usually greater than or equal to 3x3.",
            "These blocks form two parts,namely an outer boundary region of {color(\"bound_color\")} color and inner block region always a square of {color(\"block_color\")} color."
        ]
        
        self.transformation_reasoning_chain = [
            "The output grid is only the size of the square block from the input grid.",
            "The boundary region and inner block regions are copied from the input grid, but there is a slight change in color.",
            "Here, The boundary region gets {color(\"block_color\")} color and the inner block region gets {color(\"bound_color\")} color."
        ]
        
        # Define task variables
        taskvars_definitions = {
            "bound_color": "color of the boundary of the square block",
            "block_color": "color of the inner region of the square block"
        }
        
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)
    
    def create_input(self, gridvars):
        # Extract colors from gridvars
        bound_color = gridvars["bound_color"]
        block_color = gridvars["block_color"]
        
        # Randomly determine grid dimensions (between 5 and 20)
        grid_height = random.randint(5, 20)
        grid_width = random.randint(5, 20)
        
        # Randomly determine block size (at least 3x3 but smaller than grid)
        block_size = random.randint(3, min(grid_height-2, grid_width-2, 10))
        
        # Create empty grid
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Randomly place the block within the grid
        max_row = grid_height - block_size
        max_col = grid_width - block_size
        start_row = random.randint(0, max_row)
        start_col = random.randint(0, max_col)
        
        # Fill the entire block with the boundary color
        for r in range(start_row, start_row + block_size):
            for c in range(start_col, start_col + block_size):
                grid[r, c] = bound_color
        
        # Calculate inner square dimensions and position
        inner_size = block_size - 2  # Inner square is boundary-2
        inner_start_row = start_row + 1
        inner_start_col = start_col + 1
        
        # Fill the inner square with the block color
        for r in range(inner_start_row, inner_start_row + inner_size):
            for c in range(inner_start_col, inner_start_col + inner_size):
                grid[r, c] = block_color
        
        return grid
    
    def transform_input(self, input_grid, gridvars):
        # Extract colors from gridvars
        bound_color = gridvars["bound_color"]
        block_color = gridvars["block_color"]
        
        # Find the block in the input grid (connected non-zero region)
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0, monochromatic=False)
        
        if len(objects) != 1:
            raise ValueError(f"Expected exactly one object, found {len(objects)}")
        
        block_object = objects[0]
        
        # Extract the block bounds
        block_rows, block_cols = block_object.bounding_box
        block_height = block_rows.stop - block_rows.start
        block_width = block_cols.stop - block_cols.start
        
        # Create output grid of the same size as the block
        output_grid = np.zeros((block_height, block_width), dtype=int)
        
        # Extract the block and swap colors
        for r in range(block_rows.start, block_rows.stop):
            for c in range(block_cols.start, block_cols.stop):
                if input_grid[r, c] == bound_color:
                    output_grid[r - block_rows.start, c - block_cols.start] = block_color
                elif input_grid[r, c] == block_color:
                    output_grid[r - block_rows.start, c - block_cols.start] = bound_color
        
        return output_grid
    
    def create_grids(self):
        # Select distinct random colors for bound and block (between 1 and 9)
        colors = random.sample(range(1, 10), 2)
        bound_color, block_color = colors
        
        # Create gridvars dictionary
        gridvars = {
            "bound_color": bound_color,
            "block_color": block_color
        }
        
        # Generate 3-5 training pairs
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train_pairs):
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate one test pair
        test_input = self.create_input(gridvars)
        test_output = self.transform_input(test_input, gridvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
             
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)

