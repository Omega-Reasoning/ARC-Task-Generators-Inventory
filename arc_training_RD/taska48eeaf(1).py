from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import create_object, Contiguity, retry

class taska48eeafGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size MxN",
            "The grid contains one 2x2 square object of {{color(\"block_color\")}} color, and this block can be placed anywhere in the grid.",
            "The main block 2x2 square is surrounded with isolated scattered cells (<5), where the scattered cells are of a {{color(\"object_color\")}} color.",
            "These scattered cells are placed such that they are either horizontal or vertical or diagonal to the 2x2 main block square cells, but NOT touching the main block or each other."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid main 2x2 square block.",
            "The scattered cells around the main block are attached like tails to the main block with respect to their position. Suppose if one of the scattered cells is horizontally placed across the main block, then it gets attached besides the main block horizontally. The same rule is applied for cells being vertical and diagonally placed.",
            "The attachment should not disturb the main block. The scattered cells must occur just as an extension of the main block in the output grid.",
            "Also, the scattered cell alone must be attached to the main block, it must not form like a series of attachments."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables with random values
        grid_size = random.randint(8, 15)  # Larger grid for better spacing
        block_color = random.randint(1, 9)
        object_color = random.randint(1, 9)
        
        # Ensure different colors
        while object_color == block_color:
            object_color = random.randint(1, 9)
        
        # Create variables dictionary for grid generation
        gridvars = {
            'grid_size': grid_size,
            'block_color': block_color,
            'object_color': object_color
        }
        
        # Create 3-5 training pairs
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            # Create input grid with cell types info
            input_grid, cell_info = self.create_input(gridvars, min_scattered=3)    
            output_grid = self.transform_input(input_grid, cell_info)
            
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create one test pair 
        test_input, cell_info = self.create_input(gridvars, min_scattered=3)
        test_output = self.transform_input(test_input, cell_info)
            
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        return {}, TrainTestData(train=train_pairs, test=test_examples)
    
    def create_input(self, gridvars, min_scattered=1):
        grid_size = gridvars['grid_size']
        block_color = gridvars['block_color']
        object_color = gridvars['object_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Randomly place the 2x2 main block
        max_pos = grid_size - 5  # Leave more room on edges
        block_row = random.randint(2, max_pos)
        block_col = random.randint(2, max_pos)
        
        # Place the 2x2 block
        for r in range(block_row, block_row + 2):
            for c in range(block_col, block_col + 2):
                grid[r, c] = block_color
        
        # Define block boundaries
        block_top = block_row
        block_bottom = block_row + 1
        block_left = block_col
        block_right = block_col + 1
        
        # Create a list to store information about each scattered cell
        # This will help with accurate transformations later
        scattered_cell_info = []
        
        # Determine how many scattered cells to place (1-4)
        num_scattered = random.randint(max(min_scattered, 1), 4)
        
        # Create a set of possible directions ("horizontal", "vertical", "diagonal")
        directions = ["horizontal", "vertical", "diagonal"]
        random.shuffle(directions)  # Shuffle to randomize which we use first
        
        # Ensure we use at least the min_scattered number of directions
        selected_directions = directions[:num_scattered]
        
        # For additional cells beyond the first 3, repeat directions randomly
        if num_scattered > 3:
            selected_directions.extend(random.sample(directions, num_scattered - 3))
        
        for direction in selected_directions:
            # Define possible positions for this type of scattered cell
            valid_positions = []
            
            if direction == "horizontal":
                # Horizontal cells: same row as block, different column (not adjacent)
                for r in range(block_top, block_bottom + 1):
                    # Left side of block (2-3 cells away)
                    for c in range(max(0, block_left - 3), block_left - 1):
                        valid_positions.append((r, c, "left"))
                    
                    # Right side of block (2-3 cells away)
                    for c in range(block_right + 2, min(grid_size, block_right + 4)):
                        valid_positions.append((r, c, "right"))
            
            elif direction == "vertical":
                # Vertical cells: same column as block, different row (not adjacent)
                for c in range(block_left, block_right + 1):
                    # Above block (2-3 cells away)
                    for r in range(max(0, block_top - 3), block_top - 1):
                        valid_positions.append((r, c, "up"))
                    
                    # Below block (2-3 cells away)
                    for r in range(block_bottom + 2, min(grid_size, block_bottom + 4)):
                        valid_positions.append((r, c, "down"))
            
            else:  # diagonal
                # Diagonal cells: different row and column, both offset similarly
                
                # Top-left (2-3 cells diagonally)
                for d in range(2, 4):
                    r, c = block_top - d, block_left - d
                    if 0 <= r < grid_size and 0 <= c < grid_size:
                        valid_positions.append((r, c, "top-left"))
                
                # Top-right (2-3 cells diagonally)
                for d in range(2, 4):
                    r, c = block_top - d, block_right + d
                    if 0 <= r < grid_size and 0 <= c < grid_size:
                        valid_positions.append((r, c, "top-right"))
                
                # Bottom-left (2-3 cells diagonally)
                for d in range(2, 4):
                    r, c = block_bottom + d, block_left - d
                    if 0 <= r < grid_size and 0 <= c < grid_size:
                        valid_positions.append((r, c, "bottom-left"))
                
                # Bottom-right (2-3 cells diagonally)
                for d in range(2, 4):
                    r, c = block_bottom + d, block_right + d
                    if 0 <= r < grid_size and 0 <= c < grid_size:
                        valid_positions.append((r, c, "bottom-right"))
            
            # If we found valid positions for this direction, place a scattered cell
            if valid_positions:
                # Choose a random position
                row, col, orientation = random.choice(valid_positions)
                
                # Check for overlap with existing scattered cells
                overlap = False
                for r, c, _, _ in scattered_cell_info:
                    if abs(r - row) <= 1 and abs(c - col) <= 1:
                        overlap = True
                        break
                
                # Only place if no overlap
                if not overlap:
                    grid[row, col] = object_color
                    scattered_cell_info.append((row, col, direction, orientation))
        
        # If we couldn't place enough cells, try again
        if len(scattered_cell_info) < min_scattered:
            return self.create_input(gridvars, min_scattered)
        
        return grid, {
            'block': (block_top, block_left, block_bottom, block_right),
            'scattered_cells': scattered_cell_info,
            'block_color': block_color,
            'cell_color': object_color
        }
    
    def transform_input(self, input_grid, cell_info):
        # Create output grid starting with zeros
        output_grid = np.zeros_like(input_grid)
        
        # Get block info
        block_top, block_left, block_bottom, block_right = cell_info['block']
        block_color = cell_info['block_color']
        
        # First, copy the main 2x2 block to the output
        for r in range(block_top, block_bottom + 1):
            for c in range(block_left, block_right + 1):
                output_grid[r, c] = block_color
        
        # Process each scattered cell according to its orientation
        for row, col, direction, orientation in cell_info['scattered_cells']:
            cell_color = cell_info['cell_color']
            
            # Calculate attach position based on orientation
            if direction == "horizontal":
                if orientation == "left":
                    # Cell is to the left of block, attach to left edge
                    attach_r = row
                    attach_c = block_left - 1
                else:  # right
                    # Cell is to the right of block, attach to right edge
                    attach_r = row
                    attach_c = block_right + 1
            
            elif direction == "vertical":
                if orientation == "up":
                    # Cell is above block, attach to top edge
                    attach_r = block_top - 1
                    attach_c = col
                else:  # down
                    # Cell is below block, attach to bottom edge
                    attach_r = block_bottom + 1
                    attach_c = col
            
            else:  # diagonal
                if orientation == "top-left":
                    # Cell is top-left diagonal, attach to top-left corner
                    attach_r = block_top - 1
                    attach_c = block_left - 1
                elif orientation == "top-right":
                    # Cell is top-right diagonal, attach to top-right corner
                    attach_r = block_top - 1
                    attach_c = block_right + 1
                elif orientation == "bottom-left":
                    # Cell is bottom-left diagonal, attach to bottom-left corner
                    attach_r = block_bottom + 1
                    attach_c = block_left - 1
                else:  # bottom-right
                    # Cell is bottom-right diagonal, attach to bottom-right corner
                    attach_r = block_bottom + 1
                    attach_c = block_right + 1
            
            # Ensure attachment position is valid (in bounds)
            if 0 <= attach_r < input_grid.shape[0] and 0 <= attach_c < input_grid.shape[1]:
                output_grid[attach_r, attach_c] = cell_color
        
        return output_grid

