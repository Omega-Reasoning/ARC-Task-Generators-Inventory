from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import create_object, Contiguity

class Task4522001fGenerator(ARCTaskGenerator):
    def __init__(self):
        # Initialize the input reasoning chain as specified
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain exactly one {vars['grid_size']-1}x{vars['grid_size']-1} square block, positioned in one of the four corners of the grid.",
            "Three quadrants of the block are colored {color('object_color1')}, while one quadrant is colored {color('object_color2')}.",
            "The quadrant colored {color('object_color2')} is the one directly opposite to the quadrant placed in the grid corner.",
            "The position of the block should vary across examples."
        ]
        
        # Initialize the transformation reasoning chain as specified
        transformation_reasoning_chain = [
            "The output grids are of size {4*(vars['grid_size']-1)+1}x{4*(vars['grid_size']-1)+1}.",
            "They are constructed by creating two {2*(vars['grid_size']-1)}x{2*(vars['grid_size']-1)} {color('object_color1')} square blocks.",
            "The first block is positioned in the same corner where the input block was placed, and the second is placed so that it is diagonally connected to the first one."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'grid_size': random.choice([3,5, 7]),  # Odd grid size between 5 and 9
            'object_color1': random.randint(1, 9),
        }
        
        # Ensure object_color2 is different from object_color1
        available_colors = [c for c in range(1, 10) if c != taskvars['object_color1']]
        taskvars['object_color2'] = random.choice(available_colors)
        
        # Determine corner positions for train and test inputs to be different
        corner_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # Top-left, Top-right, Bottom-left, Bottom-right
        random.shuffle(corner_positions)
        
        # Create 2 train examples and 1 test example
        train_examples = []
        for i in range(2):
            corner = corner_positions[i]
            gridvars = {'corner': corner}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = [{
            'input': self.create_input(taskvars, {'corner': corner_positions[2]}),
            'output': self.transform_input(self.create_input(taskvars, {'corner': corner_positions[2]}), taskvars)
        }]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        corner = gridvars['corner']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Block size (one less than grid size)
        block_size = grid_size - 1
        half_block = block_size // 2
        
        # Determine the starting position based on the corner
        row_start = 0 if corner[0] == 0 else grid_size - block_size
        col_start = 0 if corner[1] == 0 else grid_size - block_size
        
        # Fill the block with object_color1
        for r in range(row_start, row_start + block_size):
            for c in range(col_start, col_start + block_size):
                grid[r, c] = object_color1
        
        # Determine the quadrant to color with object_color2 (opposite to the corner)
        if corner == (0, 0):  # Top-left corner
            quadrant_row = row_start + half_block
            quadrant_col = col_start + half_block
        elif corner == (0, 1):  # Top-right corner
            quadrant_row = row_start + half_block
            quadrant_col = col_start
        elif corner == (1, 0):  # Bottom-left corner
            quadrant_row = row_start
            quadrant_col = col_start + half_block
        else:  # (1, 1) Bottom-right corner
            quadrant_row = row_start
            quadrant_col = col_start
        
        # Fill the opposite quadrant with object_color2
        for r in range(quadrant_row, quadrant_row + half_block):
            for c in range(quadrant_col, quadrant_col + half_block):
                grid[r, c] = object_color2
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Get input dimensions and calculate output dimensions
        grid_size = taskvars['grid_size']
        block_size = grid_size - 1
        expanded_block_size = 2 * block_size
        output_size = 4 * block_size + 1
        object_color1 = taskvars['object_color1']
        
        # Create empty output grid
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Find the connected object to determine its position
        objects = find_connected_objects(grid, monochromatic=False)
        if len(objects) == 0:
            return output_grid  # Return empty grid if no object found
        
        # Get bounding box of the object
        block_obj = objects[0]
        bbox = block_obj.bounding_box
        
        # Determine which corner the block is positioned in
        is_top = bbox[0].start == 0
        is_left = bbox[1].start == 0
        corner_position = (0 if is_top else 1, 0 if is_left else 1)
        
        # Determine where to place the first expanded block
        first_row_start = 0 if corner_position[0] == 0 else output_size - expanded_block_size
        first_col_start = 0 if corner_position[1] == 0 else output_size - expanded_block_size
        
        # Place the first expanded block
        for r in range(first_row_start, first_row_start + expanded_block_size):
            for c in range(first_col_start, first_col_start + expanded_block_size):
                output_grid[r, c] = object_color1
        
        # Determine where to place the second expanded block so they only touch at one corner
        if corner_position == (0, 0):  # Top-left
            # Second block should touch at bottom-right corner of first block
            second_row_start = expanded_block_size
            second_col_start = expanded_block_size
        elif corner_position == (0, 1):  # Top-right
            # Second block should touch at bottom-left corner of first block
            second_row_start = expanded_block_size
            second_col_start = first_col_start - expanded_block_size
        elif corner_position == (1, 0):  # Bottom-left
            # Second block should touch at top-right corner of first block
            second_row_start = first_row_start - expanded_block_size
            second_col_start = expanded_block_size
        else:  # (1, 1) Bottom-right
            # Second block should touch at top-left corner of first block
            second_row_start = first_row_start - expanded_block_size
            second_col_start = first_col_start - expanded_block_size
        
        # Place the second expanded block
        for r in range(second_row_start, second_row_start + expanded_block_size):
            for c in range(second_col_start, second_col_start + expanded_block_size):
                output_grid[r, c] = object_color1
        
        return output_grid

