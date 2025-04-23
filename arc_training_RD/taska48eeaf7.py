from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, random_cell_coloring, enforce_object_width, enforce_object_height, retry
from transformation_library import find_connected_objects, GridObject, GridObjects
import numpy as np
import random

class ARCTaska48eeaf7Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have size {vars['rows']}x{vars['columns']}",
            "The grid contains one main block which is a 2x2 square in a color color_1(between 1-9) this block can be placed anywhere in the grid.",
            "The main block is surrounded by empty cells (0) in the grid and scattered cells (count <=5) of size 1x1 in color color_(between 1-9); these cells should not touch the main block.",
            "The scattered cells must be either horizontally, vertically or diagonally placed from the main block in any rows or columns."
        ]
        
        transformation_reasoning_chain = [
            "The output grid maintains the same dimensions as the input grid.",
            "All color_2 cells that are adjacent (either horizontally, vertically or diagonally) to the main block color_1 are attached to color_1 main block.",
            "The main color_1 block remains unchanged with respect to position, but will have attached color_2 cells besides them with respect to the position to which they were related to color_1 block."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        """Create an input grid with a main color_1 block and scattered color_2 cells."""
        #size = taskvars['size']  # Use only 'size' from taskvars
        rows = taskvars['rows']
        columns = taskvars['columns']
        color_1 = gridvars['color_1']
        color_2 = gridvars['color_2']
        scattered_count = gridvars['scattered_count']
        
        grid = np.zeros((rows, columns), dtype=int)
        
        # Create main color_1 block (2x2)
        main_block = np.full((2, 2), color_1)
        
        # Random position for main block (ensuring it fits)
        block_row = random.randint(0, rows - 2)
        block_col = random.randint(0, columns - 2)
        grid[block_row:block_row+2, block_col:block_col+2] = main_block
        
        # Generate positions for scattered cells that don't touch the main block
        def is_valid_scattered_pos(r, c):
            # Check if cell is adjacent (8-way) to main block
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if (block_row <= r + dr < block_row + 2 and 
                        block_col <= c + dc < block_col + 2):
                        return False
            return True
        
        # Generate scattered cells
        cells_added = 0
        attempts = 0
        max_attempts = 100
        
        while cells_added < scattered_count and attempts < max_attempts:
            r = random.randint(0, rows - 1)
            c = random.randint(0, columns - 1)
            
            # Check if cell is empty and not adjacent to main block
            if grid[r, c] == 0 and is_valid_scattered_pos(r, c):
                grid[r, c] = color_2
                cells_added += 1
            
            attempts += 1
        
        return grid

    def transform_input(self, grid, gridvars):
        """Transform input by merging adjacent color_2 cells into color_1 block."""
        output_grid = grid.copy()
        
        # Find all objects in the grid
        objects = find_connected_objects(output_grid, diagonal_connectivity=True)
        
        # Find the main color_1 block (should be only one)
        main_objects = objects.with_color(gridvars['color_1'])  # Use gridvars['color_1'] directly
        if len(main_objects) == 0:
            return output_grid  # No main block found
        
        main_block = main_objects[0]
        
        # Find all color_2 cells (1x1)
        scattered_objects = objects.with_color(gridvars['color_2']).with_size(min_size=1, max_size=1)
        
        # Find color_2 cells adjacent to main block (8-way)
        adjacent_scattered = []
        for obj in scattered_objects:
            if main_block.touches(obj, diag=True):
                adjacent_scattered.append(obj)
        return output_grid

    def create_grids(self):
        """Create training and test grids."""
        # Base task variables
        taskvars = {
            'rows': 10,  # Fixed size as per original requirements
      'columns': 10  # Replace semicolon with a comma
        }
        
         # Determine number of examples needed
        nr_train = random.randint(3, 5)
        nr_test = 1
        total_examples = nr_train + nr_test
        
        # Generate unique color pairs and scattered counts for each example
        all_params = []
        available_colors = list(range(1, 10))
        
        for _ in range(total_examples):
            # Pick two different random colors
            color_1 = random.choice(available_colors)
            available_colors.remove(color_1)
            
            remaining_colors = available_colors.copy()
            color_2 = random.choice(remaining_colors)
            
            scattered_count = random.randint(1, 5)
            
            all_params.append({
                'color_1': color_1,
                'color_2': color_2,
                'scattered_count': scattered_count
            })
            
            # Put color_1 back for next iterations
            available_colors.append(color_1)
            random.shuffle(available_colors)
        
        # Create training examples
        train_examples = []
        for i in range(nr_train):
            gridvars = all_params[i]
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_examples.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test example
        gridvars = all_params[-1]
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, gridvars)
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_examples, test=test_examples)