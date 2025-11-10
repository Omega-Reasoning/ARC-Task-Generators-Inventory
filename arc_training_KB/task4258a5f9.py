from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects, BorderBehavior

class Task4258a5f9Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain only {color('object_color')} and empty (0) cells.",
            "Each {color('object_color')} cell is surrounded by a one-cell wide frame of empty (0) cells.",
            "No two {color('object_color')} cells share the same frame, ensuring complete separation."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and coloring the one-cell wide empty (0) frames around each {color('object_color')} cell with {color('fill_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(7, 15),  # Keeping grid size reasonable for visibility
            'object_color': random.randint(1, 9),
            'fill_color': 0  # Will be set to a valid value later
        }
        
        # Ensure object_color and fill_color are different
        while taskvars['fill_color'] == 0 or taskvars['fill_color'] == taskvars['object_color']:
            taskvars['fill_color'] = random.randint(1, 9)
        
        # Generate 3-4 training examples (total examples will be train + 1 test)
        num_train_examples = random.randint(3, 4)

        # We need strictly different numbers of colored cells across all grids
        total_examples = num_train_examples + 1  # include test
        # Choose unique counts in the range 1..5
        counts = random.sample(range(1, 6), k=total_examples)

        train_examples = []
        for i in range(num_train_examples):
            num_cells = counts[i]
            input_grid = self.create_input(taskvars, {'num_cells': num_cells})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        # Generate one test example with a count different from all training examples
        test_num_cells = counts[-1]
        test_input = self.create_input(taskvars, {'num_cells': test_num_cells})
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        # Create an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Desired number of colored cells (must be provided by caller to enforce uniqueness);
        # fall back to a safe random choice in 1..5 if not provided.
        desired_num = gridvars.get('num_cells', None)
        if desired_num is None:
            num_cells = random.randint(1, 5)
        else:
            num_cells = int(desired_num)
        
        # Function to check if a position is valid for placing a colored cell
        def is_position_valid(grid, row, col):
            height, width = grid.shape
            
            # Check 5x5 area around the cell to ensure no overlap with other cells or their frames
            # This ensures at least 2 cell distance between any two colored cells
            min_row, max_row = max(0, row-2), min(height, row+3)
            min_col, max_col = max(0, col-2), min(width, col+3)
            
            area = grid[min_row:max_row, min_col:max_col]
            return np.all(area == 0)
        
        # Place colored cells
        cells_placed = 0
        max_attempts = 1000
        attempts = 0
        
        while cells_placed < num_cells and attempts < max_attempts:
            attempts += 1
            
            # Choose a random position, avoiding edges
            row = random.randint(2, grid_size - 3)
            col = random.randint(2, grid_size - 3)
            
            if is_position_valid(grid, row, col):
                grid[row, col] = object_color
                cells_placed += 1
        
        # If we couldn't place the desired number of cells, grow the grid and retry a few times
        if cells_placed < num_cells:
            # Try increasing the grid size to give more room (but keep it reasonable)
            if grid_size < 15:
                taskvars['grid_size'] = min(15, grid_size + 2)
                return self.create_input(taskvars, gridvars)
            # As a final fallback, try again (this is unlikely for 1..5 cells)
            return self.create_input(taskvars, gridvars)

        return grid

    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        fill_color = taskvars['fill_color']
        
        # Make a copy of the input grid
        output_grid = grid.copy()
        
        # Find all object cells
        height, width = grid.shape
        for r in range(height):
            for c in range(width):
                if grid[r, c] == object_color:
                    # Fill the 8-connected surrounding cells with the fill color
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip the center cell
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == 0:
                                output_grid[nr, nc] = fill_color
        
        return output_grid

