from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring
import numpy as np
import random

class Taskf5b8619d(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "In each input grid, a single random color is selected, and a random number of columns are chosen. In each of these columns, one cell is colored with the selected color and placed at a random row within that column."
        ]
        
        transformation_reasoning_chain = [
            "Input grids are of size {2*vars['n']} x {2*vars['n']}.",
            "The output grid is constructed by tiling the input grid in a 2×2 arrangement, duplicating it once horizontally and once vertically.",
            "For each column in the output grid that contains single-colored cells, all empty cells in that column are filled with {color('fill_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Generate task variables
        taskvars = {
            'n': random.randint(5, 15),  # Keep original grid small so 2x2 tiling doesn't exceed 30x30
            'fill_color': random.randint(1, 9)
        }
        
        # Generate 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        train_examples = []
        test_examples = []
        
        for i in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars, gridvars):
        n = taskvars['n']
        grid = np.zeros((n, n), dtype=int)
        
        # Select a random color (not 0 and not fill_color to avoid confusion)
        available_colors = [c for c in range(1, 10) if c != taskvars['fill_color']]
        selected_color = random.choice(available_colors)
        
        # Choose random number of columns (at least 1, at most n)
        num_columns = random.randint(1, min(n, 5))  # Limit to avoid too dense grids
        chosen_columns = random.sample(range(n), num_columns)
        
        # Place one cell of the selected color in each chosen column
        for col in chosen_columns:
            row = random.randint(0, n - 1)
            grid[row, col] = selected_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        n = taskvars['n']
        fill_color = taskvars['fill_color']
        
        # Create 2x2 tiled grid
        tiled_grid = np.zeros((2*n, 2*n), dtype=int)
        
        # Tile the input grid in 2x2 arrangement
        tiled_grid[0:n, 0:n] = grid      # Top-left
        tiled_grid[0:n, n:2*n] = grid    # Top-right
        tiled_grid[n:2*n, 0:n] = grid    # Bottom-left
        tiled_grid[n:2*n, n:2*n] = grid  # Bottom-right
        
        # For each column in the tiled grid that contains non-zero cells,
        # fill all empty cells in that column with fill_color
        for col in range(2*n):
            column_data = tiled_grid[:, col]
            if np.any(column_data != 0):  # Column contains colored cells
                # Fill all empty cells in this column
                empty_cells = column_data == 0
                tiled_grid[empty_cells, col] = fill_color
        
        return tiled_grid

