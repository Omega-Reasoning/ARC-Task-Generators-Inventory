# Filename: diagonal_fill_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task9NqP9sGUmXHdA69KKT9gGFGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a completely filled main diagonal (top-left to bottom-right) with multi-colored (1-9) cells, while the remaining cells are empty (0)."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling each row completely with the color of its corresponding main diagonal cell."
        ]
        
        # 3) Call parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid of size grid_size x grid_size,
        with the main diagonal (from top-left to bottom-right) filled with 
        randomly chosen colors (between 1 and 9) such that no two adjacent 
        diagonal cells have the same color. The rest of the grid is 0.
        """
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Fill the diagonal with random colors [1..9], ensuring
        # no two adjacent diagonal cells have the same color.
        previous_color = None
        for i in range(grid_size):
            # Pick a color different from the previous diagonal color
            available_colors = list(range(1, 10))
            if previous_color in available_colors:
                available_colors.remove(previous_color)
            color = random.choice(available_colors)
            
            grid[i, i] = color
            previous_color = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        According to the transformation reasoning chain:
        'The output grid is created by copying the input grid and filling
         each row completely with the color of its corresponding main diagonal cell.'
        """
        grid_size = taskvars['grid_size']
        output_grid = np.copy(grid)
        
        for row in range(grid_size):
            diag_color = grid[row, row]  # color on the diagonal
            output_grid[row, :] = diag_color
        
        return output_grid

    def create_grids(self):
        """
        Create the train/test data:
         - Randomly choose a grid_size in [5..30]
         - Generate 3-6 training examples
         - Generate 1 test example
        """
        # 1) Randomly select the grid size
        grid_size = random.randint(5, 30)
        
        # 2) Store task variables (used for template substitution)
        taskvars = {
            'grid_size': grid_size
        }
        
        # 3) Pick number of train examples (3-6) and create one test example
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1
        
        # 4) Use the parent helper to generate these examples
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        
        # Return the task variables and the train/test data
        return taskvars, train_test_data

