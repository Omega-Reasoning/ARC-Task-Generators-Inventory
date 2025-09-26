"""
ARC-AGI Task Generator

This generator creates tasks where the first row of each input grid has a set of 
multi-colored cells and all other cells are empty. The transformation fills
entire columns (top to bottom) with the color found in that column of the first row.
"""


import random
import numpy as np

# These imports are required by the instructions:
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
# (Optional) We could import from input_library.py or transformation_library.py if needed
# e.g. from input_library import ...
# e.g. from transformation_library import ...

class TaskPXtjohvsRk7hJTcxMXoQ6tGenerator(ARCTaskGenerator):
    def __init__(self):
        # Exactly 3 statements to initialize:
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "The first row of each input grid contains {int(vars['cols']/2)} multi-colored (1-9) cells, while all other cells remain empty (0).",
            "The color and position of these cells vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are created by copying the input grid and filling all columns that contain a colored cell with the same color as the original cell in the first row."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Initialize the grid with zeros
        grid = np.zeros((rows, cols), dtype=int)

        # Number of colored cells in the first row
        num_colored = cols//2
        
        # Randomly choose the columns to color in the first row
        chosen_cols = random.sample(range(cols), num_colored)

        # Randomly assign colors (1..9) to those columns.
        # Ensure at least two distinct colors if num_colored > 1.
        while True:
            colors = [random.randint(1, 9) for _ in range(num_colored)]
            if num_colored == 1 or (len(set(colors)) > 1):
                break
        
        for c, color in zip(chosen_cols, colors):
            grid[0, c] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
       
        out_grid = grid.copy()
        rows, cols = out_grid.shape
        
        for c in range(cols):
            col_color = out_grid[0, c]
            if col_color != 0:
                # Fill this entire column in the output
                out_grid[:, c] = col_color
        return out_grid

    def create_grids(self):
       
        # Randomly choose the grid size within the specified invariants
        rows = random.randint(5, 30)   # between 5 and 30
        cols = random.randint(4, 23)   # between 4 and 23
        cols = cols - (cols % 2)  # make it even
        
        taskvars = {
            'rows': rows,
            'cols': cols
        }
        
        # Decide how many training examples we want (3 or 4)
        nr_train = random.choice([3, 4])
        nr_test = 1  # We only create 1 test example as per instructions
        
        # Use the default creation loop in ARCTaskGenerator
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        
        return taskvars, train_test_data



