from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optional: These imports are available but not strictly needed for this example
# from transformation_library import find_connected_objects, ...
# from input_library import create_object, retry, ...

class TaskW8tGS5zKCegimNhtyiabHDGenerator(ARCTaskGenerator):
    def __init__(self):
        # Observation chain and reasoning chain (from the problem statement)
        input_reasoning_chain = [
            "Input grids are of size 1x{vars['col']}.",
            "Each input grid contains colored (1-9) cells in the entire row."
        ]
        transformation_reasoning_chain = [
            "The output grids are of size {vars['row']}x{vars['col']}.",
            "The transformation seems to be simply repeating the single input row, {vars['row']} times to create the output grid."
        ]
        
        # Call superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Choose random row and col (both between 3 and 7), then use create_grids_default 
        to produce 3 training pairs and 1 test pair.
        """
        row = random.randint(5, 30)
        col = random.randint(5, 30)
        
        taskvars = {
            "row": row,
            "col": col
        }
        
        # Create train/test data with 3 train examples and 1 test example
        train_test_data = self.create_grids_default(nr_train_examples=3,
                                                    nr_test_examples=1,
                                                    taskvars=taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create a 1 x col grid with multi-colored cells (no zeros). 
        Make sure at least 2 different colors are present.
        """
        col = taskvars['col']
        
        # Generate the row until it has at least two distinct non-zero colors
        while True:
            row_data = [random.randint(1, 9) for _ in range(col)]
            if len(set(row_data)) > 1:
                break
        
        # Convert to numpy array of shape (1, col)
        input_grid = np.array([row_data], dtype=int)
        return input_grid

    def transform_input(self, grid, taskvars):
        """
        Repeat the single input row 'row' times to form a row x col output grid.
        """
        row = taskvars['row']
        col = taskvars['col']  # Not strictly needed except for clarity
        
        # grid is of shape (1, col)
        # Repeat it 'row' times along axis=0
        output_grid = np.tile(grid, (row, 1))
        return output_grid


