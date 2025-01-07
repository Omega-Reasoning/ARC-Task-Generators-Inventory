# my_arc_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task5amY6LAtAAMJLViM9S922SGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (copy from provided prompt):
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input matrix has all rows filled with {color('object_color1')} color, except the last row, which is filled with {color('object_color2')} color."
        ]

        # 2) Transformation reasoning chain (copy from provided prompt):
        reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and filling the entire last column with {color('object_color2')} color."
        ]

        # 3) The call to super().__init__
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        Initialise task variables used in templates and create train/test data grids.
        
        We:
         - choose object_color1 and object_color2 randomly, ensuring they differ
         - generate 3-6 training examples and 1 test example
         - keep grid sizes within the 5..30 range
        """
        # Step 1: choose two distinct colors
        all_colors = list(range(1, 10))  # skipping 0 because 0 = empty background
        object_color1 = random.choice(all_colors)
        possible_object_color2 = [c for c in all_colors if c != object_color1]
        object_color2 = random.choice(possible_object_color2)

        # Put them in a dictionary so we can fill the template references {color('object_color1')} etc.
        taskvars = {
            'object_color1': object_color1,
            'object_color2': object_color2,
        }

        # Step 2: decide how many training examples we want
        num_train = random.randint(3, 6)
        num_test = 1  # as requested

        # We'll store (train/test) pairs
        train_grids = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_grids.append({'input': input_grid, 'output': output_grid})

        test_grids = []
        for _ in range(num_test):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_grids.append({'input': input_grid, 'output': output_grid})

        train_test_data = {
            'train': train_grids,
            'test': test_grids
        }

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid based on the observation chain:
          "Each input matrix has all rows filled with color('object_color1'), except
           the last row which is filled with color('object_color2')."
        """
        # Step 1: random grid size between 5..30 (both dimensions)
        height = random.randint(5, 30)
        width = random.randint(5, 30)

        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']

        # Step 2: fill all rows except last with object_color1, last row with object_color2
        grid = np.full((height, width), object_color1, dtype=int)
        grid[-1, :] = object_color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars):
        """
        Transform the input grid according to the transformation reasoning chain:
         "The output matrix is constructed by copying the input matrix and
          filling the entire last column with object_color2."
        """
        object_color2 = taskvars['object_color2']
        output_grid = np.copy(grid)

        # Fill the last column with object_color2
        output_grid[:, -1] = object_color2

        return output_grid

