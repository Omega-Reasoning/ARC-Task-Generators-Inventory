# my_arc_generator.py
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# Optional but encouraged: we can use our input_library / transformation_library if needed
# from input_library import create_object, retry, Contiguity
# from transformation_library import find_connected_objects, GridObject, GridObjects

class TaskA9KQWNurWPPwxye5H3bq5vGenerator(ARCTaskGenerator):

    def __init__(self):
        # 1) Input reasoning chain (exactly as provided):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid has a completely filled first row with {color('cell_color1')} and {color('cell_color2')} cells, while all other cells remain empty."
        ]
        
        # 2) Transformation reasoning chain (exactly as provided):
        transformation_reasoning_chain = [
            "The output grids are created by copying the input grid and completely filling the columns that contain a {color('cell_color1')} cell with {color('cell_color1')} color."
        ]
        
        # 3) Call to super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars,
                     gridvars):
        """
        Create an input grid according to the input reasoning chain given
        the task and grid variables.
        """

        rows = taskvars['rows']
        cols = taskvars['cols']
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']

        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Fill the first row with color1 and color2 in some pattern.
        #
        # We must ensure at least two cell_color1 and two cell_color2 appear,
        # and that the pattern changes from example to example.
        #
        # We'll randomly decide how many color1 vs color2 cells appear
        # but guarantee at least two of each.
        #
        # For instance, let's pick a random partition of the columns
        # into at least 2 of color1 and 2 of color2.

        # We want at least two cells of color1 and two cells of color2:
        # random split: number_of_color1 in [2, cols-2]
        n_color1 = random.randint(2, cols - 2)
        # fill these color1 cells in random positions in the first row
        # then fill the rest with color2

        # Let's create a list of column indices [0..cols-1], shuffle them, and pick the first n_color1 for color1
        col_indices = list(range(cols))
        random.shuffle(col_indices)
        color1_indices = set(col_indices[:n_color1])

        for c in range(cols):
            if c in color1_indices:
                grid[0, c] = color1
            else:
                grid[0, c] = color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:

        """
        Transform the input grid according to the transformation reasoning chain.
        Steps:
          1) Copy the input grid
          2) Identify which columns in the first row contain color1
          3) Fill those columns top-to-bottom with color1
        """
        color1 = taskvars['cell_color1']
        rows, cols = grid.shape
        output = grid.copy()

        # Find columns where the first row has color1
        columns_with_color1 = []
        for c in range(cols):
            if grid[0, c] == color1:
                columns_with_color1.append(c)

        # Fill these columns with color1 for all rows
        for c in columns_with_color1:
            output[:, c] = color1

        return output

    def create_grids(self) -> (dict, TrainTestData):
        """
        1) Create and return the dictionary of task variables used in
           the input and transformation reasoning chains.
        2) Create the train and test grids.
        """

        # We'll create 3 or 4 training examples and exactly 1 test example.
        nr_train = random.randint(3, 4)
        nr_test = 1

        

        # So let's pick them once:
        rows = random.randint(5, 10)
        cols = random.randint(5, 10)
        color1 = random.randint(1, 9)
        # pick a different color2
        color2 = random.choice([c for c in range(1, 10) if c != color1])

        # Our dictionary of variables
        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': color1,
            'cell_color2': color2
        }

        # We can now use our create_grids_default which calls create_input
        # and transform_input multiple times to form train/test pairs.
        # Each call to create_input will produce a different arrangement
        # in the first row because of randomization in create_input().
        # That will ensure we meet the requirement to vary the positions
        # of color1, color2 in the first row across examples.
        train_test_data = self.create_grids_default(nr_train_examples=nr_train,
                                                    nr_test_examples=nr_test,
                                                    taskvars=taskvars)

        return taskvars, train_test_data



