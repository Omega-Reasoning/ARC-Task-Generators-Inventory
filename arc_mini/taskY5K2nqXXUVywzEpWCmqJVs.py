# my_arc_task_generator.py

import random
import numpy as np
from typing import Dict, Any, Tuple
# Required imports from the framework and libraries
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, BorderBehavior
from transformation_library import GridObject, GridObjects
from input_library import retry  # Not strictly needed here, but available if desired

class TaskY5K2nqXXUVywzEpWCmqJVsGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Copy-paste the input reasoning chain as a list of strings
        self.input_reasoning_chain = [
            "Input grids can have different sizes.",
            "In each input grid, the first row contains two {color('cell_color1')} cells followed by one {color('cell_color2')} cell, while the second row has a single {color('cell_color3')} cell.",
            "The {color('cell_color1')} and {color('cell_color2')} cells in the first row are connected and positioned so that they never occupy the last cell of the row.",
            "The position of {color('cell_color1')}, {color('cell_color2')}, and {color('cell_color3')} cells varies across examples but remains within their defined rows."
        ]

        # 2) Copy-paste the transformation reasoning chain as a list of strings
        self.transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and only shifting the {color('cell_color1')} and {color('cell_color2')} cells to the right until they reach the edge of the grid.",
            "The {color('cell_color3')} cell remains in the same position in the output grid."
        ]

        # 3) Call the superclass constructor
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the specified input reasoning chain:
         - Grid size between 5x5 and 30x30.
         - First row: two cell_color1 cells followed by one cell_color2 cell, all connected,
           making sure there is at least one empty cell remaining to the right.
         - Second row: a single cell_color3 cell in a random column.
        """
        # Extract the colours from taskvars
        color1 = taskvars["cell_color1"]
        color2 = taskvars["cell_color2"]
        color3 = taskvars["cell_color3"]

        # Randomly choose grid dimensions (at least 5x5, up to 30x30)
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # We must place the pattern in the first row so that:
        #  * color1, color1, color2 appear contiguously
        #  * at least 1 cell remains empty to the right of this triple
        # The leftmost position for the triple can thus range up to (cols - 4)
        # because we need 3 colored cells plus at least 1 empty to the right.
        max_start = cols - 3 - 1
        c_start = random.randint(0, max_start)

        # Place color1, color1, color2 in row 0
        grid[0, c_start]     = color1
        grid[0, c_start + 1] = color1
        grid[0, c_start + 2] = color2

        # In the second row, place a single cell_color3 cell in a random column
        c3 = random.randint(0, cols - 1)
        grid[1, c3] = color3

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transforms the input grid by shifting {cell_color1} and {cell_color2} in the first row 
        to the right until they reach the edge, while keeping {cell_color3} in place.
        """
        # Extract the colors
        color1 = taskvars["cell_color1"]
        color2 = taskvars["cell_color2"]
        color3 = taskvars["cell_color3"]

        # Work on a copy of the input grid
        output_grid = grid.copy()

        # Get the number of columns
        n_cols = output_grid.shape[1]

        # Find the first row cells that belong to color1 and color2
        row0_indices = np.where((output_grid[0] == color1) | (output_grid[0] == color2))[0]

        if len(row0_indices) > 0:
            # Determine how far we can shift
            max_right_shift = n_cols - row0_indices[-1] - 1  # Distance to the last column

            if max_right_shift > 0:
                # Move the cells to the right
                for i in reversed(row0_indices):  # Move from right to left to avoid overwriting
                    output_grid[0, i + max_right_shift] = output_grid[0, i]
                    output_grid[0, i] = 0  # Clear old position

        return output_grid



    def create_grids(self) -> (dict, TrainTestData):
        """
        Pick distinct colors for cell_color1, cell_color2, cell_color3, then
        create 3-6 training examples and 1 test example.
        """
        # Choose three distinct colors from 1..9
        color1, color2, color3 = random.sample(range(1, 10), 3)
        taskvars = {
            "cell_color1": color1,
            "cell_color2": color2,
            "cell_color3": color3
        }

        # Randomly decide how many training pairs (3 to 6)
        nr_train = random.randint(3, 6)
        nr_test = 1  # Typically one test example

        # Use helper method to generate the data
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, train_test_data


# Below is optional test code if you'd like to run and visualize an example.
# It won't be called automatically in an ARC-AGI environment, but you can
# invoke it for debugging or demonstration purposes.


