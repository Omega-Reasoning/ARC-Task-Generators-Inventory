# example_arc_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import Contiguity, create_object, retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task9TagX93bq5oSUjtZyGtpu9Generator(ARCTaskGenerator):
    def __init__(self):
        # Input Reasoning Chain
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains at most one colored (1-9) cell in each column.",
            "Each colored cell is of a different color.",
            "The remaining cells are empty (0)."
        ]

        # Transformation Reasoning Chain
        reasoning_chain = [
            "To construct the output grid, copy the input grid.",
            "Iterate through each column and if a filled cell is found, fill the entire column with the same color as the filled cell."
        ]

        super().__init__(observation_chain, reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create a random input grid with:
         * Size between 3x3 and 9x9
         * At most one colored (1-9) cell in each column
         * Different colors in each filled column
         * The row of each colored cell is different from the row of a colored cell in another column
        """
        # Randomize grid size
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        grid = np.zeros((height, width), dtype=int)

        # Decide how many columns will contain a colored cell (at least 1)
        nr_colored = random.randint(1, min(width, height, 9))

        # Pick distinct columns to color
        colored_columns = random.sample(range(width), nr_colored)
        # Pick distinct row positions for these columns
        # (ensures the row number for these colored cells is different for each chosen column)
        colored_rows = random.sample(range(height), nr_colored)
        # Pick distinct colors (1-9)
        possible_colors = list(range(1, 10))
        random.shuffle(possible_colors)
        chosen_colors = possible_colors[:nr_colored]

        # Place one colored cell per selected column
        for col, row, color in zip(colored_columns, colored_rows, chosen_colors):
            grid[row, col] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transformation:
        1) Copy the grid
        2) For each column, if a filled cell is found, fill that entire column with the same color
        """
        # Make a copy
        output = grid.copy()

        rows, cols = grid.shape
        for c in range(cols):
            # Find any colored cell in column c
            column_slice = grid[:, c]
            # Non-zero entries
            nonzero_positions = np.where(column_slice != 0)[0]
            if len(nonzero_positions) > 0:
                # Suppose the column has exactly one non-zero cell by construction
                color = column_slice[nonzero_positions[0]]
                # Fill entire column
                output[:, c] = color

        return output

    def create_grids(self) -> (dict, TrainTestData):
        """
        Create 4 train examples and 1 test example using create_grids_default
        (i.e., each example calls create_input() -> transform_input()).
        No particular taskvars are needed here, so we return an empty dictionary.
        """
        # We do not need to define special variables for the templates, so pass empty dict.
        taskvars = {}

        # Usually, we want 4 training pairs and 1 test pair
        nr_train_examples = 4
        nr_test_examples = 1

        # create_grids_default will call create_input + transform_input multiple times
        data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return {}, data

