from typing import Any, Dict, Tuple
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from input_library import Contiguity, create_object, random_cell_coloring

class Task0520fde7Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Two rectangular subgrids each contain {color('color_main')} and empty cells.",
            "A vertical bar of {color('color_vertical_separator')} cells separates these subgrids.",
            "The size of each subgrid is {vars['subgrid_rows']} rows by {vars['subgrid_cols']} columns."
        ]
        transformation_reasoning_chain = [
            "The output matrix has the same dimension as each of the subgrids.",
            "Cells that are {color('color_main')} in both rectangular subgrids (logical AND) are painted {color('color_output')}.",
            "All other cells remain empty in the output matrix."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        This method sets up task-wide variables such as the subgrid size, the colors, etc.
        Then it generates train and test pairs by calling create_input() and transform_input().
        It returns (taskvars, train_test_data).
        """
        # 1) Create random task variables
        subgrid_rows = random.randint(3, 10)
        subgrid_cols = random.randint(3, 10)

        # Choose distinct non-zero colors (vertical bar vs. the subgrid color vs. output color)
        available_colors = list(range(1, 10))  # 1 to 9
        color_vertical_separator = random.choice(available_colors)
        available_colors.remove(color_vertical_separator)
        color_main = random.choice(available_colors)
        available_colors.remove(color_main)
        color_output = random.choice(available_colors)

        taskvars = {
            "subgrid_rows": subgrid_rows,
            "subgrid_cols": subgrid_cols,
            "color_vertical_separator": color_vertical_separator,
            "color_main": color_main,
            "color_output": color_output
        }

        # 2) Create train/test data. 
        nr_train = random.randint(3, 6)
        nr_test = 1
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        
        return taskvars, train_test_data

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Creates an input matrix with two subgrids of size subgrid_rows x subgrid_cols, 
        separated by a vertical bar of color_vertical_separator.
        """
        rows = taskvars["subgrid_rows"]
        cols = taskvars["subgrid_cols"]
        # Create two empty subgrids
        left = np.zeros((rows, cols), dtype=int)
        right = np.zeros((rows, cols), dtype=int)

        # Choose a number of matching positions (at least 2) that will be set
        # to color_main in both subgrids so the transformed output has >= 2
        max_match = min(rows * cols, 4)
        n_matches = random.randint(2, max_match)

        # Pick distinct cell positions
        all_positions = [(r, c) for r in range(rows) for c in range(cols)]
        match_positions = random.sample(all_positions, n_matches)

        for (r, c) in match_positions:
            left[r, c] = taskvars["color_main"]
            right[r, c] = taskvars["color_main"]

        # Optionally sprinkle additional color_main cells independently in each subgrid
        # to make examples more varied, without breaking the matching constraint.
        density = 0.25
        left = random_cell_coloring(left, taskvars["color_main"], density=density, background=0, overwrite=False)
        right = random_cell_coloring(right, taskvars["color_main"], density=density, background=0, overwrite=False)

        # Ensure the matching positions remain colored (random_cell_coloring won't overwrite them,
        # but enforce again to be safe)
        for (r, c) in match_positions:
            left[r, c] = taskvars["color_main"]
            right[r, c] = taskvars["color_main"]

        # Create separator and concatenate all parts
        separator = np.full((rows, 1), taskvars["color_vertical_separator"], dtype=int)
        return np.concatenate((left, separator, right), axis=1)

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        # Find the separator column using boolean indexing
        separator_col = np.where(np.all(grid == taskvars["color_vertical_separator"], axis=0))[0][0]
        
        # Split into left and right subgrids
        left_subgrid = grid[:, :separator_col]
        right_subgrid = grid[:, separator_col + 1:]
        
        # Create output using boolean operations
        return np.where(
            (left_subgrid == taskvars["color_main"]) & 
            (right_subgrid == taskvars["color_main"]),
            taskvars["color_output"],
            0
        )
