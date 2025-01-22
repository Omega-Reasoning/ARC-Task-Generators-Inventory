# single_column_fill_to_row_fill_generator.py

import numpy as np
import random

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import Contiguity, create_object, retry
# We don't actually need transformation_library here, but it's available if needed.
from typing import Dict, Any, Tuple
class TaskFCUY2yh6ka8QS6WWHYZ7Un_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid contains a single, completely filled column, with same-colored cells. The remaining cells are empty.",
            "The color of this column can only be {color('object_color1')}, {color('object_color2')}, or {color('object_color3')}.",
            "The position of the filled column varies across examples."
        ]
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by completely filling either the first row or the last row, based on the input grid, using same-colored cells.",
            "If the filled column in the input grid is in the left half, the first row of the output grid is filled; otherwise, the last row is filled.",
            "The color choice for the output grid is based on the input grid; {color('object_color1')} → {color('object_color4')}, {color('object_color2')} → {color('object_color5')}, {color('object_color3')} → {color('object_color6')}."

        ]
        # 3) Call the superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid of shape (rows x cols) with exactly one filled column.
        The column's color is one of the three object colors from the taskvars.
        The column index is chosen to be in the left half or right half, based on gridvars['half'].
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        color = gridvars["column_color"]
        half = gridvars["half"]  # "left" or "right"

        # Initialize the grid
        grid = np.zeros((rows, cols), dtype=int)

        # Choose a column index in the left or right half
        # left half columns => [0 .. (cols//2 - 1)], right half => [(cols//2) .. (cols - 1)]
        if half == "left":
            col_idx = random.randint(0, (cols // 2) - 1)
        else:  # "right"
            col_idx = random.randint(cols // 2, cols - 1)

        # Fill that entire column with the chosen color
        grid[:, col_idx] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation chain:
        - Check if the filled column is in the left half or right half of the grid.
        - If left half, fill the first row in the output with the corresponding color.
        - If right half, fill the last row in the output with the corresponding color.
        - Color mapping: object_color1->object_color4, object_color2->object_color5, object_color3->object_color6.
        """
        # --- We'll define the transform using the variables from taskvars below ---
        # The final code that is stored in the ARCTask will be partially evaluated.

        rows, cols = grid.shape
        # Identify which column is filled and the input color
        # Because there's only one fully filled column, we can find it by summation or 'any(axis=0)'
        filled_cols = np.where(grid.any(axis=0))[0]
        if len(filled_cols) == 0:
            # Edge case: if no filled column was found (shouldn't happen in our generation), return input as is
            return grid.copy()
        col_idx = filled_cols[0]  # We only expect one such column
        input_color = grid[0, col_idx]

        # Prepare color mapping
        color_map = {
            taskvars["object_color1"]: taskvars["object_color4"],
            taskvars["object_color2"]: taskvars["object_color5"],
            taskvars["object_color3"]: taskvars["object_color6"]
        }
        # Determine if column is in left or right half
        # Use integer division to handle any fraction
        half_threshold = cols // 2
        out_grid = np.zeros_like(grid)

        if col_idx < half_threshold:
            # Fill the first row
            out_grid[0, :] = color_map[input_color]
        else:
            # Fill the last row
            out_grid[rows - 1, :] = color_map[input_color]

        return out_grid

    def create_grids(self) -> tuple:
        """
        Create the task variables (rows, cols, object_color1..6) and the training/test grids.
        We must ensure:
          - rows in [5..30], cols even in [7..30].
          - object_color1..object_color6 are distinct in [1..9].
          - 3-6 training examples, 2 test examples.
          - One training and testing example with column in left half, one with column in right half.
          - Training must cover columns of color1, color2, color3 (at least once each).
        """
        # 1) Choose the task variables
        # rows
        rows = random.randint(5, 30)

        # cols (even, between 7 and 30)
        possible_even_cols = [c for c in range(7, 31) if c % 2 == 0]
        cols = random.choice(possible_even_cols)

        # Distinct colors for object_color1..6
        # each must be between 1 and 9
        color_choices = random.sample(range(1, 10), 6)
        object_color1, object_color2, object_color3, object_color4, object_color5, object_color6 = color_choices

        taskvars = {
            "rows": rows,
            "cols": cols,
            "object_color1": object_color1,
            "object_color2": object_color2,
            "object_color3": object_color3,
            "object_color4": object_color4,
            "object_color5": object_color5,
            "object_color6": object_color6
        }

        # 2) Build the training data
        # We must have at least one training grid with each of the input colors (color1, color2, color3).
        # We also need at least one training grid in left half and one in right half.
        # We'll create exactly 3 training examples to satisfy the constraints:
        #   1) (object_color1, left)
        #   2) (object_color2, right)
        #   3) (object_color3, left or right). We'll pick randomly for a bit more diversity.
        left_or_right_for_third = random.choice(["left", "right"])

        train_specs = [
            (object_color1, "left"),
            (object_color2, "right"),
            (object_color3, left_or_right_for_third)
        ]

        train_data = []
        for color_val, half in train_specs:
            gridvars = {
                "column_color": color_val,
                "half": half
            }
            in_grid = self.create_input(taskvars, gridvars)
            out_grid = self.transform_input(in_grid, taskvars)
            train_data.append(GridPair(input=in_grid, output=out_grid))

        # 3) Build the test data
        # We want 2 test examples, one with the column in the left half, one in the right half.
        # Also choose which of the 3 input colors to use for each test.
        # For variety, let's pick two distinct colors among color1..color3.
        test_colors = random.sample([object_color1, object_color2, object_color3], 2)
        test_specs = [
            (test_colors[0], "left"),
            (test_colors[1], "right")
        ]

        test_data = []
        for color_val, half in test_specs:
            gridvars = {
                "column_color": color_val,
                "half": half
            }
            in_grid = self.create_input(taskvars, gridvars)
            # Test outputs are typically withheld or left blank in ARC, but here we produce them
            # so we can visualize or confirm correctness.
            out_grid = self.transform_input(in_grid, taskvars)
            test_data.append(GridPair(input=in_grid, output=out_grid))

        train_test_data: TrainTestData = {
            "train": train_data,
            "test": test_data
        }

        return taskvars, train_test_data



