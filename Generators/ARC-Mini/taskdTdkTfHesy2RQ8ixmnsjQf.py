import numpy as np
import random
from typing import Dict, Any, Tuple
# Required imports from the framework/libraries
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects  # not strictly needed here, but example usage
from input_library import create_object, retry

class TaskdTdkTfHesy2RQ8ixmnsjQfGenerator(ARCTaskGenerator):
    def __init__(self):
        # Step 1: Input reasoning chain (exact strings from the specification):
        observation_chain = [
            "Input grids are of size {vars['rows']}x{vars['columns']}.",
            "Each input grid contains one or two multi-colored (1-9) cells in each row, with the remaining cells being empty (0)."
        ]

        # Step 2: Transformation reasoning chain (exact strings from the specification):
        reasoning_chain = [
            "To create the output grid, copy the input grid.",
            "Iterate through each row to find the first filled cell, and fill all empty (0) cells to the right of it until another filled cell is encountered or the row ends.",
            "Fill the empty (0) cells with {color('fill_color1')} color if it is an odd-numbered row and with {color('fill_color2')} color if it is an even-numbered row."
        ]

        # Step 3: Call super constructor
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        Creates the dictionary of task-level variables (rows, columns, fill_color1, fill_color2)
        and the train/test data (5 total examples: 4 train + 1 test).
        """
       
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        # 2) Randomly choose fill_color1 and fill_color2 in [1..9], ensuring they are different
        fill_color1 = random.randint(1, 9)
        fill_color2 = random.randint(1, 9)
        while fill_color2 == fill_color1:
            fill_color2 = random.randint(1, 9)

        # Store the chosen variables
        taskvars = {
            'rows': rows,
            'columns': cols,
            'fill_color1': fill_color1,
            'fill_color2': fill_color2
        }

        # 3) Generate train/test data: 4 train examples, 1 test example
        # Using the built-in helper method create_grids_default that uses create_input() -> transform_input().
        train_test_data = self.create_grids_default(nr_train_examples=4,
                                                    nr_test_examples=1,
                                                    taskvars=taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid according to the input reasoning chain:
          - The grid is rows x columns (from taskvars).
          - Each row has either 1 or 2 colored cells, the rest are empty (0).
          - Exactly 1 or 2 rows in the entire grid have exactly 1 colored cell (others have 2).
          - The colored cells have values 1..9.
          - Ensures that in at least one row, the leftmost colored cell is not in the last column (to avoid trivial transformations).
        """
        rows = taskvars['rows']
        cols = taskvars['columns']

        grid = np.zeros((rows, cols), dtype=int)

        # We must have 1 or 2 rows with a single colored cell; pick how many
        num_single_cell_rows = random.choice([1, 2])

        # Randomly pick which rows will have the single cell
        single_cell_rows = random.sample(range(rows), num_single_cell_rows)

        # Fill each row
        for r in range(rows):
            if r in single_cell_rows:
                # Place exactly 1 colored cell
                c = random.randint(0, cols - 1)
                grid[r, c] = random.randint(1, 9)
            else:
                # Place exactly 2 colored cells in different columns
                c1, c2 = random.sample(range(cols), 2)
                # Sort them so c1 < c2 (for convenience)
                if c1 > c2:
                    c1, c2 = c2, c1
                grid[r, c1] = random.randint(1, 9)
                grid[r, c2] = random.randint(1, 9)

        # We want to ensure that at least one row can be changed by the transformation:
        # i.e. in at least one row, the first filled cell is not in the last column.
        # If this never happens, we forcibly adjust at least one row.
        can_change = any(
            (np.argmax(grid[r] != 0) < (cols - 1))  # first filled cell is not last column
            for r in range(rows)
        )
        if not can_change:
            # Force the first row to have its first fill not in the last column
            # e.g. pick a column < cols-1 for the first filled cell
            if single_cell_rows and single_cell_rows[0] == 0:
                # If the 0th row was singled out for 1 color,
                # we just move that single color to a random col < cols-1
                c = random.randint(0, cols - 2)
                grid[0] = 0  # reset
                grid[0, c] = random.randint(1, 9)
            else:
                # For the 0th row which might have 2 colored cells,
                # ensure the leftmost is < cols-1
                # Clear that row and fill again
                grid[0] = 0
                c1 = random.randint(0, cols - 2)
                c2 = random.randint(c1 + 1, cols - 1)
                grid[0, c1] = random.randint(1, 9)
                grid[0, c2] = random.randint(1, 9)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
          1) Copy the input grid to output.
          2) For each row, find the first filled (non-zero) cell.
             Fill all empty cells (0) to the right, up until we encounter another filled cell or reach the end of the row.
          3) Fill color is fill_color1 if row index is even, fill_color2 if row index is odd.
        """
        rows, cols = grid.shape
        fill_color1 = taskvars['fill_color1']
        fill_color2 = taskvars['fill_color2']

        output = np.copy(grid)

        for r in range(rows):
            # find the first filled (non-zero) cell in row r
            row_data = output[r, :]
            non_zero_cols = np.where(row_data != 0)[0]
            if len(non_zero_cols) == 0:
                # no filled cells in this row, do nothing
                continue
            # first filled cell
            first_filled_col = non_zero_cols[0]

            # choose row color
            row_color = fill_color1 if (r % 2 == 0) else fill_color2

            # fill empty cells from (first_filled_col+1) onward until next filled or end
            for c in range(first_filled_col + 1, cols):
                if output[r, c] != 0:
                    # we've encountered another colored cell; stop filling
                    break
                # fill it
                output[r, c] = row_color

        return output


