"""
SingleFilledColumnTaskGenerator

Generates ARC-AGI tasks where each input grid has exactly one filled column
(not the first column) using a specified color. The transformation then
fills all empty columns to the left of the filled column with that same color.

Usage:
  1. Instantiate the generator:
       generator = SingleFilledColumnTaskGenerator()
  2. Call create_grids() to get a set of training and test grids:
       taskvars, train_test_data = generator.create_grids()
  3. Optionally visualize:
       SingleFilledColumnTaskGenerator.visualize_train_test_data(train_test_data)
"""

import numpy as np
import random

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
# We can optionally import from transformation_library, but for this example,
# a simple manual approach is sufficient to fill columns.

class TaskXL7W8LmN2kTVYPztmzcN67Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input Reasoning Chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a single completely filled {color('object_color')} column, while all other cells remain empty (0).",
            "The filled column can be any column except the first column in the grid and varies across examples."
        ]
        
        # 2) Transformation Reasoning Chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling all empty (0) columns to the left of the filled column until the edge of the grid is reached.",
            "The empty (0) columns are filled with {color('object_color')} color."
        ]
        
        # 3) Call to superclass init
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Randomly initializes rows, cols, and object_color as task variables.
        Then creates train/test pairs with distinct filled-column positions.
        
        Returns:
            taskvars: Dict[str, Any]
                A dictionary containing 'rows', 'cols', and 'object_color'.
            train_test_data: TrainTestData
                Containing the train and test GridPairs.
        """
        # Randomly choose the puzzle-wide variables
        rows = random.randint(3, 10)
        cols = random.randint(7, 12)
        object_color = random.randint(1, 9)

        taskvars = {
            "rows": rows,
            "cols": cols,
            "object_color": object_color
        }

        # Decide how many training examples to create
        nr_train_examples = random.choice([3, 4])
        nr_test_examples = 1

        # We must pick distinct columns (other than 0) for each example
        # We need nr_train_examples + nr_test_examples distinct columns
        needed_columns = nr_train_examples + nr_test_examples
        possible_columns = list(range(1, cols))  # exclude col 0
        if needed_columns > len(possible_columns):
            # fallback: reduce nr_train_examples if we don't have enough distinct columns
            nr_train_examples = len(possible_columns) - 1
            needed_columns = nr_train_examples + nr_test_examples

        chosen_columns = random.sample(possible_columns, needed_columns)

        # Build the train GridPairs
        train_data = []
        for i in range(nr_train_examples):
            filled_col = chosen_columns[i]
            gridvars = {"filled_col": filled_col}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })

        # Build the test GridPairs
        test_data = []
        filled_col = chosen_columns[-1]
        gridvars = {"filled_col": filled_col}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_data.append({
            'input': input_grid,
            'output': output_grid
        })

        return taskvars, {
            "train": train_data,
            "test": test_data
        }

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid of shape (rows, cols) with all zeros except
        one column (given by gridvars['filled_col']) fully colored with
        taskvars['object_color'].
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        object_color = taskvars["object_color"]
        filled_col = gridvars["filled_col"]

        # Create the grid
        grid = np.zeros((rows, cols), dtype=int)

        # Fill the chosen column
        grid[:, filled_col] = object_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid by filling all empty (0) columns
        to the left of the existing filled column with object_color.
        """
        object_color = taskvars["object_color"]
        rows, cols = grid.shape

        # Identify the column that is fully filled with object_color
        # Because we know there's exactly one such column, we can find it by checking
        # which columns have all cells == object_color.
        col_sums = (grid == object_color).sum(axis=0)
        # There should be exactly one column where col_sums == rows
        filled_column = None
        for c in range(cols):
            if col_sums[c] == rows:
                filled_column = c
                break

        if filled_column is None:
            # Edge case: no filled column found (shouldn't happen in our generation)
            return grid.copy()

        # Create output by copying input
        output_grid = grid.copy()
        # Fill all columns from 0 up to but not including `filled_column`
        for c in range(filled_column):
            output_grid[:, c] = object_color

        return output_grid


