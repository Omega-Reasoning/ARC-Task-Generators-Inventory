from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optional but encouraged imports from the libraries:
# (We do not use these here, but you could use them for more variety if desired)
# from input_library import create_object, Contiguity, retry
# from transformation_library import find_connected_objects, GridObject, GridObjects

class TaskQ27925pdC3wgd2ggZfQiXhGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each row of the input grid contains two differently colored (1-9) cells, while the remaining cells are empty (0).",
            "The first colored cell always appears in the first column, with the second colored cell placed at least two empty (0) cells apart."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and shifting the first colored cell from the first column to the second column.",
            "The second colored cell is moved to the third column, connecting both colored cells."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        We create random 'rows' and 'cols' between 6 and 30 and store them in taskvars.
        Then we generate 3-6 training pairs and 1 test pair using create_grids_default.
        This ensures variety if the generator is called multiple times.
        """
        rows = random.randint(6, 30)
        cols = random.randint(6, 30)
        nr_train = random.randint(3, 6)

        taskvars = {
            'rows': rows,
            'cols': cols
        }

        train_test_data = self.create_grids_default(nr_train_examples=nr_train,
                                                    nr_test_examples=1,
                                                    taskvars=taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Creates an input grid of shape (rows x cols) where each row has:
          * The first colored cell in column 0.
          * A second colored cell in a column >= 3.
          * Both colors are distinct from each other (1..9).
          * All other cells are empty (0).
        The row-by-row choice of colors is random, ensuring variety.
        """

        rows = taskvars['rows']
        cols = taskvars['cols']

        grid = np.zeros((rows, cols), dtype=int)

        for r in range(rows):
            # Pick two distinct colors in [1..9]
            color1 = random.randint(1, 9)
            color2 = random.randint(1, 9)
            while color2 == color1:
                color2 = random.randint(1, 9)

            # Place color1 in column 0
            grid[r, 0] = color1

            # Place color2 in a column >= 3, ensuring at least two empty cells from col 0
            second_col = random.randint(3, cols - 1)
            grid[r, second_col] = color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        According to the transformation reasoning chain:
          1) Copy the input grid to output.
          2) Shift the first colored cell from column 0 to column 1.
          3) Shift the second colored cell from wherever it is to column 2.
        Thus, each row ends up with two adjacent colored cells in columns 1 and 2.
        """
        rows = grid.shape[0]
        cols = grid.shape[1]
        output_grid = grid.copy()

        # For each row, find the two colored cells, move them to col=1 and col=2
        for r in range(rows):
            # Identify colored cells in this row (excluding 0)
            # We expect exactly two colored cells, with one at col=0 and the other at col>=3
            colored_cols = np.where(output_grid[r] != 0)[0]  # Indices of non-zero columns
            if len(colored_cols) == 2:
                c1, c2 = colored_cols
                color1 = output_grid[r, c1]
                color2 = output_grid[r, c2]

                # Erase them from current positions
                output_grid[r, c1] = 0
                output_grid[r, c2] = 0

                # Move them to columns 1 and 2
                output_grid[r, 1] = color1
                output_grid[r, 2] = color2

        return output_grid

