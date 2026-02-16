

import random
import numpy as np

# Required imports from the specification
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, Contiguity, retry
from Framework.transformation_library import find_connected_objects

class TaskHvvoLDKhDwDkuv4pGmu2hu_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain four completely filled columns of {color('object_color1')} color.",
            "The {color('object_color1')} columns are arranged so that at least two are separated by exactly one empty (0) column, while two others are separated by more than one empty (0) column."
        ]

        # 2. Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling all empty (0) columns with {color('object_color2')} color if they are positioned exactly between two {color('object_color1')} columns.",
            "If multiple empty (0) columns exist between two {color('object_color1')} columns, they remain unchanged."
        ]

        # 3. Call parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates and returns task variables plus the train/test data.

        We pick object_color1 and object_color2 (1..9, different from each other).
        Then produce 3-6 training examples, each with a unique size, plus 1 test example.
        """
        # Choose two distinct colors for object_color1 and object_color2
        color1 = random.randint(1, 9)
        while True:
            color2 = random.randint(1, 9)
            if color2 != color1:
                break

        taskvars = {
            "object_color1": color1,
            "object_color2": color2,
            # We'll store used_sizes here to ensure all grids have distinct sizes
            "used_sizes": set()
        }

        # Randomly decide how many training examples (3 to 6)
        nr_train = random.randint(3, 6)
        nr_test = 1

        # Use the default approach to generate train/test pairs,
        # which will call create_input() and transform_input().
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)

        # Remove 'used_sizes' from the final dictionary since it's an internal helper
        del taskvars["used_sizes"]

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid following the requirements:
        1) Pick a random size [12..30 x 12..30] not used so far.
        2) Fill exactly four columns fully with object_color1.
        3) Ensure among the consecutive differences of these columns, there is
           at least one exactly 2 (i.e., exactly one empty column between them)
           and at least one >= 3 (more than one empty column).
        """
        color1 = taskvars["object_color1"]
        used_sizes = taskvars["used_sizes"]

        min_dim, max_dim = 8, 30
        # Pick a random size not used before
        while True:
            H = random.randint(min_dim, max_dim)
            W = random.randint(min_dim, max_dim)
            if (H, W) not in used_sizes:
                used_sizes.add((H, W))
                break

        # Create the empty grid
        grid = np.zeros((H, W), dtype=int)

        # We'll pick 4 distinct columns for color1 such that:
        # among (c2-c1, c3-c2, c4-c3), we have at least one difference == 2 and one >= 3
        def valid_columns(cols):
            cols = sorted(cols)
            diffs = [cols[i+1] - cols[i] for i in range(len(cols) - 1)]
            has_diff_2 = any(d == 2 for d in diffs)
            has_diff_ge3 = any(d >= 3 for d in diffs)
            return has_diff_2 and has_diff_ge3

        while True:
            columns_chosen = random.sample(range(W), 4)
            if valid_columns(columns_chosen):
                columns_chosen.sort()
                break

        # Fill these columns fully with color1
        for col_idx in columns_chosen:
            grid[:, col_idx] = color1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid to the output grid following the transformation steps:

        1) Copy the input.
        2) Identify all columns fully filled with object_color1.
        3) For each consecutive pair of color1 columns, if they differ by exactly 2
           in column index, fill the middle column with object_color2.
        4) Otherwise, leave multiple-gap columns as is.
        """
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]

        out_grid = grid.copy()
        H, W = out_grid.shape

        # Find which columns are fully color1
        color1_columns = []
        for col in range(W):
            if np.all(out_grid[:, col] == color1):
                color1_columns.append(col)

        color1_columns.sort()
        # For each consecutive pair of color1 columns, check if difference == 2
        # If so, fill that in-between column with color2
        for i in range(len(color1_columns) - 1):
            c_left = color1_columns[i]
            c_right = color1_columns[i + 1]
            if c_right - c_left == 2:
                out_grid[:, c_left + 1] = color2

        return out_grid



