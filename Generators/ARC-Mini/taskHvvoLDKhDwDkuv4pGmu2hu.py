# columns_filling_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
class TaskHvvoLDKhDwDkuv4pGmu2huGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain:
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain several completely filled columns of {color('object_color1')} color.",
            "Each {color('object_color1')} column is separated by at least one empty (0) column."
        ]
        
        # 2) Transformation reasoning chain:
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling all empty (0) columns with {color('object_color2')} color if they are positioned between two {color('object_color1')} columns.",
            "If multiple empty (0) columns exist between two {color('object_color1')} columns, they are all filled with {color('object_color2')} color."
        ]
        
        # 3) Call superclass initializer:
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid with:
        - A random number of columns (≥ 3) fully filled with 'object_color1',
        - Each filled column separated by at least one empty column,
        - Grid size given by gridvars['rows'] x gridvars['cols'].
        """
        color1 = taskvars['object_color1']
        rows = gridvars['rows']
        cols = gridvars['cols']

        # Create a blank grid of zeros
        grid = np.zeros((rows, cols), dtype=int)

        # We need at least 3 columns of color1
        # We'll allow up to 5, or as many as can comfortably fit in this width
        max_fillable = min(5, (cols + 1) // 2)  # Quick upper bound
        n_filled = random.randint(3, max_fillable)

        # Randomly choose n_filled columns with at least 1 empty column between them
        # We'll do a simple retry approach to ensure we find a valid arrangement
        def try_place_filled():
            # pick n_filled distinct column indices in ascending order
            chosen = []
            # We'll pick from 0..(cols-1), ensuring at least 1 gap
            # We'll keep trying up to 100 times
            for _ in range(100):
                candidate = sorted(random.sample(range(cols), n_filled))
                # Check if there's at least 1 gap between consecutive columns
                valid = True
                for i in range(n_filled - 1):
                    if candidate[i+1] - candidate[i] < 2:
                        valid = False
                        break
                if valid:
                    return candidate
            return None

        filled_cols = try_place_filled()
        if filled_cols is None:
            # If we somehow fail, default to a simple fixed arrangement
            # e.g. put them at columns 0, 2, 4 if n_filled=3 (it must fit given the constraints)
            filled_cols = list(range(0, 2*n_filled, 2))

        # Fill those columns with color1
        for c in filled_cols:
            grid[:, c] = color1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Implement the transformation:
        * Copy the input grid,
        * For every pair of columns that are fully color1, fill all intervening empty columns with color2.
        """
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']

        out_grid = grid.copy()
        rows, cols = grid.shape

        # 1) Identify which columns are fully color1
        color1_columns = []
        for c in range(cols):
            if np.all(grid[:, c] == color1):
                color1_columns.append(c)

        # 2) For every consecutive pair (left_col, right_col),
        #    fill the empty columns between them with color2.
        for i in range(len(color1_columns) - 1):
            left_col = color1_columns[i]
            right_col = color1_columns[i + 1]
            # Fill columns (left_col+1) .. (right_col-1) if they are entirely empty
            for c in range(left_col + 1, right_col):
                if np.all(out_grid[:, c] == 0):
                    out_grid[:, c] = color2

        return out_grid

    def create_grids(self):
        """
        Creates 3-4 training examples plus one test example, each with a distinct grid size.
        Also picks color1 != color2 in [1..9].
        Returns:
            taskvars: Dictionary with chosen color1, color2
            train_test_data: { 'train': [...], 'test': [...] }
        """
        # 1) Select distinct colors
        color1, color2 = random.sample(range(1, 10), 2)

        # 2) Decide how many train grids (3 or 4) + 1 test grid
        nr_train = random.choice([3, 4])
        total = nr_train + 1

        # 3) Pick distinct grid sizes (rows x cols) in range [7..30]
        #    We'll just pick them up to e.g. 15 to keep them reasonably small.
        #    It's valid up to 30, but this is arbitrary.
        sizes = set()
        while len(sizes) < total:
            r = random.randint(7, 30)
            c = random.randint(7, 30)
            sizes.add((r, c))
        sizes = list(sizes)
        random.shuffle(sizes)

        # 4) Build train/test pairs
        train_data = []
        for i in range(nr_train):
            rows, cols = sizes[i]
            input_grid = self.create_input(
                {'object_color1': color1, 'object_color2': color2},
                {'rows': rows, 'cols': cols}
            )
            output_grid = self.transform_input(input_grid,
                                               {'object_color1': color1, 'object_color2': color2})
            train_data.append({'input': input_grid, 'output': output_grid})

        # Test pair
        rows, cols = sizes[-1]
        test_input = self.create_input(
            {'object_color1': color1, 'object_color2': color2},
            {'rows': rows, 'cols': cols}
        )
        test_output = self.transform_input(test_input,
                                           {'object_color1': color1, 'object_color2': color2})
        test_data = [{'input': test_input, 'output': test_output}]

        taskvars = {
            'object_color1': color1,
            'object_color2': color2
        }
        train_test_data = {
            'train': train_data,
            'test': test_data
        }
        return taskvars, train_test_data



