import numpy as np
import random

# Required imports from the framework
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData



class TaskcLpAqHMFgGScEdjba3Qv84Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "In each row of the input grid, there are several same-colored (1-9) cells, along with some empty cells.",
            "The cell colors are consistent within each row and vary across rows."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid.",
            "For each row, fill all the empty (0) cells which are between the first and last colored cells with the same color."
        ]

        # 3) Call super constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)


    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid following the constraints:
          - Size is rows x cols (from taskvars).
          - Each row has a unique color among {1..9}.
          - Each row has multiple 'lumps' (either single colored cells or pairs)
            separated by empty cells (value=0).
        """
        rows = taskvars['rows']
        cols = taskvars['cols']

        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # We have at most 9 distinct colors; we choose exactly 'rows' distinct colors.
        # (We assume rows <= 9 to satisfy "vary across rows".)
        possible_colors = list(range(1, 10))
        random.shuffle(possible_colors)
        row_colors = possible_colors[:rows]

        for r in range(rows):
            color = row_colors[r]
            # We want 2..4 lumps in this row, each lumps is size 1 or 2
            lumps_needed = random.randint(2, 4)
            lumps_placed = 0
            used_cols = set()  # track used columns to ensure lumps don't overlap or touch
            attempts = 0
            max_attempts = 100

            while lumps_placed < lumps_needed and attempts < max_attempts:
                attempts += 1
                lump_size = random.choice([1, 2])
                # If lump_size=2, valid start columns are [0..(cols-2)]
                # If lump_size=1, valid start columns are [0..(cols-1)]
                max_start = cols - lump_size
                if max_start < 0:
                    # Not enough room for lumps
                    break

                start_col = random.randint(0, max_start)
                # Check if this space is free plus a buffer on each side
                # We want to ensure lumps don't touch, so we check these columns:
                # [start_col - 1, start_col..(start_col + lump_size-1), start_col + lump_size]
                # All must be free (and in range) except we ignore out-of-bounds checks
                candidate_cols = range(start_col, start_col + lump_size)
                neighbor_cols = list(candidate_cols)
                if start_col - 1 >= 0:
                    neighbor_cols.append(start_col - 1)
                if start_col + lump_size < cols:
                    neighbor_cols.append(start_col + lump_size)

                # Check if any col is used
                if any(c in used_cols for c in neighbor_cols):
                    continue

                # If free, place lumps
                for c in candidate_cols:
                    grid[r, c] = color
                # Mark used columns plus a buffer
                for c in neighbor_cols:
                    used_cols.add(c)
                lumps_placed += 1

            # If we couldn't place the desired lumps, it's still okay but might have fewer lumps

        return grid


    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Apply the transformation:
          - Copy the grid.
          - For each row, fill all empty cells between the leftmost and rightmost
            non-empty cell with that row's color.
        """
        out_grid = np.copy(grid)
        rows, cols = out_grid.shape

        for r in range(rows):
            # Find the leftmost and rightmost colored cells in row r
            nonzeros = np.where(out_grid[r] != 0)[0]
            if len(nonzeros) < 2:
                # No fill needed if there's only 0 or 1 colored cell
                continue

            left_col = nonzeros[0]
            right_col = nonzeros[-1]
            # The color is consistent in that row, so we can read the color from the leftmost cell
            row_color = out_grid[r, left_col]

            # Fill all zeros between left_col and right_col with row_color
            for c in range(left_col, right_col + 1):
                if out_grid[r, c] == 0:
                    out_grid[r, c] = row_color

        return out_grid


    def create_grids(self):
        """
        We randomly choose rows in [5..9], cols in [7..30].
        Then we create 3..6 train examples + 1 test example.
        """
        rows = random.randint(5, 9)
        cols = random.randint(7, 30)

        # Because each row must have a distinct color, we ensure rows <= 9
        # (which is consistent with the possible range above).
        taskvars = {
            'rows': rows,
            'cols': cols
        }

        # Decide how many train examples to create (between 3 and 6)
        nr_train = random.randint(3, 6)
        # We'll create exactly one test example
        nr_test = 1

        # Use the helper function to produce the train/test data
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        return taskvars, train_test_data



