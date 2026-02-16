# my_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import retry

class TaskMCvs8GJdqaKcrG3QMPSdDpGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        observation_chain = [
            "Input grids are of size {vars['grid_rows']} x {vars['grid_cols']}.",
            "Each input grid contains exactly two colored cells: {color('color_1')} and {color('color_2')}, seperated by empty (0) cells."
        ]
        
        # 2) Transformation reasoning chain
        reasoning_chain = [
            "The output grid is constructed by copying the input grid and transforming the single-colored cells to 2x2 blocks, given that it is possible.",
            "For each colored cell in the input grid; if the adjacent cells to the right, below, and diagonally downwards are empty (0), fill them with the same color as the corresponding colored cell."
        ]

        # 3) Call super init with the reasoning chains
        super().__init__(observation_chain, reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:

        rows = taskvars['grid_rows']
        cols = taskvars['grid_cols']
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']

        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # We must place two distinct colored cells in the grid.
        # And ensure they are "separated by empty cells" 
        # as well as ensuring at least one of them can expand.

        # A convenient way is to keep retrying random placements 
        # until we meet the conditions.
        def attempt_placement():
            grid_attempt = np.zeros((rows, cols), dtype=int)

            # Random positions for two colored cells
            r1 = random.randint(0, rows - 1)
            c1 = random.randint(0, cols - 1)
            r2 = random.randint(0, rows - 1)
            c2 = random.randint(0, cols - 1)

            # Make sure the second cell is not in the same position
            # and is not immediately on top of the first cell
            # (the problem statement says separated by empty cells,
            #  which typically implies they cannot be the same cell or 
            #  share adjacency if we interpret "separated" strictly. 
            #  But the statement is a bit ambiguous: 
            #  "Colored cells are separated by empty (0) cells."
            #  We'll interpret it as simply they are placed in different coordinates.)
            if (r1 == r2 and c1 == c2):
                return None  # invalid attempt

            # Place them
            grid_attempt[r1, c1] = color_1
            grid_attempt[r2, c2] = color_2

            # Now, check if at least one colored cell has space to expand.
            # i.e. (r, c+1), (r+1, c), (r+1, c+1) are within bounds and 0
            def can_expand(r, c):
                if (r + 1 < rows) and (c + 1 < cols):
                    return (grid_attempt[r, c+1] == 0 and
                            grid_attempt[r+1, c] == 0 and
                            grid_attempt[r+1, c+1] == 0)
                return False

            # We want at least one cell to be able to expand
            if can_expand(r1, c1) or can_expand(r2, c2):
                return grid_attempt
            else:
                return None  # Doesn't meet expansion requirement

        # Use a simple retry approach
        # (We could use input_library.retry, but let's keep it simple here.)
        max_attempts = 200
        for _ in range(max_attempts):
            result = attempt_placement()
            if result is not None:
                return result

        raise ValueError("Could not place two colored cells with required constraints after many attempts.")

    def transform_input(self,grid: np.ndarray,taskvars: dict) -> np.ndarray:
        rows, cols = grid.shape
        output = grid.copy()

        for r in range(rows):
            for c in range(cols):
                current_color = grid[r, c]
                # Only act on non-empty cells
                if current_color != 0:
                    # Check adjacency in-bounds and empty
                    if r + 1 < rows and c + 1 < cols:
                        if (output[r, c+1] == 0 and
                            output[r+1, c] == 0 and
                            output[r+1, c+1] == 0):
                            # Fill them in
                            output[r, c+1] = current_color
                            output[r+1, c] = current_color
                            output[r+1, c+1] = current_color

        return output

    def create_grids(self) -> (dict, TrainTestData):
        # Randomly choose grid size within the 30x30 limit, but keep it big enough
        # to allow 2x2 expansions. We'll keep them from 4 to 10 in each dimension for variety.
        grid_rows = random.randint(4, 30)
        grid_cols = random.randint(4, 30)

        # Pick two distinct colors among 1..9
        color_1 = random.randint(1, 9)
        color_2 = random.randint(1, 9)
        while color_2 == color_1:
            color_2 = random.randint(1, 9)

        taskvars = {
            'grid_rows': grid_rows,
            'grid_cols': grid_cols,
            'color_1': color_1,
            'color_2': color_2
        }

        # Decide how many train examples to generate (3-6)
        nr_train = random.randint(3, 6)
        nr_test = 1

        train_test_data: TrainTestData = {
            'train': [],
            'test': []
        }

        # Create the train examples
        for _ in range(nr_train):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            train_test_data['train'].append(
                GridPair(input=inp, output=out)
            )

        # Create the test example
        inp_test = self.create_input(taskvars, {})
        out_test = self.transform_input(inp_test, taskvars)
        train_test_data['test'].append(
            GridPair(input=inp_test, output=out_test)
        )

        return taskvars, train_test_data

