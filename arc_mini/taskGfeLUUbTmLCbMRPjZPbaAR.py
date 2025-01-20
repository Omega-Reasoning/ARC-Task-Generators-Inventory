#!/usr/bin/env python3
"""
Example ARC Task Generator following the specified instructions.

We create a subclass of ARCTaskGenerator that produces input grids
with a 1×2 colored block in every second row, off-center to either
the left or right (but not both sides having >=2 empty cells).
Each such input grid must include at least one row with the block
placed left-of-center (i.e. extra space on the right) and at least
one row with the block placed right-of-center (i.e. extra space on
the left). We then transform the input grid by duplicating each
block into the side that has sufficient space, keeping it adjacent.
"""

import numpy as np
import random

# Imports from the provided libraries
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
# We may import from input_library if desired (for randomness / creation helpers),
# but here we implement a straightforward custom approach.

class TaskGfeLUUbTmLCbMRPjZPbaARGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "In every second row of the input grid, there is a colored block with dimensions 1x2, and the remaining cells are empty.",
            "The 1x2 block is placed off-center, ensuring that only one side of the block, either left or right, has two or more empty cells."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and duplicating each 1x2 block.",
            "The duplicated block is always connected to the original and placed on the side with sufficient empty cells.",
            "The duplicated block has the same color as the original."
        ]

        # 3) Call super().__init__ with these chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the instructions:
        - Grid size is rows x cols, both odd in [5..15].
        - A 1x2 colored block appears in every second row, off-center to the left or right.
        - We ensure at least one row is placed left-of-center and at least one row is placed right-of-center.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        while True:
            # Start with an all-empty grid
            grid = np.zeros((rows, cols), dtype=int)

            used_left = False
            used_right = False

            # Place a 1x2 block in every second row (0, 2, 4, ...).
            # Each block is either placed left-of-center or right-of-center.
            for r in range(0, rows, 2):
                side = random.choice(["L", "R"])
                color = random.randint(1, 9)

                if side == "L":
                    # Place block near the left so that the right side has >=2 free columns
                    # and the left side has <2 free columns.
                    # Valid choices: c=0 or c=1 (if it fits).
                    # We do a simple approach: if possible, pick c in {0,1} randomly.
                    possible_cs = []
                    if cols >= 4:
                        possible_cs.append(0)  # always valid if cols >= 4
                    if cols >= 5:
                        possible_cs.append(1)  # valid if cols >= 5

                    if not possible_cs:
                        # fallback (if somehow no placement is possible, skip placing)
                        continue
                    c = random.choice(possible_cs)
                    grid[r, c] = color
                    grid[r, c+1] = color
                    used_left = True

                else:  # side == "R"
                    # Place block near the right so that the left side has >=2 free columns
                    # and the right side has <2 free columns.
                    # Valid choices: c=cols-3 or c=cols-2 if that doesn't go out of range.
                    possible_cs = []
                    if cols >= 5:
                        # c=cols-3 => block covers (cols-3, cols-2)
                        # c=cols-2 => block covers (cols-2, cols-1)
                        possible_cs = [cols-3, cols-2]
                        # Filter out any negative or out-of-bounds picks
                        possible_cs = [cc for cc in possible_cs if 0 <= cc < cols-1]
                    if not possible_cs:
                        # fallback: just place left if right not possible
                        if cols >= 4:
                            c = 0
                            grid[r, c] = color
                            grid[r, c+1] = color
                            used_left = True
                        continue
                    c = random.choice(possible_cs)
                    grid[r, c] = color
                    grid[r, c+1] = color
                    used_right = True

            # Verify we have at least one left-of-center block and at least one right-of-center block
            if used_left and used_right:
                return grid
            # Otherwise, retry until constraints are satisfied.

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transform the input grid:
        - For each row containing the 1x2 block, duplicate it on the side with enough space,
          placing it adjacent to the existing block so that it forms a connected shape.
        """
        rows, cols = grid.shape
        out_grid = grid.copy()

        for r in range(0, rows, 2):
            # Find the 1x2 colored block in row r, if it exists
            # We'll look for the first pair of consecutive non-zero cells in that row.
            row_data = out_grid[r]
            # Identify any consecutive non-zero pair
            for c in range(cols - 1):
                if row_data[c] != 0 and row_data[c+1] == row_data[c]:
                    color = row_data[c]
                    # Check if block is near left or near right
                    # If c <= 1, we placed it to the left => duplicate to the right
                    # If c >= cols-3, we placed it to the right => duplicate to the left
                    if c <= 1:
                        # place new block at c+2, c+3 (should be in range if we constructed properly)
                        out_grid[r, c+2] = color
                        out_grid[r, c+3] = color
                    else:
                        # place new block at c-2, c-1
                        out_grid[r, c-2] = color
                        out_grid[r, c-1] = color
                    break  # only one block per row, so break once found

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        We pick a random odd number of rows and cols in [5, 7, 9, 11, 13, 15].
        We then create multiple train examples and one test example using
        the same rows and cols, but with random block placements (subject to constraints).
        """
        possible_dims = [5, 7, 9, 11, 13, 15]
        rows = random.choice(possible_dims)
        cols = random.choice(possible_dims)

        # Store them in taskvars
        taskvars = {
            'rows': rows,
            'cols': cols
        }

        # We can pick 3 or 4 training examples at random
        nr_train = random.choice([3, 4])
        nr_test = 1

        # Use create_grids_default as a shortcut if we want the same approach for each example
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, train_test_data


