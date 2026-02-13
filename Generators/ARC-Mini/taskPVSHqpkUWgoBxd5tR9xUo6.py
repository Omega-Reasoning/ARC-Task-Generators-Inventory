"""
Example ARC-AGI Task Generator

Generates tasks where we have two 2x2 blocks of the same color in a grid.
They are placed such that there is exactly one empty row between them,
and the second block begins in the column immediately after the first block ends.
The puzzle solution is to connect the two blocks by filling two empty cells:
one directly to the left of the lower block's top-left cell, and one above that cell.
"""

import numpy as np
import random

# Required imports from the provided framework
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData


class TaskPVSHqpkUWgoBxd5tR9xUo6Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain two {color('object_color')} objects, each forming a 2x2 block, separated by empty (0) cells.",
            "Place the first object so that there are at least three empty (0) rows below it and two empty (0) columns to its right.",
            "Then, position the second object by leaving one empty row below the first object and placing it exactly in the next column where the first object ends."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and connecting the two {color('object_color')} objects by filling two empty (0) cells.",
            "The two empty cells are directly below the bottom-right cell of the first 2x2 block."
        ]

        # 3) Call superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an input grid of size rows x cols with two 2x2 blocks of the same color.
        The first block is at (start_row, start_col), the second block is at
        (start_row+3, start_col+2), ensuring exactly one empty row in between them
        and the second block starts immediately to the right of the first block's columns.
        """

        rows = taskvars['rows']
        cols = taskvars['cols']
        color = taskvars['object_color']

        start_row = gridvars['start_row']
        start_col = gridvars['start_col']

        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Place the first 2x2 block
        for r in range(2):
            for c in range(2):
                grid[start_row + r, start_col + c] = color

        # Place the second 2x2 block (one row gap, shifted right one column)
        for r in range(2):
            for c in range(2):
                grid[start_row + 3 + r, start_col + 2 + c] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transform the input grid by copying it and connecting the two 2x2 blocks:
        - The first newly filled cell is to the left of the lower block's top-left cell.
        - The second newly filled cell is directly above that cell.
        """

        color = taskvars['object_color']
        output_grid = grid.copy()

        # Locate the lower 2x2 block
        coords = [(r, c) for r in range(grid.shape[0]) 
                  for c in range(grid.shape[1]) if grid[r, c] == color]

        # Group into two blocks (each block is 2x2)
        all_blocks = []
        visited = set()
        for rc in coords:
            if rc in visited:
                continue
            r, c = rc
            block_points = []
            for rr in range(r, r+2):
                for cc in range(c, c+2):
                    if (rr, cc) in coords:
                        block_points.append((rr, cc))
                        visited.add((rr, cc))
            if len(block_points) == 4:  # Ensure it's a 2x2 block
                all_blocks.append(block_points)

        # Sort blocks by row position (lower one comes last)
        all_blocks.sort(key=lambda block: min(p[0] for p in block))

        if len(all_blocks) < 2:
            return output_grid  # Should not happen, but just in case

        # Lower block bounding box
        lower_block = all_blocks[-1]
        minr = min(p[0] for p in lower_block)
        minc = min(p[1] for p in lower_block)

        # Ensure we don't go out of bounds before modifying the grid
        if minc - 1 >= 0 and minr < grid.shape[0] and minc < grid.shape[1]:  # Bounds check
            output_grid[minr, minc - 1] = color
            if minr - 1 >= 0:
                output_grid[minr - 1, minc - 1] = color

        return output_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Randomly choose overall puzzle parameters once (rows, cols, object_color)
        Then create 2-3 training examples and 1 test example with distinct positions.
        Return the dictionary of task variables plus the train/test data.
        """

        # 1. Choose puzzle-wide parameters
        rows = random.randint(7, 30)
        cols = random.randint(7, 30)
        object_color = random.randint(1, 9)

        # This dictionary is used in the f-string expansions {vars['rows']} etc.
        taskvars = {
            'rows': rows,
            'cols': cols,
            'object_color': object_color
        }

        nr_train = random.choice([2, 3])  # 2 or 3 training examples
        train_data = []
        used_positions = set()

        def pick_random_position():
            max_start_row = rows - 5
            max_start_col = cols - 4
            if max_start_row < 0 or max_start_col < 1:
                return None

            attempts = 0
            while attempts < 100:
                r = random.randint(0, max_start_row)
                c = random.randint(1, max_start_col)
                if (r, c) not in used_positions:
                    return (r, c)
                attempts += 1
            return None

        for _ in range(nr_train):
            pos = pick_random_position()
            if pos is None:
                break
            used_positions.add(pos)
            grid_in = self.create_input(taskvars, {'start_row': pos[0], 'start_col': pos[1]})
            grid_out = self.transform_input(grid_in, taskvars)
            train_data.append({'input': grid_in, 'output': grid_out})

        pos_test = pick_random_position()
        if pos_test is None:
            pos_test = list(used_positions)[-1]

        grid_in_test = self.create_input(taskvars, {'start_row': pos_test[0], 'start_col': pos_test[1]})
        grid_out_test = self.transform_input(grid_in_test, taskvars)

        test_data = [{'input': grid_in_test, 'output': grid_out_test}]

        return taskvars, {'train': train_data, 'test': test_data}



