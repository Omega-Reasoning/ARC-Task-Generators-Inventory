# my_arcagi_task.py
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# We only need a subset of functions from transformation_library.py and input_library.py.
# However, we import them here to demonstrate how they could be used, if desired:
# (Nothing from input_library is strictly required, but it's available for more complex generation.)
from transformation_library import find_connected_objects
from input_library import retry, create_object, random_cell_coloring

class TaskLpvjRkGtrWhyFDaoj2gHo6_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain a static rectangular {color('static_object')} {vars['rows_static_object']}x{vars['columns_static_object']} block and a single colored (1-9) cell.",
            "The {color('static_object')} block is positioned to ensure that there is always at least one empty row below it and at least one empty column after it.",
            "The remaining cells are empty."
        ]
        # 2) The transformation reasoning chain
        transformation_reasoning_chain = [
            "The output matrix has the same shape as the input matrix.",
            "The {color('static_object')} object can be directly copied into the output matrix as it remains static.",
            "The single-colored cell is moved to connect diagonally to the bottom-right edge of the {color('static_object')} object."
        ]
        # 3) Call super constructor (no 'taskvars_definitions' parameter needed in this version)
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates task variables and all train/test grids for the ARC task.
        We pick random puzzle-level variables for the static block (color, height, width),
        then generate multiple training pairs plus one test pair, each with varying grid sizes
        and single cell colors.
        """
        # Randomly decide block dimensions and color for the entire puzzle
        rows_static_object = random.randint(2, 4)
        columns_static_object = random.randint(2, 4)
        static_object_color = random.randint(1, 9)

        taskvars = {
            'rows_static_object': rows_static_object,
            'columns_static_object': columns_static_object,
            'static_object': static_object_color
        }

        # Choose how many training examples (3-6) plus 1 test example
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1

        # Use the convenience method that calls create_input() / transform_input() repeatedly
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid according to the input reasoning chain. 
        We place a static rectangular block of color taskvars['static_object'],
        ensure at least one empty row below and one empty column after it,
        and add exactly one single-colored cell that is never initially touching
        the bottom-right edge of the static object.
        """
        # Unpack puzzle-level variables
        rso = taskvars['rows_static_object']      # block height
        cso = taskvars['columns_static_object']   # block width
        sc  = taskvars['static_object']           # block color

        # Generate random grid size: must be at least (rso + 1) x (cso + 1)
        # to ensure there's space below and to the right of the block
        rows = random.randint(rso + 1, 30)
        cols = random.randint(cso + 1, 30)
        grid = np.zeros((rows, cols), dtype=int)

        # Position the block so that there's at least one row below and one column after
        top_row = random.randint(0, rows - rso - 1)
        left_col = random.randint(0, cols - cso - 1)
        grid[top_row:top_row + rso, left_col:left_col + cso] = sc

        # Pick a color for the single cell, ensuring it differs from the block color
        possible_colors = [clr for clr in range(1, 10) if clr != sc]
        single_cell_color = random.choice(possible_colors)

        # Place the single cell in an empty location that is not the final diagonal spot:
        # final diagonal spot is (top_row + rso, left_col + cso).
        # We must ensure the single cell is not placed there.
        while True:
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            if grid[r, c] == 0:
                if not (r == top_row + rso and c == left_col + cso):
                    grid[r, c] = single_cell_color
                    break

        return grid

    def transform_input(self, grid: np.ndarray, taskvars):
        """
        Transform the input grid to the output grid according to the transformation reasoning chain:
        1) Keep the same shape.
        2) The block of color taskvars['static_object'] remains in place.
        3) The single-colored cell is moved to diagonally connect to the block's bottom-right edge.
        """
        out_grid = grid.copy()
        static_color = taskvars['static_object']

        # Find the single cell (the one that is neither 0 nor static_color)
        single_positions = np.argwhere((out_grid != 0) & (out_grid != static_color))
        if len(single_positions) != 1:
            # Ideally should never happen if the input is well-formed
            return out_grid

        sr, sc = single_positions[0]
        single_cell_color = out_grid[sr, sc]

        # Remove the single cell from its current location
        out_grid[sr, sc] = 0

        # Identify the bottom-right corner of the static block
        block_positions = np.argwhere(out_grid == static_color)
        br_row = np.max(block_positions[:, 0])
        br_col = np.max(block_positions[:, 1])

        # Place the single cell diagonally down-right from the block's bottom-right corner
        out_grid[br_row + 1, br_col + 1] = single_cell_color

        return out_grid


