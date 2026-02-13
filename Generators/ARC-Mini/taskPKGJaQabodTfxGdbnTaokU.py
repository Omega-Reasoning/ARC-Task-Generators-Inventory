# example_arc_task_generator.py

import numpy as np
import random
from typing import Dict, Any, Tuple, List
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
# If you wish to use from the transformation_library, you can import it here
# e.g. from transformation_library import find_connected_objects, GridObjects, GridObject

# We only use input_library for create_input() usage:
from input_library import Contiguity

class TaskPKGJaQabodTfxGdbnTaokUGenerator(ARCTaskGenerator):
    def __init__(self):
        # (1) Copy the input reasoning chain exactly as given
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They only contain two vertically connected objects: a 1x2 block of {color('object_color1')} color and an L-shaped object of {color('object_color2')} color.",
            "An L-shaped object is of the form [[{color('object_color2')},0],[{color('object_color2')},0],[{color('object_color2')},{color('object_color2')}]]",
            "The 1x2 block is placed only in the first or last row, and the L-shaped object is positioned directly below or above it, respectively."
        ]

        # (2) Copy the transformation reasoning chain exactly as given
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid.",
            "Move the {color('object_color2')} L-shaped object one unit down or up so that there is always one empty (0) row between the {color('object_color1')} object and the {color('object_color2')} object."
        ]

        # (3) Call super().__init__ with the two reasoning chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)


    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid with:
          - Random dimensions from 5..30 in both directions.
          - A 1×2 block of color object_color1 placed in the first or last row.
          - An L-shaped object of color object_color2 placed directly below or above that block.
            The L-shape is:
               [[color2, 0],
                [color2, 0],
                [color2, color2]]
        No empty row should separate them in the input (they share a border).
        """
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]

        # Decide whether the 1×2 block goes on top or bottom based on gridvars or random
        block_on_top = gridvars.get("block_on_top", random.choice([True, False]))

        # Random grid size
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        grid = np.zeros((height, width), dtype=int)

        # The 1×2 block:
        #   If block_on_top: place in row=0
        #   else: place in row=height-1
        # We choose a random column so that the block fits entirely.
        start_col = random.randint(0, width - 2)

        if block_on_top:
            block_row = 0
        else:
            block_row = height - 1

        # Place the 1×2 block with color1
        grid[block_row, start_col] = color1
        grid[block_row, start_col + 1] = color1

        # The L-shaped object is 3 rows tall by 2 cols wide. We'll place it so that
        # it touches the block: either right below (if block_on_top) or right above (if block_on_bottom).
        # L-shape pattern using color2:
        #   [ [ color2, 0      ],
        #     [ color2, 0      ],
        #     [ color2, color2 ] ]

        # If block_on_top, top-left of L-shape at (block_row+1, start_col).
        # If block_on_bottom, bottom of L-shape touches the block row.
        if block_on_top:
            l_top_row = block_row + 1
        else:
            # The L-shape bottom row is block_row - 1 => top row is block_row - 3
            l_top_row = (block_row - 3)

        # Insert the L-shape pattern:
        # row 0 of L-shape
        grid[l_top_row,     start_col]     = color2
        # row 1 of L-shape
        grid[l_top_row + 1, start_col]     = color2
        # row 2 of L-shape
        grid[l_top_row + 2, start_col]     = color2
        grid[l_top_row + 2, start_col + 1] = color2

        return grid


    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Given the input grid, transform it by:
          1. Copying the grid.
          2. Moving the color(object_color2) L-shaped object one unit down or up
             so there is exactly one empty row between the 1×2 block and the L-shaped object.
        """
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]

        # We'll identify which object is the L-shape by its color2 cells.
        # There's only one object with color2 in the grid.
        input_grid = grid
        output_grid = np.copy(input_grid)

        # Collect coordinates of all color2 cells (the L-shape).
        coords_l = np.argwhere(output_grid == color2)

        # If there's no color2, just return a copy (edge case).
        if coords_l.size == 0:
            return output_grid

        # Remove L-shape from output_grid
        for (r, c) in coords_l:
            output_grid[r, c] = 0

        # Figure out if the 1×2 block is on top or bottom:
        # We'll check for color1 in top row vs bottom row
        # A simpler approach is to see if color1 appears in row=0 or row=height-1
        height = output_grid.shape[0]

        block_is_top = False
        # If we find color1 in the top row => block_is_top = True
        top_row_color1 = (color1 in output_grid[0, :])
        bottom_row_color1 = (color1 in output_grid[height - 1, :])

        if top_row_color1 and (not bottom_row_color1):
            block_is_top = True
        elif bottom_row_color1 and (not top_row_color1):
            block_is_top = False
        else:
            # In ambiguous or random cases, pick up or down. Usually, one or the other should be true.
            block_is_top = True

        # Move the L-shape coords up or down by 1
        # so that there's exactly one empty row between the block and L-shape.
        # If block_is_top => move L-shape down by 1
        # If block_is_bottom => move L-shape up by 1
        shift = 1 if block_is_top else -1

        shifted_coords = []
        for (r, c) in coords_l:
            new_r = r + shift
            shifted_coords.append((new_r, c))

        # Paste them back in output_grid
        for (r, c) in shifted_coords:
            output_grid[r, c] = color2

        return output_grid


    def create_grids(self):
        """
        We generate 3-6 training examples and 2 test examples, ensuring:
          - At least one training input has the block on top row.
          - At least one training input has the block on bottom row.
          - The same pattern for test examples.
        We also randomly choose object_color1 and object_color2 (distinct).
        """
        # Random distinct colors
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)

        taskvars = {
            "object_color1": color1,
            "object_color2": color2
        }

        # We want between 3 and 6 train examples total
        n_extra_train = random.randint(1, 4)  # we already fix 2 examples, plus 1..4 more => total 3..6

        def create_example(block_on_top: bool):
            """Helper to build input-output pair with forced block_on_top/bottom."""
            inp = self.create_input(taskvars, {"block_on_top": block_on_top})
            out = self.transform_input(inp, taskvars)
            return {"input": inp, "output": out}

        # Ensure at least 1 top & 1 bottom among the training examples
        train = []
        train.append(create_example(True))
        train.append(create_example(False))
        for _ in range(n_extra_train):
            train.append(create_example(random.choice([True, False])))

        # We also want exactly 2 test examples: one top, one bottom
        test = [
            create_example(True),
            create_example(False)
        ]

        return taskvars, {
            "train": train,
            "test": test
        }



