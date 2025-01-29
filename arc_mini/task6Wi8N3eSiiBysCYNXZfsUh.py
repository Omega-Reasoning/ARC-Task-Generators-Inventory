#!/usr/bin/env python3
"""
Example ARC-AGI Task Generator

This generator creates tasks where:
1) The input grids always contain exactly two 2x2 objects of distinct colors,
   with the first (object_color1) always at position (1,1),
   and the second (object_color2) at least two rows below and two columns to
   the right of the first object.
2) The output grid is constructed by:
   - Recoloring the first object to change_color1.
   - Moving the second object upward so that it shares the same top row as the
     first object (row=1) but remains in the same columns, then recoloring it
     to change_color2.
3) We vary grid sizes and positions for variety, ensuring constraints are satisfied.
"""

import numpy as np
import random

# Imports required by the instructions:
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects, GridObject

class Task6Wi8N3eSiiBysCYNXZfsUhGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain as given:
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain exactly two objects, which are 2x2 blocks of {color('object_color1')} and {color('object_color2')} colors.",
            "The {color('object_color1')} object always starts at position (1,1), while the {color('object_color2')} object is positioned at least one row below and at least one column to the right of the {color('object_color1')} object."
        ]

        # 2) Transformation reasoning chain as given:
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and changing the color of the {color('object_color1')} object to {color('change_color1')}.",
            "The {color('object_color2')} object is moved upward but remains in the same columns to align with the {color('object_color1')} object, and its color is changed to {color('change_color2')}."
        ]

        # 3) Call super constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain.
        The input grid has shape (rows, cols).
        - A 2x2 block of object_color1 is placed at position (1,1).
        - A 2x2 block of object_color2 is placed randomly at position (r2, c2)
          where r2 >= 3 and c2 >= 3 (1-based indexing) so that it is at least
          two rows below and two columns right of the first object.
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]

        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Place the first object (2x2) at (1,1) with color object_color1
        # Note: Using 0-based indexing in NumPy: that means we place it at grid[1,1], grid[1,2], grid[2,1], grid[2,2].
        # Because "position (1,1)" in a typical puzzle sense is row=1, col=1 (1-based). We'll adapt as needed below.
        grid[1,1] = color1
        grid[1,2] = color1
        grid[2,1] = color1
        grid[2,2] = color1

        # Randomly place the second 2x2 object subject to constraints:
        # "At least two rows below and two columns to the right" of the first object
        # means top-left of second object must be at row >= 3, col >= 3 in 1-based indexing.
        # So in 0-based indexing, min row/col is 2, but the puzzle text says row=1 for the first object.
        # We'll keep it simple: the second 2x2 must start at row >= 3 (1-based), col >= 3 (1-based).
        # In 0-based indexing, that means row >= 2, col >= 2. But to be "at least 2 below," we actually shift a bit more.
        # However, the problem statement specifically: 
        # "The second object is positioned at least two rows below and two columns to the right of the first object."
        # If the first object top-left is (1,1) 1-based, that means second object top-left row >= 1+2=3,
        # col >= 1+2=3. So in 0-based indexing: row >= 2, col >= 2. We'll do that.

        # However, we must ensure enough space so 2x2 fits. The bottom-right corner must be within (rows-1, cols-1).
        # We'll pick from [2..(rows-2)] for row, [2..(cols-2)] for col in 0-based indexing.
        possible_row_positions = list(range(3, rows-1))  # 1-based indexing min=3 => 0-based min=2, but for a 2x2 we do rows-1 - 1 => rows-2
        possible_col_positions = list(range(3, cols-1))
        # But let's be consistent with 1-based indexing => 0-based: row >= 2 => let's do range(2, rows-1)
        # We'll just do a random choice ensuring we can place a 2x2 block safely.
        # Actually let's keep it simple with the constraints: row >= 3 => 0-based >= 2. We'll implement that directly:

        # Correction for clarity: If top-left must be row=3 in 1-based, that is row=2 in 0-based. 
        # But let's use row >= 3 in 1-based => row>=2 in 0-based. 
        # The top-left can go up to row=rows-2 in 0-based for a 2x2 block. 
        # So final range for row in 0-based is [2..(rows-2)] inclusive.
        possible_row_positions = range(5, rows-1)
        possible_col_positions = range(5, cols-1)

        r2 = random.choice(list(possible_row_positions))
        c2 = random.choice(list(possible_col_positions))

        # Place the 2x2 block in the chosen location with color2
        grid[r2, c2] = color2
        grid[r2, c2+1] = color2
        grid[r2+1, c2] = color2
        grid[r2+1, c2+1] = color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        1) Copy the input grid.
        2) Recolor the first 2x2 block (object_color1) to change_color1.
        3) Move the second 2x2 block (object_color2) upward so that its top row
           is row=1 (i.e., 0-based index row=1) but preserving its columns. Then
           recolor it to change_color2.
        """
        object_color1 = taskvars["object_color1"]
        object_color2 = taskvars["object_color2"]
        change_color1 = taskvars["change_color1"]
        change_color2 = taskvars["change_color2"]

        # Make a copy
        out_grid = grid.copy()

        # Find all objects (monochromatic = True, background=0)
        objects = find_connected_objects(
            out_grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=True
        )

        # We expect to find exactly one 2x2 block with color1 (4 cells) and one with color2 (4 cells).
        # 1) Recolor object1 => object_color1 -> change_color1
        #    object1 is presumably the 2x2 block containing color object_color1 in row=1..2, col=1..2 (0-based).
        obj1_candidates = objects.with_color(object_color1).with_size(min_size=4, max_size=4)
        if len(obj1_candidates) == 1:
            obj1 = obj1_candidates[0]
            # Recolor in-place
            for (r, c, col) in list(obj1.cells):
                # remove old cell
                obj1.cells.remove((r, c, col))
                # add recolored cell
                obj1.cells.add((r, c, change_color1))
            # Now cut and paste to apply the recoloring to out_grid
            obj1.cut(out_grid).paste(out_grid, overwrite=True)

        # 2) Move object2 upward to row=1 (0-based), preserve columns, recolor -> change_color2
        obj2_candidates = objects.with_color(object_color2).with_size(min_size=4, max_size=4)
        if len(obj2_candidates) == 1:
            obj2 = obj2_candidates[0]

            # figure out how far to move it up so top row = 1
            # bounding box top => obj2.bounding_box[0].start
            top_row = obj2.bounding_box[0].start
            desired_top = 1
            delta = desired_top - top_row

            # Recolor it first, or after translation - order doesn't matter for final result
            # We'll do the recoloring first
            for (r, c, col) in list(obj2.cells):
                obj2.cells.remove((r, c, col))
                obj2.cells.add((r, c, change_color2))

            # Cut from out_grid so we can paste after translation
            obj2.cut(out_grid)

            # Translate it
            obj2.translate(dx=delta, dy=0, grid_shape=out_grid.shape)

            # Paste back
            obj2.paste(out_grid)

        return out_grid

    def create_grids(self):
        """
        Randomly generate task variables (rows, cols, object_color1, object_color2,
        change_color1, change_color2) and create 3-6 training pairs + 1 test pair.

        Returns:
            (taskvars, train_test_data)
        """
        # 1) Pick random grid size from 8..30
        rows = random.randint(8, 30)
        cols = random.randint(8, 30)

        # 2) Pick 4 distinct colors from 1..9
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        object_color1, object_color2, change_color1, change_color2 = all_colors[:4]

        # Prepare dictionary of task variables
        taskvars = {
            "rows": rows,
            "cols": cols,
            "object_color1": object_color1,
            "object_color2": object_color2,
            "change_color1": change_color1,
            "change_color2": change_color2
        }

        # We produce 3-6 training pairs + 1 test pair
        nr_train = random.randint(3, 6)
        nr_test = 1

        # We'll just create train/test data with the create_grids_default, which
        # calls create_input() & transform_input() multiple times. However, we need
        # to ensure the position of the second object changes across examples. We'll
        # rely on the internal random call in create_input() to do that naturally.
        # Because create_input() picks a random position for the second object each time,
        # we already get a different position in each example.

        data = self.create_grids_default(nr_train, nr_test, taskvars)
        return taskvars, data



