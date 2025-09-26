from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# We can import from the libraries provided:
# input_library.py (for create_input) and transformation_library.py
# though the solution below only needs minor array indexing for transform_input.
from input_library import retry
from input_library import Contiguity
from transformation_library import find_connected_objects  # example usage if needed

class TaskcmBhVbGzL8ZgWXDE5CUS6BGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (Observation chain) - list of strings
        observation_chain = [
            "Input grids can have different sizes.",
            "The input grids contain a single {color('object_color1')} rectangular frame.",
            "The frame is one cell wide, with empty (0) cells both inside and outside it.",
            "The position of this frame can vary across examples."
        ]

        # 2) Transformation reasoning chain - list of strings
        reasoning_chain = [
            "To construct the output grid, copy the input grid and all non-corner {color('object_color1')} cells on the frame are changed to {color('object_color2')}."
        ]

        # 3) Call parent constructor
        super().__init__(observation_chain, reasoning_chain)

    def create_input(self, 
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid containing a single rectangular frame of color object_color1.
        The frame is one cell wide, with interior and exterior cells = 0.
        The frame is placed at a random position inside the grid, with at least 1 cell of empty space 
        around it, and is at least 4x3 in bounding box size (or 3x4).
        """
        # Extract colors from the overall task variables
        color1 = taskvars["object_color1"]

        # Random grid size between 10 and 20
        rows = random.randint(10, 30)
        cols = random.randint(10, 30)
        grid = np.zeros((rows, cols), dtype=int)

        # Choose a random bounding box size for the frame:
        # Min bounding box dimension is 3 in one dimension and 4 in the other, 
        # but we typically want to allow the bounding box to be reversed, e.g. 4x3 or 3x4
        min_height, min_width = 3, 4
        max_height, max_width = rows - 2, cols - 2  # leave space for external margin

        # random bounding box size, ensuring minimum size
        frame_height = random.randint(min_height, max_height)
        frame_width = random.randint(min_width, max_width)

        # We also want to ensure the bounding box is fully inside with at least 1 margin
        # The top-left corner of the frame bounding box can vary
        top = random.randint(1, rows - frame_height - 1)
        left = random.randint(1, cols - frame_width - 1)

        # Place a 1-pixel thick rectangular frame in the bounding box region
        # top row
        grid[top, left:left+frame_width] = color1
        # bottom row
        grid[top + frame_height - 1, left:left+frame_width] = color1
        # left column
        grid[top:top+frame_height, left] = color1
        # right column
        grid[top:top+frame_height, left + frame_width - 1] = color1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        - Copy the input grid.
        - Change all non-corner frame cells of color object_color1 to object_color2.
          Corners remain in object_color1.
        """
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]

        # Copy the grid
        out_grid = np.copy(grid)

        # We only have one rectangular frame, but let's be systematic:
        # We can find all cells of color1 that are in the "frame".
        # Then we skip corners by checking adjacency or bounding box.

        # A quick approach is to locate all color1 cells that have at least two neighbors
        # or use direct corner detection. Since we know it's a single rectangular frame:
        # 1) find all color1 cells
        row_coords, col_coords = np.where(out_grid == color1)

        # If there's only one rectangle, corners can be recognized by counting how many color1
        # neighbors they have in row or col. However, it's simpler to check if the cell is
        # not corner by looking at local adjacency or bounding box extremes.
        # We'll identify bounding box of color1.
        if len(row_coords) == 0:
            # No color1 cells found; edge case
            return out_grid

        # bounding box for color1
        min_r, max_r = row_coords.min(), row_coords.max()
        min_c, max_c = col_coords.min(), col_coords.max()

        # define corners
        corners = {
            (min_r, min_c),
            (min_r, max_c),
            (max_r, min_c),
            (max_r, max_c),
        }

        for r, c in zip(row_coords, col_coords):
            # skip corners
            if (r, c) in corners:
                continue
            # if it is in bounding box, we assume it's part of the frame
            # (since interior is 0, only the frame is color1)
            if min_r <= r <= max_r and min_c <= c <= max_c:
                out_grid[r, c] = color2

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Randomly initialise object_color1 and object_color2 (in [1..9], different),
        and create 3 training pairs + 1 test pair using the default helper in ARCTaskGenerator.
        The returned dictionary has the keys needed for color('object_color1') and color('object_color2').
        """
        # Choose two distinct nonzero colors for the frame
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)

        taskvars = {
            "object_color1": color1,
            "object_color2": color2
        }

        # We will create 3 train examples and 1 test example.
        # Each is guaranteed to differ in size/position by the random calls in create_input().
        train_test_data = self.create_grids_default(
            nr_train_examples=3,
            nr_test_examples=1,
            taskvars=taskvars
        )

        return taskvars, train_test_data


