
import numpy as np
import random

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import retry  # we only use 'retry' here for demonstration
# We do not strictly need create_object() etc. for this puzzle, so we omit them.

class Task2fJ984g27gSFKHfq53RTVHGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain exactly two rectangular objects, one touching the rightmost edge of the grid and the other touching the leftmost edge, always separated by empty (0) columns.",
            "Each rectangular object has a length of {vars['rows']}, extending from the first row to the last row, but they have different widths and colors (1-9)."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling the empty columns between the two rectangular objects with the color of the rightmost object."
        ]

        # 3) Call the superclass initializer
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create a grid that:
          - Has dimensions rows x cols from taskvars.
          - Contains exactly two tall rectangles: one at the left edge, one at the right edge.
            Each rectangle spans all rows. They must have different widths, different colors,
            and at least 3 empty columns between them.
        """

        rows = taskvars['rows']
        cols = taskvars['cols']

        # Create a blank grid (all zeros)
        grid = np.zeros((rows, cols), dtype=int)

        # Randomly choose widths for left and right rectangles, ensuring they differ
        # and that there's room for at least 3 columns in between.
        # We also ensure left_width + right_width <= cols - 3
        possible_widths = list(range(2, cols - 3))  # at least 2 col each, leaving 3 in the middle
        # We'll pick pairs (w1, w2) until we find a valid combination
        def pick_widths():
            w1 = random.choice(possible_widths)
            w2 = random.choice(possible_widths)
            # Must differ, must fit
            if w1 != w2 and w1 + w2 <= cols - 3:
                return w1, w2
            else:
                return None

        w1, w2 = retry(
            lambda: pick_widths(),
            lambda x: x is not None
        )

        # Randomly choose two distinct colors in [1..9]
        left_color = random.randint(1, 9)
        right_color = random.randint(1, 9)
        while right_color == left_color:
            right_color = random.randint(1, 9)

        # Fill leftmost rectangle: columns [0..w1-1] for all rows
        grid[:, :w1] = left_color

        # Fill rightmost rectangle: columns [cols - w2..cols-1] for all rows
        grid[:, cols - w2:] = right_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Follows the transformation reasoning:
          "The output grid is constructed by copying the input grid and
           filling the empty columns between the two rectangular objects
           with the color of the rightmost object."
           
        Implementation notes:
          - We identify the columns of the left rectangle and the columns of
            the right rectangle. Then fill all zero-cells in-between with the
            color of the right rectangle.
        """

        # Copy the grid
        out_grid = grid.copy()

        rows, cols = out_grid.shape

        # 1) Identify the rightmost rectangle's color:
        #    By puzzle design, the last column is fully occupied by the right rectangle's color.
        color_right = out_grid[0, cols - 1]  # every cell in that column has the same color, so top cell suffices

        # 2) Identify the boundary of the left rectangle:
        #    The left rectangle is at columns from 0..(left_boundary), that last column of left rect is not 0.
        #    We can find the largest index col where out_grid[0, col] != 0 from the left side.
        left_boundary = 0
        for c in range(cols):
            if out_grid[0, c] != 0:
                left_boundary = c
            else:
                break

        # 3) Identify the boundary of the right rectangle:
        #    The right rectangle is at columns from (right_boundary)..(cols - 1).
        #    We find the smallest index col (from the right) where out_grid[0, col] != 0.
        right_boundary = cols - 1
        for c in range(cols - 1, -1, -1):
            if out_grid[0, c] != 0:
                right_boundary = c
            else:
                break

        # 4) Fill the columns between (left_boundary+1) and (right_boundary-1) that are 0 with color_right
        for c in range(left_boundary + 1, right_boundary):
            # Only fill if they're empty columns
            if np.all(out_grid[:, c] == 0):
                out_grid[:, c] = color_right

        return out_grid

    def create_grids(self):
        """
        Creates multiple (train) input-output pairs and 1 test pair.
        We store {rows, cols} in taskvars for the reasoning chain placeholders.
        """

        # Choose a single rows/cols for the entire puzzle
        rows = random.randint(5, 30)
        cols = random.randint(10, 30)
        taskvars = {'rows': rows, 'cols': cols}

        # Randomly choose how many training examples: 3-6
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1

        # Use the default convenience method, which calls create_input/transform_input
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data



