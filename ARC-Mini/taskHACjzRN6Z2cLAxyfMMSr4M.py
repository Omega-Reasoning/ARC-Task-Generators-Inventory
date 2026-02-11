# Filename: arc_task_example_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# We can optionally use functions from these libraries:
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects

class TaskHACjzRN6Z2cLAxyfMMSr4MGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialise the input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}×{vars['cols']}.",
            "Each input grid contains {vars['cols']-4} single-colored objects, where each object consists of at least 2 and at most 4 cells.",
            "The objects are colored {color('object_color1')}, {color('object_color2')}, and {color('object_color3')} and are completely separated by empty (0) cells."
        ]

        # 2) Initialise the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids are created by initializing an empty grid and filling the first row with {color('cell_color')} cells, starting from position (0,1).",
            "The number of {color('cell_color')} cells is equal to the number of {color('object_color1')} objects in the input grids."
        ]

        # 3) Call the parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates the task variables and the training/test grids.
        - We choose random rows and cols
        - We choose 4 distinct color values (cell_color, object_color1, object_color2, object_color3).
        - We create 3-4 training examples and 1 test example using create_grids_default.
        """

        # Randomly pick rows and cols within the specified range
        rows = random.randint(8, 30)
        cols = random.randint(8, 30)

        # Pick 4 distinct colors from 1..9
        color_choices = random.sample(range(1, 10), 4)
        object_color1 = color_choices[0]
        object_color2 = color_choices[1]
        object_color3 = color_choices[2]
        cell_color    = color_choices[3]

        # Make sure we satisfy the requirement that
        # cell_color != object_color1 != object_color2 != object_color3
        # (Already guaranteed by the random.sample approach.)

        taskvars = {
            'rows': rows,
            'cols': cols,
            'object_color1': object_color1,
            'object_color2': object_color2,
            'object_color3': object_color3,
            'cell_color': cell_color
        }

        # Randomly choose how many train examples (3 or 4) and always 1 test example
        nr_train = random.choice([3, 4])
        nr_test = 1

        # Let the parent convenience method create the train/test data
        # by calling create_input() and transform_input() for each example.
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Generate an input grid that satisfies:
          - The grid has dimension rows x cols (from taskvars).
          - It contains exactly (cols - 4) single-colored objects.
          - Each object has between 2..4 cells, is contiguous,
            and colored either object_color1, object_color2, or object_color3.
          - The number of object_color1 objects is strictly greater
            than the number of object_color2 and object_color3 objects.
          - All objects are completely separated by empty cells (no adjacency).
        """

        rows = taskvars['rows']
        cols = taskvars['cols']
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        color3 = taskvars['object_color3']

        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # We must place exactly (cols - 4) objects in total.
        # Let n_total = cols - 4.
        # We choose n2, n3 >= 1, and n1 > n2, n1 > n3 with n1 + n2 + n3 = n_total.
        n_total = cols - 4
        if n_total <= 0:
            # Edge case if cols < 4, but specs say cols >= 8, so this should not happen.
            return grid

        # We want to find a distribution n1, n2, n3 such that:
        #   n1 + n2 + n3 = n_total
        #   n1 > n2 >= 1
        #   n1 > n3 >= 1
        #   n1 >= 1 as well
        # We'll do a simple retry approach.
        def valid_distribution(dist):
            n1d, n2d, n3d = dist
            return (n1d + n2d + n3d == n_total
                    and n2d >= 1 and n3d >= 1 and n1d > n2d and n1d > n3d
                    and n1d > 0)

        def random_distribution():
            # random attempt for n2, n3 in [1..n_total-1]
            n2d = random.randint(1, n_total-1)
            n3d = random.randint(1, n_total-1)
            n1d = n_total - n2d - n3d
            return (n1d, n2d, n3d)

        # Keep trying up to 100 times
        n1, n2, n3 = None, None, None
        for _ in range(100):
            candidate = random_distribution()
            if valid_distribution(candidate):
                n1, n2, n3 = candidate
                break

        if n1 is None:
            # Fallback if not found (should be unlikely):
            # Just pick a trivial distribution like:
            # n2=1, n3=1, n1 = n_total-2 (if that is valid)
            n2 = 1
            n3 = 1
            n1 = n_total - 2
            if n1 <= 1:
                # If even that doesn't work, just return an empty grid
                # but in practice with cols>=8, we always have (cols-4)>=4
                # so n_total >= 4 => n1=2, n2=1, n3=1 => that satisfies n1>n2 and n1>n3
                return grid

        # Now place n1 objects of color1, n2 objects of color2, n3 objects of color3
        # Each object is contiguous, has between 2..4 cells, and is separated from others.
        # We'll define a helper function to create a single random object of the given color
        # with 2..4 cells in total, then place it non-overlapping and with at least 1-cell gap.
        def random_object_matrix(this_color):
            """
            Creates a small contiguous object (height up to 4, width up to 4)
            with 2..4 non-zero cells (all same color).
            """
            while True:
                # create_object can produce a random shape, but we control dimensions
                h = random.randint(1, 4)
                w = random.randint(1, 4)
                obj = create_object(
                    height=h,
                    width=w,
                    color_palette=this_color,
                    contiguity=Contiguity.FOUR,  # or EIGHT
                    background=0
                )
                # check number of non-zero cells
                count = np.count_nonzero(obj)
                if 2 <= count <= 4:
                    return obj

        def can_place(obj_mat, top_r, left_c):
            """
            Checks if we can place obj_mat at (top_r, left_c) in grid
            such that:
             - no overlap with existing objects
             - no adjacency in 8 directions to existing objects
            """
            h, w = obj_mat.shape
            if top_r + h > rows or left_c + w > cols:
                return False
            # For each cell of obj_mat != 0, ensure the corresponding region in grid
            # is free (0) and also all 8 neighbors of that cell are 0.
            for r in range(h):
                for c in range(w):
                    if obj_mat[r, c] != 0:
                        R = top_r + r
                        C = left_c + c
                        # check the 3x3 block around (R, C)
                        for dR in [-1, 0, 1]:
                            for dC in [-1, 0, 1]:
                                nR = R + dR
                                nC = C + dC
                                if 0 <= nR < rows and 0 <= nC < cols:
                                    if grid[nR, nC] != 0:
                                        return False
            return True

        def place_object_in_grid(obj_mat, top_r, left_c):
            """Places obj_mat in grid (in-place)."""
            h, w = obj_mat.shape
            for r in range(h):
                for c in range(w):
                    if obj_mat[r, c] != 0:
                        grid[top_r + r, left_c + c] = obj_mat[r, c]

        def place_n_objects(n, color_value):
            """Attempt to place n objects of the given color."""
            for _ in range(n):
                obj_mat = random_object_matrix(color_value)
                # Try random positions
                placed = False
                for attempt in range(200):
                    rr = random.randint(0, rows - 1)
                    cc = random.randint(0, cols - 1)
                    if can_place(obj_mat, rr, cc):
                        place_object_in_grid(obj_mat, rr, cc)
                        placed = True
                        break
                if not placed:
                    # If we fail to place, raise and let a new input be tried
                    raise ValueError("Could not place object after many attempts.")

        try:
            place_n_objects(n1, color1)
            place_n_objects(n2, color2)
            place_n_objects(n3, color3)
        except ValueError:
            # If we fail to place all objects, we can either retry or just return empty
            # We will do a simple fallback: return an all-zero grid
            # so that create_grids_default tries again.
            return self.create_input(taskvars, gridvars)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transformation per the reasoning chain:
          1) Create an empty grid of the same size.
          2) Fill the first row with {color(cell_color)} cells, 
             starting from position (0,1).
          3) The number of these cells == the count of {color(object_color1)} objects in the input.
        """
        import numpy as np

        # same shape
        rows = taskvars['rows']
        cols = taskvars['cols']
        out_grid = np.zeros((rows, cols), dtype=int)

        color1 = taskvars['object_color1']
        cell_color = taskvars['cell_color']

        # Count how many color1 objects are in the input
        # We'll use the transformation_library's find_connected_objects with monochromatic=True
        all_objs = find_connected_objects(grid, diagonal_connectivity=False,
                                          background=0, monochromatic=True)
        # Filter to objects that contain color1
        color1_objs = all_objs.with_color(color1)
        # The number of color1 objects
        count_color1_objects = len(color1_objs)

        # Fill row 0, columns [1 .. 1 + count_color1_objects - 1] with cell_color
        # Make sure we don't exceed the grid's width
        max_fill = min(count_color1_objects, cols - 1)
        for c in range(1, 1 + max_fill):
            out_grid[0, c] = cell_color

        return out_grid



