from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# Optionally import from input_library if needed for more elaborate randomization
# (currently not strictly used here because our input generation is straightforward)
# from input_library import create_object, retry, Contiguity

# We do use transformation_library nomenclature if needed
# from transformation_library import find_connected_objects, GridObject, GridObjects

class TaskDmDQsjVVzFXBMDD4RDkG8pGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains a single {color('object_color1')} object, which is a rectangular block that fills the entire interior of the grid, leaving only the borders of the grid empty (0)."
        ]
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and adding {color('object_color2')} lines that intersect at the vertical and horizontal center of the {color('object_color1')} object, forming a {color('object_color2')} cross pattern.",
            "If the {color('object_color1')} object has an odd width or length, a single {color('object_color2')} line is used. If it has an even width or length, two {color('object_color2')} lines are used to fully cover the center.",
            "The {color('object_color2')} lines extend from the first row to the last row and from the first column to the last column."
        ]
        # 3) Call super constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> (dict, TrainTestData):
        """
        We create 3-4 train grids and 2 test grids with the following constraints:
        * We have at least one square grid with odd dimension in both train and test.
        * We have at least one grid with even rows and columns in both train and test.
        * The size of each input grid is different (including test grids).
        * 5 <= rows, cols <= 30
        * object_color1 != object_color2
        """
        # 1. Randomly choose how many training examples
        num_train = random.choice([3, 4])
        num_test = 2
        total_needed = num_train + num_test

        # 2. Pick distinct colors for object_color1 and object_color2
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)

        taskvars = {
            "object_color1": color1,
            "object_color2": color2
        }

        # Helper functions to get random odd or even dimensions in [5..30]
        def random_odd_dim():
            return random.choice([x for x in range(5, 31) if x % 2 == 1])

        def random_even_dim():
            return random.choice([x for x in range(6, 31) if x % 2 == 0])

        # We need:
        #   * 1 odd-square in train, 1 odd-square in test
        #   * 1 even x even in train, 1 even x even in test
        # That accounts for at least 4 distinct grids. The rest (if any) are random distinct sizes.

        sizes_set = set()

        # We define a helper to get a distinct dimension pair
        def add_distinct_size(rows, cols):
            while (rows, cols) in sizes_set:
                # If it collides, pick new random
                rows = random.randint(5, 30)
                cols = random.randint(5, 30)
            sizes_set.add((rows, cols))
            return (rows, cols)

        # Force train odd-square
        odd_sq_train = add_distinct_size(*(2 * [random_odd_dim()]))
        # Force train even-even
        even_train_rows = random_even_dim()
        even_train_cols = random_even_dim()
        even_even_train = add_distinct_size(even_train_rows, even_train_cols)

        # If we still have more needed in train, fill with random distinct
        train_sizes = [odd_sq_train, even_even_train]
        while len(train_sizes) < num_train:
            r = random.randint(5, 30)
            c = random.randint(5, 30)
            new_size = add_distinct_size(r, c)
            train_sizes.append(new_size)

        # Force test odd-square
        odd_sq_test = add_distinct_size(*(2 * [random_odd_dim()]))
        # Force test even-even
        even_test_rows = random_even_dim()
        even_test_cols = random_even_dim()
        even_even_test = add_distinct_size(even_test_rows, even_test_cols)

        test_sizes = [odd_sq_test, even_even_test]

        # We'll not add more test sizes since we want exactly 2 test

        # Optionally shuffle them for variety; 
        # or we can keep them in the order we inserted to satisfy the "one example should be" constraints
        # random.shuffle(train_sizes)
        # random.shuffle(test_sizes)

        # 3. Create the actual train/test grids
        train_data = []
        for sz in train_sizes:
            grid_in = self.create_input(taskvars, {"rows": sz[0], "cols": sz[1]})
            grid_out = self.transform_input(grid_in, taskvars)
            train_data.append(GridPair(input=grid_in, output=grid_out))

        test_data = []
        for sz in test_sizes:
            grid_in = self.create_input(taskvars, {"rows": sz[0], "cols": sz[1]})
            grid_out = self.transform_input(grid_in, taskvars)
            test_data.append(GridPair(input=grid_in, output=grid_out))

        return taskvars, TrainTestData(train=train_data, test=test_data)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid with the specified rows, cols (in gridvars).
        A single object_color1 block fills the interior, leaving a 1-cell-wide
        border of 0s around the edges.
        """
        rows = gridvars.get("rows", 5)
        cols = gridvars.get("cols", 5)
        color1 = taskvars["object_color1"]

        # Create the empty grid (all zeros)
        grid = np.zeros((rows, cols), dtype=int)

        # Fill the interior
        if rows > 2 and cols > 2:
            grid[1:rows-1, 1:cols-1] = color1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Copy the input grid and add color(object_color2) lines that intersect
        at the vertical and horizontal center of the color(object_color1) object,
        forming a cross. For even interior dimension, use two 'center lines'.
        """
        color2 = taskvars["object_color2"]

        out_grid = np.copy(grid)
        nrows, ncols = out_grid.shape

        # Interior bounding box
        top, left = 0, 0
        bottom, right = nrows - 1, ncols - 1
        if bottom < top or right < left:
            # If the grid is too small, just return the copy
            return out_grid

        interior_height = bottom - top + 1
        interior_width = right - left + 1

        # Center rows: one if odd, two if even
        if interior_height % 2 == 0:
            center_rows = [top + (interior_height // 2) - 1,
                           top + (interior_height // 2)]
        else:
            center_rows = [top + (interior_height // 2)]

        # Center cols: one if odd, two if even
        if interior_width % 2 == 0:
            center_cols = [left + (interior_width // 2) - 1,
                           left + (interior_width // 2)]
        else:
            center_cols = [left + (interior_width // 2)]

        # Paint the horizontal center line(s)
        for row in center_rows:
            for c in range(left, right + 1):
                out_grid[row, c] = color2

        # Paint the vertical center line(s)
        for col in center_cols:
            for r in range(top, bottom + 1):
                out_grid[r, col] = color2

        return out_grid



