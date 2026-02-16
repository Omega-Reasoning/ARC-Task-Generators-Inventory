# my_arc_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

# Optional imports from the libraries:
from Framework.input_library import create_object
from Framework.transformation_library import find_connected_objects, GridObject

class Task4iT7fWTpKsv65dGdade2gfGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialize the input reasoning chain
        self.input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain two same-colored objects, each forming a 2x2 block.",
            "The first object is located in the top-left quadrant, and the second object is located in the bottom-right quadrant of the grid.",
            "All other cells are empty (0)."
        ]
        # 2) Initialize the transformation reasoning chain
        self.transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and repositioning the second object so that its top-left corner aligns with the bottom-right corner of the first object.",
            "Both objects retain their shape and color."

        ]
        # 3) Call the parent constructor
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)

    def create_input(self,
                     taskvars,
                     gridvars) -> np.ndarray:
        """
        Create a single input grid of size NxN (N odd, 5 <= N <= 30),
        with two 2x2 blocks of the same color:
          - First block strictly in the top-left quadrant (not touching the middle),
          - Second block strictly in the bottom-right quadrant (not touching the middle).
        """

        color = gridvars["color"]
        # Use the grid_size stored in taskvars
        size = taskvars["grid_size"]
        grid = np.zeros((size, size), dtype=int)

        # ---------- Place the first 2×2 block in top-left quadrant -----------
        row_max_1 = max((size // 2) - 2, 0)
        col_max_1 = max((size // 2) - 2, 0)
        r1 = random.randint(0, row_max_1)
        c1 = random.randint(0, col_max_1)
        grid[r1:r1+2, c1:c1+2] = color

        # ---------- Place the second 2×2 block in bottom-right quadrant -------
        row_min_2 = min((size // 2) + 1, size - 2)
        col_min_2 = min((size // 2) + 1, size - 2)
        r2 = random.randint(row_min_2, size - 2)
        c2 = random.randint(col_min_2, size - 2)
        grid[r2:r2+2, c2:c2+2] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transformation:
         1) Copy the grid
         2) Detect the two objects (2x2 blocks, same color)
         3) Move the second object so its top-left corner
            diagonally touches the bottom-right corner of the first object.
        """

        output_grid = np.copy(grid)

        # Find the 2 connected objects
        objects = find_connected_objects(
            output_grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=True
        )
        if len(objects) != 2:
            return output_grid

        # Sort so the "first" object is the one that appears top-left
        obj_list = sorted(
            objects,
            key=lambda obj: (
                min(r for (r, _, _) in obj.cells),
                min(c for (_, c, _) in obj.cells)
            )
        )
        first_obj = obj_list[0]
        second_obj = obj_list[1]

        # Bottom-right corner of the first object
        frmax = max(r for (r, _, _) in first_obj.cells)
        fcmax = max(c for (_, c, _) in first_obj.cells)

        # Top-left corner of the second object
        srmin = min(r for (r, _, _) in second_obj.cells)
        scmin = min(c for (_, c, _) in second_obj.cells)

        # Remove the second object from the grid
        second_obj.cut(output_grid)

        # Shift second so it is diagonally adjacent to the first
        dr = (frmax - srmin + 1)
        dc = (fcmax - scmin + 1)

        new_cells = set()
        for (r, c, col) in second_obj.cells:
            new_cells.add((r + dr, c + dc, col))
        second_obj.cells = new_cells

        # Paste it back into the output grid
        second_obj.paste(output_grid, overwrite=True, background=0)

        return output_grid

    def create_grids(self):
        """
        Generate training and test data:
          - 3 or 4 train examples + 1 test example
          - Each example uses a different color
          - All share the same grid_size (odd, from 5..30)
        """
        # 1) Pick the grid_size (must be odd between 5 and 30)
        valid_sizes = [n for n in range(5, 31) if n % 2 == 1]
        chosen_size = random.choice(valid_sizes)

        # 2) Randomly decide how many training examples
        nr_train = random.choice([3, 4])
        nr_test = 1

        # 3) Pick distinct colors from 1..9
        distinct_colors = random.sample(range(1, 10), nr_train + nr_test)

        # We'll store the chosen_size in taskvars
        taskvars = {"grid_size": chosen_size}

        train_examples = []
        for i in range(nr_train):
            color_i = distinct_colors[i]
            # Pass both 'grid_size' and 'color'
            in_grid = self.create_input(taskvars, {"color": color_i})
            out_grid = self.transform_input(in_grid, taskvars)
            train_examples.append(GridPair(input=in_grid, output=out_grid))

        test_examples = []
        for i in range(nr_test):
            color_i = distinct_colors[nr_train + i]
            in_grid = self.create_input(taskvars, {"color": color_i})
            out_grid = self.transform_input(in_grid, taskvars)
            test_examples.append(GridPair(input=in_grid, output=out_grid))

        train_test_data = TrainTestData(train=train_examples, test=test_examples)
        return taskvars, train_test_data


