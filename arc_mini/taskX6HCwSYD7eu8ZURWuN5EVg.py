# my_corner_rectangle_filler.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# (Optional) If you wish to use input_library or transformation_library,
# you can import them as well. For example:
# from input_library import retry, create_object, ...
# from transformation_library import find_connected_objects, GridObject, ...

class TaskX6HCwSYD7eu8ZURWuN5EVgGenerator(ARCTaskGenerator):
    def __init__(self):
        # -----------------------------
        # 1) Input Reasoning Chain
        # -----------------------------
        observation_chain = [
            "Input grids are of size {vars['rows']}x{vars['columns']}.",
            "Each input grid contains two {color('cell_color')} cells, with the remaining cells being empty (0).",
            "The two {color('cell_color')} cells serve as two corners of a rectangle, either top-left and bottom-right or top-right and bottom-left."
        ]

        # -----------------------------
        # 2) Transformation Reasoning Chain
        # -----------------------------
        reasoning_chain = [
            "To create the output grid, copy the input grid.",
            "The empty (0) cells between the two corner cells are filled with {color('fill_color')} color to complete the rectangle."
        ]

        # -----------------------------
        # 3) Call super().__init__ with these reasoning chains
        # -----------------------------
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        1) Randomly pick task-level variables: 'rows', 'columns', 'cell_color', 'fill_color' 
        2) Generate 4 train pairs and 1 test pair. 
        3) Return a tuple: (taskvars_dict, train_test_data).
        """
        # -----------------------------
        # Randomly assign 'rows' and 'columns' in [4..9]
        # Randomly assign 'cell_color' and 'fill_color' in [1..9]
        # (Optionally enforce fill_color != cell_color for clarity)
        # -----------------------------
        rows = random.randint(4, 9)
        columns = random.randint(4, 9)
        cell_color = random.randint(1, 9)
        fill_color = random.randint(1, 9)
        while fill_color == cell_color:
            fill_color = random.randint(1, 9)

        taskvars = {
            "rows": rows,
            "columns": columns,
            "cell_color": cell_color,
            "fill_color": fill_color
        }

        # -----------------------------
        # Create 4 train examples + 1 test example
        # We'll ensure distinct positions for the two corners in each example
        # -----------------------------
        def generate_one_example():
            input_grid = self.create_input(taskvars, gridvars={})
            output_grid = self.transform_input(input_grid, taskvars)
            return {
                "input": input_grid,
                "output": output_grid
            }

        # We want to ensure each example has *different* corner placements.
        # We'll keep a set of used (corner1, corner2) pairs for uniqueness.
        # Because corners might appear in any order, we store them in a canonical (min, max) form.
        used_corner_pairs = set()
        
        train_examples = []
        while len(train_examples) < 4:
            # Attempt to make an example
            in_grid = self.create_input(taskvars, {})
            # Extract the two colored positions:
            corners = np.argwhere(in_grid == cell_color)
            # Sort them so that order doesn't matter
            corners_sorted = sorted(map(tuple, corners))
            corner_key = (corners_sorted[0], corners_sorted[1])
            if corner_key not in used_corner_pairs:
                used_corner_pairs.add(corner_key)
                out_grid = self.transform_input(in_grid, taskvars)
                train_examples.append({
                    "input": in_grid,
                    "output": out_grid
                })

        # Now create 1 test example
        test_examples = []
        while len(test_examples) < 1:
            in_grid = self.create_input(taskvars, {})
            corners = np.argwhere(in_grid == cell_color)
            corners_sorted = sorted(map(tuple, corners))
            corner_key = (corners_sorted[0], corners_sorted[1])
            if corner_key not in used_corner_pairs:
                used_corner_pairs.add(corner_key)
                out_grid = self.transform_input(in_grid, taskvars)
                test_examples.append({
                    "input": in_grid,
                    "output": out_grid
                })

        # Create the final data structure
        train_test_data = {
            "train": train_examples,
            "test": test_examples
        }

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Creates an input grid of shape taskvars['rows'] x taskvars['columns']
        containing exactly two 'cell_color' cells, placed in distinct rows and columns.
        The rest cells are 0.
        """
        rows = taskvars["rows"]
        columns = taskvars["columns"]
        cell_color = taskvars["cell_color"]

        # Start with an empty grid
        grid = np.zeros((rows, columns), dtype=int)

        # Choose two distinct row/column pairs
        r1, r2 = random.sample(range(rows), 2)
        c1, c2 = random.sample(range(columns), 2)

        grid[r1, c1] = cell_color
        grid[r2, c2] = cell_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Copies the input grid and fills the rectangle between the two 'cell_color' corners 
        with 'fill_color' (only empty cells are changed).
        """
        cell_color = taskvars["cell_color"]
        fill_color = taskvars["fill_color"]

        # Make a copy so we don't modify the original
        output_grid = np.copy(grid)

        # Find the two corner cells of color == cell_color
        corners = np.argwhere(output_grid == cell_color)
        # We expect exactly 2
        if len(corners) != 2:
            # If for some reason there's an invalid grid, just return copy
            return output_grid

        (r1, c1), (r2, c2) = corners
        rmin, rmax = min(r1, r2), max(r1, r2)
        cmin, cmax = min(c1, c2), max(c1, c2)

        # Fill all empty cells in the bounding rectangle
        for rr in range(rmin, rmax + 1):
            for cc in range(cmin, cmax + 1):
                if output_grid[rr, cc] == 0:
                    output_grid[rr, cc] = fill_color

        return output_grid


