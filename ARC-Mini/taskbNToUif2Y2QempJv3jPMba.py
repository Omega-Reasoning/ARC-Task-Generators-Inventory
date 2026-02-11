# my_arc_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# We'll still import 'retry' in case you want it, but we'll no longer rely on it for shape generation:
from input_library import retry, Contiguity  # Not strictly necessary now
# from transformation_library import find_connected_objects, GridObject  # If desired

class TaskbNToUif2Y2QempJv3jPMbaGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "The input grids contain a single object, 4-way connected cells of the same color, which is always touching the top edge.",
            "The object can only be either of {color('object_color1')} or {color('object_color2')} color.",
            "All other cells are empty(0)."
        ]

        # 2. Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by rotating the input grid 90 degrees clockwise.",
            "After the rotation, the color of the object changes.",
            "If the object in the input grid is {color('object_color1')}, the rotated object becomes {color('object_color3')}.",
            "If the object in the input grid is {color('object_color2')}, the rotated object becomes {color('object_color4')}."
        ]

        # 3. Call superclass constructor with our chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        1) Randomly pick the grid size (between 5 and 20).
        2) Pick 4 distinct colors (1..9) for object_color1, object_color2, object_color3, object_color4.
        3) Create 3-4 training pairs (we do 4 here) and 2 test pairs:
           - Make sure we use object_color1 for at least one training and one test example.
           - Make sure we use object_color2 for at least one training and one test example.
        4) Return task variables + train/test data.
        """

        # 1) Random grid size
        grid_size = random.randint(5, 30)

        # 2) Pick 4 distinct colors from 1..9
        distinct_colors = random.sample(range(1, 10), 4)
        object_color1, object_color2, object_color3, object_color4 = distinct_colors

        taskvars = {
            'grid_size': grid_size,
            'object_color1': object_color1,
            'object_color2': object_color2,
            'object_color3': object_color3,
            'object_color4': object_color4
        }

        # 3) Create training pairs
        train_data = []
        # We produce exactly 4 training examples: two with object_color1, two with object_color2.
        colors_for_training = [object_color1, object_color1, object_color2, object_color2]
        random.shuffle(colors_for_training)

        for col in colors_for_training:
            inp = self.create_input(taskvars, {'chosen_color': col})
            outp = self.transform_input(inp, taskvars)
            train_data.append(GridPair(input=inp, output=outp))

        # 4) Create test pairs
        # We'll produce 2 test examples: one with object_color1 and one with object_color2
        test_data = []
        colors_for_testing = [object_color1, object_color2]
        random.shuffle(colors_for_testing)

        for col in colors_for_testing:
            inp = self.create_input(taskvars, {'chosen_color': col})
            outp = self.transform_input(inp, taskvars)
            test_data.append(GridPair(input=inp, output=outp))

        train_test_data = TrainTestData(train=train_data, test=test_data)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Creates an NxN grid with:
          - A single 4-way connected object in either object_color1 or object_color2
          - That object must:
            * Touch the top edge
            * Have at least 4 cells in the first row
            * Have total size between 10 and 15 cells
          - All other cells = 0

        Implementation: We explicitly build a shape from row 0 using a BFS/DFS approach,
        ensuring it meets the constraints, rather than random-filling submatrices.
        """
        N = taskvars['grid_size']
        color = gridvars['chosen_color']

        # Initialize an empty NxN grid
        grid = np.zeros((N, N), dtype=int)

        # We'll attempt up to e.g. 100 times to build a valid shape (should be very likely with BFS).
        for _attempt in range(100):
            # Start BFS from a random column on row 0
            start_col = random.randint(0, N - 1)
            # We'll build a list of cells we fill.
            shape_cells = [(0, start_col)]
            visited = set(shape_cells)

            # BFS/DFS queue
            queue = [(0, start_col)]

            # We'll add cells randomly until we have between 10 and 15 total cells, or we can't expand.
            while queue and len(shape_cells) < 15:
                r, c = queue.pop(0)
                # Check 4 neighbors
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r+dr, c+dc
                    # Only expand if in bounds, not visited, and row>=0
                    if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in visited:
                        # We'll add new neighbor with some probability so shape is somewhat random
                        # but we need at least 10 cells, so let's be fairly permissive until we near 15
                        if len(shape_cells) < 15:
                            # Probability-based expansion
                            if random.random() < 0.8:  # 80% chance to expand
                                visited.add((nr, nc))
                                shape_cells.append((nr, nc))
                                queue.append((nr, nc))

            # Now we have some shape with up to 15 cells. 
            # If we have fewer than 10 cells, try again.
            if len(shape_cells) < 10:
                continue

            # Check if at least 4 cells in row 0
            top_row_count = sum(1 for (r, _) in shape_cells if r == 0)
            if top_row_count < 4:
                continue

            # We have a valid shape of 10..15 cells, 4-way contiguous, touches top row.
            # Place it in the grid
            for (r, c) in shape_cells:
                grid[r, c] = color

            return grid

        # If we cannot find a shape in 100 tries (very unlikely), raise an error:
        raise ValueError("Could not place a shape satisfying the constraints after 100 BFS attempts.")

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Rotate the grid by 90 degrees clockwise, then recolor:
          - if object_color1 => object_color3
          - if object_color2 => object_color4
        """
        # np.rot90(..., k=-1) rotates 90 degrees clockwise
        rotated = np.rot90(grid, k=-1)

        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        color3 = taskvars['object_color3']
        color4 = taskvars['object_color4']

        out_grid = rotated.copy()
        out_grid[out_grid == color1] = color3
        out_grid[out_grid == color2] = color4

        return out_grid



