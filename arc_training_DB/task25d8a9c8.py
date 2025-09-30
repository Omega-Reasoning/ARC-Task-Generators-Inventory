import numpy as np
import random

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

# Optional, if you want to use the input library or transformation library:
# from input_library import create_object, retry, random_cell_coloring, enforce_object_width, enforce_object_height, Contiguity
# from transformation_library import find_connected_objects, GridObject, GridObjects

class ARCTask25d8a9c8Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}",
            "All the cells in the input grid have color(between 1-9).",
            "Exactly one row has all the cells of the same color.",
            "No cells in the input grid are empty(0)."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First identify the row where all the cells have the same color.",
            "Change all the cells found in the above row to the color {color('output_color')}",
            "Remaining all the colors are empty(0) in the output grid."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # 1. Create task variables
        # We want rows in [10..20]
        rows = random.randint(10, 20)
        output_color = random.randint(1, 9)
        taskvars = {
            'rows': rows,
            'output_color': output_color
        }

        # 2. Create train and test grids
        # We'll produce 3 or 4 training examples randomly, and 1 test example.
        nr_train = random.choice([3, 4])
        nr_test = 1
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        # We'll create an NxN grid, no zeros, exactly one uniform row.
        grid = np.zeros((rows, rows), dtype=int)

        # 1) Choose which row to be uniform.
        uniform_row_index = random.randrange(rows)
        uniform_color = random.randint(1, 9)

        # 2) For each row:
        for r in range(rows):
            if r == uniform_row_index:
                # fill entire row with uniform_color
                grid[r, :] = uniform_color
            else:
                # fill row with random colors, ensuring it's not uniform
                # start by filling with random colors in [1..9]
                for c in range(rows):
                    grid[r, c] = random.randint(1, 9)
                # to ensure it's not uniform, we can forcibly override at least one cell
                # to be a different color if the row accidentally was uniform.
                # check if the entire row is the same color:
                first_color = grid[r, 0]
                if np.all(grid[r, :] == first_color):
                    # override one cell with a different color
                    override_col = random.randrange(rows)
                    new_color = random.randint(1, 9)
                    while new_color == first_color:
                        new_color = random.randint(1, 9)
                    grid[r, override_col] = new_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars):
        # Implementation of the transformation reasoning chain:
        # 1) Output grid same size as input.
        rows = grid.shape[0]
        out_grid = np.zeros_like(grid)

        # 2) Identify the row where all cells have the same color.
        # We'll find exactly one such row per constraints.
        found_uniform_row = None
        for r in range(rows):
            row_vals = grid[r, :]
            if np.all(row_vals == row_vals[0]):
                found_uniform_row = r
                break

        if found_uniform_row is None:
            # According to constraints, it should never happen. But just in case, do no changes.
            return out_grid

        # 3) Change the entire row to output_color in out_grid.
        out_color = taskvars['output_color']
        out_grid[found_uniform_row, :] = out_color

        # 4) All remaining cells in out_grid are 0 (already done by initialization).

        return out_grid


