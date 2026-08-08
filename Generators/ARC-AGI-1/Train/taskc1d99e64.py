from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import random_cell_coloring, retry


class Taskc1d99e64Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each grid contains a dense random texture in one visible foreground color on empty cells.",
            "2. One or more complete rows, complete columns, or both have been left entirely empty.",
            "3. Every row or column not designated as a separator contains at least one foreground cell.",
            "4. Grid dimensions, texture, foreground color, and separator positions vary between examples.",
            "5. The future separator color {color('separator_color')} does not occur in the input.",
        ]
        transformation_reasoning_chain = [
            "1. Identify every row whose cells are all empty and every column whose cells are all empty.",
            "2. Fill each identified row completely with {color('separator_color')}.",
            "3. Fill each identified column completely with {color('separator_color')}.",
            "4. Keep all original foreground cells and every non-separator empty cell unchanged.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"separator_color": random.randint(1, 9)}
        train_gridvars = [
            {"rows": 8, "cols": 10, "empty_rows": 1, "empty_cols": 0, "density": 0.46},
            {"rows": 10, "cols": 8, "empty_rows": 0, "empty_cols": 1, "density": 0.52},
            {"rows": 10, "cols": 12, "empty_rows": 1, "empty_cols": 1, "density": 0.42},
            {"rows": 12, "cols": 13, "empty_rows": 2, "empty_cols": 2, "density": 0.57},
        ]
        test_gridvars = {
            "rows": 15,
            "cols": 16,
            "empty_rows": 3,
            "empty_cols": 3,
            "density": 0.48,
        }

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        return taskvars, {
            "train": [make_pair(gridvars) for gridvars in train_gridvars],
            "test": [make_pair(test_gridvars)],
        }

    def create_input(self, taskvars, gridvars):
        rows, cols = gridvars["rows"], gridvars["cols"]
        expected_rows = gridvars["empty_rows"]
        expected_cols = gridvars["empty_cols"]
        foreground_choices = [
            color
            for color in range(1, 10)
            if color != taskvars["separator_color"]
        ]
        foreground_color = gridvars.get(
            "foreground_color",
            random.choice(foreground_choices),
        )

        def sample_grid():
            grid = np.zeros((rows, cols), dtype=int)
            random_cell_coloring(
                grid,
                foreground_color,
                density=gridvars["density"],
                background=0,
            )
            grid = np.asarray(
                gridvars.get("texture", grid),
                dtype=int,
            ).copy()
            row_separators = gridvars.get(
                "row_separators",
                random.sample(range(rows), expected_rows),
            )
            col_separators = gridvars.get(
                "col_separators",
                random.sample(range(cols), expected_cols),
            )
            if row_separators:
                grid[row_separators, :] = 0
            if col_separators:
                grid[:, col_separators] = 0
            return grid

        def has_exact_empty_axes(grid):
            return bool(
                np.count_nonzero(np.all(grid == 0, axis=1)) == expected_rows
                and np.count_nonzero(np.all(grid == 0, axis=0)) == expected_cols
                and np.count_nonzero(grid == foreground_color) >= 4
            )

        try:
            return retry(sample_grid, has_exact_empty_axes, max_attempts=60)
        except ValueError:
            grid = np.full((rows, cols), foreground_color, dtype=int)
            row_separators = [
                (index + 1) * rows // (expected_rows + 1)
                for index in range(expected_rows)
            ]
            col_separators = [
                (index + 1) * cols // (expected_cols + 1)
                for index in range(expected_cols)
            ]
            if row_separators:
                grid[row_separators, :] = 0
            if col_separators:
                grid[:, col_separators] = 0
            return grid

    def transform_input(self, grid, taskvars):
        output = grid.copy()
        empty_rows = np.all(grid == 0, axis=1)
        empty_cols = np.all(grid == 0, axis=0)
        output[empty_rows, :] = taskvars["separator_color"]
        output[:, empty_cols] = taskvars["separator_color"]
        return output
