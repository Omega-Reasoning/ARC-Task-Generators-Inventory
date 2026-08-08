from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
import numpy as np
import random


class Taskdc0a314fGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['grid_size']}x{vars['grid_size']} multicolored pattern that is symmetric under both horizontal and vertical reflection.",
            "2. One {vars['patch_size']}x{vars['patch_size']} region has been overwritten by a solid {color('mask_color')} mask.",
            "3. The mask position varies and can overlap one reflection axis, but at least one reflected counterpart of every hidden cell remains visible.",
            "4. Pattern colors and local geometry vary between examples while the canvas, mask role, and reflection rule remain fixed.",
        ]
        transformation_reasoning_chain = [
            "1. Locate the tight {vars['patch_size']}x{vars['patch_size']} rectangle of {color('mask_color')} cells.",
            "2. For each masked coordinate, inspect its horizontal, vertical, and double-reflection counterparts in the {vars['grid_size']}x{vars['grid_size']} input.",
            "3. Copy a counterpart that is not {color('mask_color')} into the same relative position of a new output patch.",
            "4. Return only the reconstructed patch, preserving its orientation at the masked location.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {
            "grid_size": 16,
            "patch_size": 5,
            "mask_color": random.randint(1, 9),
        }

        def make_pair(mask_row, mask_col):
            palette = random.sample(
                [color for color in range(1, 10) if color != taskvars["mask_color"]],
                4,
            )
            input_grid = self.create_input(
                taskvars,
                {
                    "mask_row": mask_row,
                    "mask_col": mask_col,
                    "palette": palette,
                },
            )
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [
            make_pair(0, 2),
            make_pair(2, 9),
            make_pair(5, 0),
            make_pair(0, 7),
        ]
        test = [make_pair(6, 9)]
        return taskvars, {"train": train, "test": test}

    def create_input(self, taskvars, gridvars):
        half = taskvars["grid_size"] // 2

        def sample_quadrant():
            sampled = np.asarray(
                gridvars.get(
                    "quadrant",
                    create_object(
                        half,
                        half,
                        gridvars["palette"],
                        contiguity=Contiguity.NONE,
                        background=0,
                    ),
                ),
                dtype=int,
            )
            sampled = sampled.copy()
            empty = np.argwhere(sampled == 0)
            for row, col in empty:
                sampled[row, col] = gridvars.get(
                    f"fill_{int(row)}_{int(col)}",
                    random.choice(gridvars["palette"]),
                )
            return sampled

        try:
            quadrant = retry(
                sample_quadrant,
                lambda value: len(np.unique(value)) >= 3,
                max_attempts=30,
            )
        except ValueError:
            quadrant = np.fromfunction(
                lambda row, col: (row + 2 * col) % len(gridvars["palette"]),
                (half, half),
                dtype=int,
            )
            quadrant = np.asarray(
                [[gridvars["palette"][int(value)] for value in row] for row in quadrant],
                dtype=int,
            )
        top = np.hstack([quadrant, np.fliplr(quadrant)])
        grid = np.vstack([top, np.flipud(top)])
        row = gridvars["mask_row"]
        col = gridvars["mask_col"]
        size = taskvars["patch_size"]
        grid[row : row + size, col : col + size] = taskvars["mask_color"]
        return grid

    def transform_input(self, grid, taskvars):
        mask_color = taskvars["mask_color"]
        patch_size = taskvars["patch_size"]
        mask_cells = np.argwhere(grid == mask_color)
        if mask_cells.size == 0:
            return np.zeros((patch_size, patch_size), dtype=int)
        row_start, col_start = np.min(mask_cells, axis=0)
        row_stop, col_stop = np.max(mask_cells, axis=0) + 1
        output = np.zeros((row_stop - row_start, col_stop - col_start), dtype=int)
        rows, cols = grid.shape
        for row in range(row_start, row_stop):
            for col in range(col_start, col_stop):
                counterparts = [
                    (row, cols - 1 - col),
                    (rows - 1 - row, col),
                    (rows - 1 - row, cols - 1 - col),
                ]
                value = 0
                for other_row, other_col in counterparts:
                    if grid[other_row, other_col] != mask_color:
                        value = grid[other_row, other_col]
                        break
                output[row - row_start, col - col_start] = value
        return output
