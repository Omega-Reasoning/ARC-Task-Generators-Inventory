from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.transformation_library import find_connected_objects


class Taskea786f4aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is an odd square filled with one foreground color.",
            "2. Exactly one {color('background_color')} cell replaces the foreground at the geometric center.",
            "3. The square dimension and foreground color vary between examples.",
            "4. The four corners are initially foreground and are equally distant from the center seed.",
        ]
        transformation_reasoning_chain = [
            "1. Locate the unique {color('background_color')} seed at the center of the odd square.",
            "2. Follow both diagonals through that center toward all four corners.",
            "3. Recolor every cell on those two diagonals with {color('background_color')}.",
            "4. Preserve every foreground cell that is not part of the centered X.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"background_color": 0}
        train_gridvars = [
            {"size": 3},
            {"size": 5},
            {"size": 7},
            {"size": 9},
        ]
        test_gridvars = {"size": 13}

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
        background = taskvars["background_color"]
        size = gridvars["size"]
        foreground = gridvars.get(
            "foreground_color",
            random.choice([color for color in range(10) if color != background]),
        )
        if (
            isinstance(foreground, (bool, np.bool_))
            or not isinstance(foreground, (int, np.integer))
            or int(foreground) not in {
                color for color in range(10) if color != background
            }
        ):
            raise ValueError(
                "foreground_color must be an integer ARC color distinct from background"
            )
        foreground = int(foreground)
        grid = np.full((size, size), foreground, dtype=int)
        grid[size // 2, size // 2] = background
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        foreground = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        )
        if len(foreground) == 0:
            return grid.copy()
        seeds = np.argwhere(grid == background)
        if len(seeds) != 1:
            return grid.copy()
        center_row, center_col = map(int, seeds[0])
        output = grid.copy()
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if abs(row - center_row) == abs(col - center_col):
                    output[row, col] = background
        return output
