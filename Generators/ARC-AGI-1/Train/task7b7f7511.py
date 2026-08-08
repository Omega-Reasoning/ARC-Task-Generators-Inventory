from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import random_cell_coloring, retry


class Task7b7f7511Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is densely colored and consists of exactly {vars['repeat_count']} identical rectangular copies of one base pattern.",
            "2. The copies are concatenated either horizontally or vertically with no separator or padding.",
            "3. Base dimensions, aspect ratio, colors, and cell arrangement vary between examples.",
            "4. The two copies match exactly at every corresponding coordinate.",
        ]
        transformation_reasoning_chain = [
            "1. Compare equal halves along both possible concatenation axes.",
            "2. Identify the axis whose {vars['repeat_count']} halves are identical.",
            "3. Remove the repeated copy and return one unchanged base pattern.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"repeat_count": 2}
        train_specs = [
            (3, 4, "horizontal"),
            (5, 3, "horizontal"),
            (4, 5, "vertical"),
            (3, 7, "vertical"),
        ]
        train = []
        for rows, cols, axis in train_specs:
            input_grid = self.create_input(
                taskvars, {"rows": rows, "cols": cols, "axis": axis}
            )
            train.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )
        test_input = self.create_input(
            taskvars, {"rows": 7, "cols": 5, "axis": "vertical"}
        )
        return taskvars, {
            "train": train,
            "test": [
                {
                    "input": test_input,
                    "output": self.transform_input(test_input, taskvars),
                }
            ],
        }

    def create_input(self, taskvars, gridvars):
        repeat_count = taskvars["repeat_count"]
        rows = gridvars["rows"]
        cols = gridvars["cols"]
        axis = gridvars["axis"]
        colors = [
            int(value)
            for value in gridvars.get(
                "colors",
                random.sample(range(1, 10), random.randint(2, 5)),
            )
        ]

        def sample_base():
            base = np.zeros((rows, cols), dtype=int)
            return np.asarray(
                gridvars.get(
                    "base",
                    random_cell_coloring(
                        base,
                        colors,
                        density=1.0,
                        background=0,
                        overwrite=False,
                    ),
                ),
                dtype=int,
            )

        def base_is_unambiguous(base):
            if len(np.unique(base)) < 2:
                return False
            if cols % repeat_count == 0:
                width = cols // repeat_count
                if np.array_equal(base[:, :width], base[:, width:]):
                    return False
            if rows % repeat_count == 0:
                height = rows // repeat_count
                if np.array_equal(base[:height, :], base[height:, :]):
                    return False
            return True

        try:
            base = retry(sample_base, base_is_unambiguous, max_attempts=40)
        except ValueError:
            base = np.empty((rows, cols), dtype=int)
            for row in range(rows):
                for col in range(cols):
                    base[row, col] = colors[(row * cols + col) % len(colors)]
            base[0, 0] = colors[0]
            base[-1, -1] = colors[1]
        if axis == "horizontal":
            return np.concatenate([base] * repeat_count, axis=1)
        return np.concatenate([base] * repeat_count, axis=0)

    def transform_input(self, grid, taskvars):
        repeat_count = taskvars["repeat_count"]
        rows, cols = grid.shape
        if cols % repeat_count == 0:
            width = cols // repeat_count
            horizontal_parts = [
                grid[:, index * width : (index + 1) * width]
                for index in range(repeat_count)
            ]
            if all(
                np.array_equal(horizontal_parts[0], part)
                for part in horizontal_parts[1:]
            ):
                return np.array(horizontal_parts[0], copy=True)
        if rows % repeat_count == 0:
            height = rows // repeat_count
            vertical_parts = [
                grid[index * height : (index + 1) * height, :]
                for index in range(repeat_count)
            ]
            if all(
                np.array_equal(vertical_parts[0], part)
                for part in vertical_parts[1:]
            ):
                return np.array(vertical_parts[0], copy=True)
        return np.array(grid, copy=True)
