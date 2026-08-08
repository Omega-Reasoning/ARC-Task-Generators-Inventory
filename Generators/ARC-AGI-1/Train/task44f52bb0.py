from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import random_cell_coloring, retry


class Task44f52bb0Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Input grids are odd squares of size {vars['grid_size']}x{vars['grid_size']} with {color('background_color')} background cells.",
            "2. A nonempty subset of cells is colored {color('pattern_color')}.",
            "3. Some patterns match their mirror image across the vertical center line and other patterns do not.",
            "4. Pattern density and connectedness vary between examples while the classification rule stays fixed.",
        ]
        transformation_reasoning_chain = [
            "1. Reflect the complete input grid across its vertical center line.",
            "2. Compare the reflected grid cell-for-cell with the original input.",
            "3. If they match, output a 1x1 grid colored {color('symmetric_color')}.",
            "4. Otherwise output a 1x1 grid colored {color('asymmetric_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        colors = random.sample(range(1, 10), 3)
        taskvars = {
            "grid_size": random.choice([3, 5, 7]),
            "background_color": 0,
            "pattern_color": colors[0],
            "symmetric_color": colors[1],
            "asymmetric_color": colors[2],
        }
        schedule = [(True, 0.35), (False, 0.35), (True, 0.6), (False, 0.6)]
        train = []
        for symmetric, density in schedule:
            grid = self.create_input(
                taskvars, {"symmetric": symmetric, "density": density}
            )
            train.append({"input": grid, "output": self.transform_input(grid, taskvars)})
        test_grid = self.create_input(
            taskvars,
            {"symmetric": random.choice([True, False]), "density": 0.48},
        )
        return taskvars, {
            "train": train,
            "test": [
                {"input": test_grid, "output": self.transform_input(test_grid, taskvars)}
            ],
        }

    def create_input(self, taskvars, gridvars):
        size = taskvars["grid_size"]
        background = taskvars["background_color"]
        pattern = taskvars["pattern_color"]
        density = gridvars["density"]
        if gridvars["symmetric"]:
            half_width = size // 2 + 1

            def symmetric_sample():
                half = np.full((size, half_width), background, dtype=int)
                half = np.asarray(
                    gridvars.get(
                        "symmetric_half",
                        random_cell_coloring(
                            half,
                            pattern,
                            density=density,
                            background=background,
                        ),
                    ),
                    dtype=int,
                )
                candidate = np.full((size, size), background, dtype=int)
                candidate[:, :half_width] = half
                candidate[:, half_width:] = np.fliplr(half[:, : size // 2])
                return candidate

            try:
                return retry(
                    symmetric_sample,
                    lambda value: np.any(value == pattern),
                    max_attempts=20,
                )
            except ValueError:
                fallback = np.full((size, size), background, dtype=int)
                fallback[size // 2, size // 2] = pattern
                return fallback

        def asymmetric_sample():
            candidate = np.full((size, size), background, dtype=int)
            candidate = np.asarray(
                gridvars.get(
                    "asymmetric_grid",
                    random_cell_coloring(
                        candidate,
                        pattern,
                        density=density,
                        background=background,
                    ),
                ),
                dtype=int,
            )
            return candidate

        try:
            return retry(
                asymmetric_sample,
                lambda value: np.any(value == pattern)
                and not np.array_equal(value, np.fliplr(value)),
                max_attempts=30,
            )
        except ValueError:
            fallback = np.full((size, size), background, dtype=int)
            fallback[0, 0] = pattern
            return fallback

    def transform_input(self, grid, taskvars):
        if np.array_equal(grid, np.fliplr(grid)):
            color = taskvars["symmetric_color"]
        else:
            color = taskvars["asymmetric_color"]
        return np.full((1, 1), color, dtype=int)
