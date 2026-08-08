from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import GridObject


class Task6430c8c4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains an upper {vars['panel_size']}x{vars['panel_size']} panel of {color('top_color')} cells and {color('background_color')} gaps.",
            "2. A full-width {color('divider_color')} row separates it from a lower {vars['panel_size']}x{vars['panel_size']} panel.",
            "3. The lower panel contains {color('bottom_color')} cells and {color('background_color')} gaps aligned coordinate-for-coordinate with the upper panel.",
            "4. Upper and lower occupancy patterns vary, and some coordinates are empty in both panels.",
        ]
        transformation_reasoning_chain = [
            "1. Locate the {color('divider_color')} row and align the panels immediately above and below it.",
            "2. At each coordinate, test whether both aligned panel cells are {color('background_color')}.",
            "3. Color exactly those jointly empty coordinates {color('output_color')}.",
            "4. Return only the resulting {vars['panel_size']}x{vars['panel_size']} mask with {color('background_color')} elsewhere.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        top_color, bottom_color, divider_color, output_color = random.sample(range(1, 10), 4)
        taskvars = {
            "background_color": 0,
            "top_color": top_color,
            "bottom_color": bottom_color,
            "divider_color": divider_color,
            "output_color": output_color,
            "panel_size": 4,
        }
        masks = [
            [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
            [[1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 1]],
            [[0, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1]],
        ]
        train = []
        for mask in masks:
            grid = self.create_input(taskvars, {"jointly_empty": mask})
            train.append({"input": grid, "output": self.transform_input(grid, taskvars)})
        test_grid = self.create_input(
            taskvars,
            {"jointly_empty": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        )
        return taskvars, {
            "train": train,
            "test": [{"input": test_grid, "output": self.transform_input(test_grid, taskvars)}],
        }

    def create_input(self, taskvars, gridvars):
        size = taskvars["panel_size"]
        background = taskvars["background_color"]
        top_color = taskvars["top_color"]
        bottom_color = taskvars["bottom_color"]
        jointly_empty = np.array(gridvars["jointly_empty"], dtype=bool)

        def sample_panels():
            upper = np.full((size, size), background, dtype=int)
            lower = np.full((size, size), background, dtype=int)
            upper = np.asarray(
                gridvars.get(
                    "upper_panel",
                    random_cell_coloring(
                        upper,
                        top_color,
                        density=gridvars.get(
                            "upper_density", random.uniform(0.35, 0.7)
                        ),
                        background=background,
                    ),
                ),
                dtype=int,
            )
            lower = np.asarray(
                gridvars.get(
                    "lower_panel",
                    random_cell_coloring(
                        lower,
                        bottom_color,
                        density=gridvars.get(
                            "lower_density", random.uniform(0.35, 0.7)
                        ),
                        background=background,
                    ),
                ),
                dtype=int,
            )
            upper[jointly_empty] = background
            lower[jointly_empty] = background
            for row, col in zip(*np.where(~jointly_empty)):
                if upper[row, col] == background and lower[row, col] == background:
                    fill_upper = gridvars.get(
                        f"fill_upper_{int(row)}_{int(col)}",
                        random.choice([True, False]),
                    )
                    if fill_upper:
                        upper[row, col] = top_color
                    else:
                        lower[row, col] = bottom_color
            return upper, lower

        def varied_occupancy(panels):
            upper, lower = panels
            upper_only = (upper != background) & (lower == background)
            lower_only = (upper == background) & (lower != background)
            both = (upper != background) & (lower != background)
            return np.any(upper_only) and np.any(lower_only) and np.any(both)

        try:
            upper, lower = retry(sample_panels, varied_occupancy, max_attempts=40)
        except ValueError:
            upper = np.full((size, size), background, dtype=int)
            lower = np.full((size, size), background, dtype=int)
            index = 0
            for row, col in zip(*np.where(~jointly_empty)):
                if index % 3 in (0, 2):
                    upper[row, col] = top_color
                if index % 3 in (1, 2):
                    lower[row, col] = bottom_color
                index += 1
        divider = np.full((1, size), taskvars["divider_color"], dtype=int)
        return np.vstack([upper, divider, lower])

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        divider = taskvars["divider_color"]
        output_color = taskvars["output_color"]
        panel_size = taskvars["panel_size"]
        divider_rows = [
            row for row in range(grid.shape[0]) if np.all(grid[row, :] == divider)
        ]
        if len(divider_rows) != 1:
            return np.full((panel_size, panel_size), background, dtype=int)
        divider_row = divider_rows[0]
        upper = grid[divider_row - panel_size : divider_row, :panel_size]
        lower = grid[divider_row + 1 : divider_row + panel_size + 1, :panel_size]
        jointly_empty = (upper == background) & (lower == background)
        output = np.full((panel_size, panel_size), background, dtype=int)
        cells = {
            (int(row), int(col), output_color)
            for row, col in zip(*np.where(jointly_empty))
        }
        GridObject(cells).paste(output)
        return output
