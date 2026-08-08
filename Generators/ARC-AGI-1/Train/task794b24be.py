from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import GridObject


class Task794b24beGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['grid_size']}x{vars['grid_size']} grid filled with {color('background_color')} except for several {color('marker_color')} cells.",
            "2. The input contains between one and {vars['maximum_count']} marker cells.",
            "3. Marker coordinates and connectivity vary freely; only their total number matters.",
            "4. No colors other than {color('background_color')} and {color('marker_color')} occur in the input.",
        ]
        transformation_reasoning_chain = [
            "1. Count all {color('marker_color')} cells in the input regardless of position.",
            "2. Start with a {vars['grid_size']}x{vars['grid_size']} {color('background_color')} output.",
            "3. Encode the count using the fixed order top-left, top-middle, top-right, then center.",
            "4. Color exactly that many leading code positions {color('output_color')} and leave all other output cells unchanged.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        marker_color, output_color = random.sample(range(1, 10), 2)
        taskvars = {
            "background_color": 0,
            "marker_color": marker_color,
            "output_color": output_color,
            "grid_size": 3,
            "maximum_count": 4,
        }
        train = []
        train_inputs = []
        for count in range(1, 5):
            input_grid = self.create_input(taskvars, {"count": count})
            train_inputs.append(input_grid)
            train.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )
        try:
            test_input = retry(
                lambda: self.create_input(taskvars, {"count": 4}),
                lambda grid: not any(
                    np.array_equal(grid, train_grid) for train_grid in train_inputs
                ),
                max_attempts=30,
            )
        except ValueError:
            test_input = np.full((3, 3), 0, dtype=int)
            GridObject(
                {
                    (2, 2, marker_color),
                    (2, 1, marker_color),
                    (1, 2, marker_color),
                    (1, 1, marker_color),
                }
            ).paste(test_input)
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
        background = taskvars["background_color"]
        marker = taskvars["marker_color"]
        size = taskvars["grid_size"]
        count = gridvars["count"]
        sampled_grid = np.full((size, size), background, dtype=int)
        density = (count + 0.1) / (size * size)
        random_cell_coloring(
            sampled_grid,
            marker,
            density=density,
            background=background,
            overwrite=False,
        )
        default_marker_cells = [
            [int(row), int(col)]
            for row, col in np.argwhere(sampled_grid == marker)
        ]
        marker_cells = {
            (int(row), int(col))
            for row, col in gridvars.get(
                "marker_cells",
                default_marker_cells,
            )
        }
        if len(marker_cells) != count:
            raise ValueError("marker_cells must contain exactly count coordinates")
        if any(
            not 0 <= row < size or not 0 <= col < size
            for row, col in marker_cells
        ):
            raise ValueError("marker_cells must lie within the input grid")
        grid = np.full((size, size), background, dtype=int)
        if marker_cells:
            GridObject({
                (row, col, marker) for row, col in marker_cells
            }).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        marker = taskvars["marker_color"]
        output_color = taskvars["output_color"]
        grid_size = taskvars["grid_size"]
        maximum_count = taskvars["maximum_count"]
        count = min(int(np.count_nonzero(grid == marker)), maximum_count)
        output = np.full((grid_size, grid_size), background, dtype=int)
        code_positions = [(0, 0), (0, 1), (0, 2), (1, 1)]
        cells = {
            (row, col, output_color)
            for row, col in code_positions[:count]
        }
        if cells:
            GridObject(cells).paste(output)
        return output
