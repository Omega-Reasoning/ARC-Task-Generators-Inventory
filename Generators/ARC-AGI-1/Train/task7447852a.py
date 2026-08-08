from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects


class Task7447852aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input has {vars['grid_rows']} rows, a variable width, and a {color('background_color')} background.",
            "2. A one-cell-thick {color('wall_color')} diagonal zigzag repeatedly travels between the top and bottom borders.",
            "3. Under four-neighbor connectivity, the zigzag separates the background into an ordered sequence of pockets.",
            "4. The first and last pockets may be clipped by the left or right grid boundary.",
        ]
        transformation_reasoning_chain = [
            "1. Find the four-connected {color('background_color')} pockets separated by the {color('wall_color')} zigzag.",
            "2. Order the pockets from left to right by their horizontal position.",
            "3. Beginning with the leftmost pocket, select every {vars['selection_period']}th pocket.",
            "4. Recolor every selected pocket {color('fill_color')} and preserve the zigzag and unselected pockets.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        wall_color, fill_color = random.sample(range(1, 10), 2)
        grid_rows = random.randint(3, 5)
        selection_period = random.randint(2, 4)
        taskvars = {
            "background_color": 0,
            "wall_color": wall_color,
            "fill_color": fill_color,
            "grid_rows": grid_rows,
            "selection_period": selection_period,
        }
        span = grid_rows - 1
        first_width = selection_period * span + 2
        train_widths = [
            first_width,
            first_width + 1,
            first_width + span,
            first_width + span + 2,
        ]
        train = []
        for index, width in enumerate(train_widths):
            input_grid = self.create_input(
                taskvars,
                {"cols": width, "start_top": index % 2 == 0},
            )
            train.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )
        try:
            test_width = retry(
                lambda: random.randint(max(train_widths) + 1, 30),
                lambda width: width not in train_widths,
                max_attempts=20,
            )
        except ValueError:
            test_width = min(30, max(train_widths) + span + 1)
        test_input = self.create_input(
            taskvars,
            {"cols": test_width, "start_top": False},
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
        background = taskvars["background_color"]
        wall_color = taskvars["wall_color"]
        rows = taskvars["grid_rows"]
        cols = gridvars["cols"]
        start_top = gridvars["start_top"]
        span = rows - 1
        cycle = 2 * span
        wall_cells = set()
        for col in range(cols):
            phase = col % cycle
            row = phase if phase <= span else cycle - phase
            if not start_top:
                row = span - row
            wall_cells.add((row, col, wall_color))
        grid = np.full((rows, cols), background, dtype=int)
        GridObject(wall_cells).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        wall_color = taskvars["wall_color"]
        fill_color = taskvars["fill_color"]
        period = taskvars["selection_period"]
        output = np.array(grid, copy=True)
        pockets = list(
            find_connected_objects(
                grid,
                diagonal_connectivity=False,
                background=wall_color,
                monochromatic=True,
            )
        )
        pockets.sort(
            key=lambda obj: (
                sum(col for _, col, _ in obj.cells) / len(obj.cells),
                sum(row for row, _, _ in obj.cells) / len(obj.cells),
            )
        )
        for index, pocket in enumerate(pockets):
            if index % period == 0:
                pocket.copy().color_all(fill_color).paste(output)
        return output
