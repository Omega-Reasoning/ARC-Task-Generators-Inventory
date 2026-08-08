from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import retry
from Framework.transformation_library import GridObject, get_objects_from_raster


class Task6773b310Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['panel_rows']} by {vars['panel_cols']} array of square {vars['panel_size']}x{vars['panel_size']} panels.",
            "2. Full {color('divider_color')} rows and columns separate neighboring panels, while all other unmarked cells use {color('background_color')}.",
            "3. Every panel contains a varying number of {color('marker_color')} cells at varying coordinates.",
            "4. Panels with exactly {vars['target_count']} markers may occur at any positions in the panel array.",
        ]
        transformation_reasoning_chain = [
            "1. Use the full {color('divider_color')} lines to isolate the {vars['panel_size']}x{vars['panel_size']} panels.",
            "2. Count the {color('marker_color')} cells inside each panel independently.",
            "3. Create a {vars['panel_rows']}x{vars['panel_cols']} {color('background_color')} output grid aligned with the panel array.",
            "4. Color an output cell {color('output_color')} exactly when its corresponding panel contains {vars['target_count']} {color('marker_color')} cells.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        divider_color, marker_color, output_color = random.sample(range(1, 10), 3)
        panel_size = random.randint(3, 5)
        panel_rows = random.randint(3, 4)
        panel_cols = random.randint(3, 4)
        target_count = random.randint(2, min(4, panel_size * panel_size - 2))
        taskvars = {
            "background_color": 0,
            "divider_color": divider_color,
            "marker_color": marker_color,
            "output_color": output_color,
            "panel_size": panel_size,
            "panel_rows": panel_rows,
            "panel_cols": panel_cols,
            "target_count": target_count,
        }

        last_row = panel_rows - 1
        last_col = panel_cols - 1
        middle_row = panel_rows // 2
        middle_col = panel_cols // 2
        train_masks = [
            {(0, 0), (last_row, last_col)},
            {(0, last_col), (last_row, 0)},
            {(0, middle_col), (middle_row, 0), (last_row, middle_col)},
            {(middle_row, 0), (middle_row, middle_col), (middle_row, last_col)},
        ]
        test_mask = {
            (0, middle_col),
            (middle_row, middle_col),
            (last_row, middle_col),
            (middle_row, last_col),
        }

        train = []
        for target_positions in train_masks:
            input_grid = self.create_input(
                taskvars, {"target_positions": target_positions}
            )
            train.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )
        test_input = self.create_input(
            taskvars, {"target_positions": test_mask}
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
        divider = taskvars["divider_color"]
        marker = taskvars["marker_color"]
        panel_size = taskvars["panel_size"]
        panel_rows = taskvars["panel_rows"]
        panel_cols = taskvars["panel_cols"]
        target_count = taskvars["target_count"]
        target_positions = {
            tuple(int(value) for value in position)
            for position in gridvars["target_positions"]
        }
        height = panel_rows * panel_size + panel_rows - 1
        width = panel_cols * panel_size + panel_cols - 1
        grid = np.full((height, width), background, dtype=int)

        for panel_row in range(1, panel_rows):
            divider_row = panel_row * panel_size + panel_row - 1
            grid[divider_row, :] = divider
        for panel_col in range(1, panel_cols):
            divider_col = panel_col * panel_size + panel_col - 1
            grid[:, divider_col] = divider

        marker_cells = set()
        maximum_count = panel_size * panel_size
        for panel_row in range(panel_rows):
            for panel_col in range(panel_cols):
                if (panel_row, panel_col) in target_positions:
                    count = target_count
                else:
                    try:
                        count = retry(
                            lambda: gridvars.get(
                                f"panel_{panel_row}_{panel_col}_count",
                                random.randint(
                                    1,
                                    min(maximum_count, target_count + 2),
                                ),
                            ),
                            lambda value: value != target_count,
                            max_attempts=20,
                        )
                    except ValueError:
                        count = min(maximum_count, target_count + 1)
                local_cells = [
                    tuple(int(value) for value in cell)
                    for cell in gridvars.get(
                        f"panel_{panel_row}_{panel_col}_cells",
                        random.sample(
                            [
                                (row, col)
                                for row in range(panel_size)
                                for col in range(panel_size)
                            ],
                            count,
                        ),
                    )
                ]
                row_offset = panel_row * (panel_size + 1)
                col_offset = panel_col * (panel_size + 1)
                marker_cells.update(
                    (row_offset + row, col_offset + col, marker)
                    for row, col in local_cells
                )
        GridObject(marker_cells).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        marker = taskvars["marker_color"]
        output_color = taskvars["output_color"]
        panel_size = taskvars["panel_size"]
        panel_rows = taskvars["panel_rows"]
        panel_cols = taskvars["panel_cols"]
        target_count = taskvars["target_count"]
        panels = get_objects_from_raster(
            grid,
            panel_size,
            panel_size,
            has_delimiters=True,
            initial_rows=panel_size,
            initial_cols=panel_size,
        )
        output = np.full((panel_rows, panel_cols), background, dtype=int)
        for panel_row in range(min(panel_rows, len(panels))):
            for panel_col in range(min(panel_cols, len(panels[panel_row]))):
                count = sum(
                    1
                    for _, _, color in panels[panel_row][panel_col].cells
                    if color == marker
                )
                if count == target_count:
                    output[panel_row, panel_col] = output_color
        return output
