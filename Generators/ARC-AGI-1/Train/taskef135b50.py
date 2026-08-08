from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects


class Taskef135b50Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {color('background_color')} grid containing multiple separated solid rectangular foreground objects.",
            "2. The foreground color, dimensions, rectangle count, sizes, and placements vary by example.",
            "3. Some rectangle pairs overlap in row range while other rectangles occupy rows alone.",
            "4. Horizontally aligned foreground runs may be separated by short or long background gaps.",
            "5. The output bridge color is always {color('bridge_color')}.",
        ]
        transformation_reasoning_chain = [
            "1. Use connected-object perception to identify the separated foreground rectangles on {color('background_color')}.",
            "2. For each row, locate the leftmost and rightmost foreground cells.",
            "3. On non-border rows, recolor every {color('background_color')} gap cell between those extremes with {color('bridge_color')}.",
            "4. Preserve original foreground cells, all background outside the horizontal span, and the complete outer grid border.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {
            "background_color": 0,
            "bridge_color": 9,
            "preserve_outer_border": True,
        }
        train_gridvars = [
            {
                "rows": 10,
                "cols": 10,
                "rectangles": [[2, 0, 3, 3], [3, 7, 5, 2], [6, 3, 4, 2]],
            },
            {
                "rows": 12,
                "cols": 13,
                "rectangles": [
                    [1, 1, 5, 2],
                    [3, 5, 3, 3],
                    [2, 10, 6, 2],
                    [8, 4, 3, 4],
                ],
            },
            {
                "rows": 14,
                "cols": 12,
                "rectangles": [[2, 1, 8, 3], [5, 8, 7, 2]],
            },
            {
                "rows": 15,
                "cols": 16,
                "rectangles": [
                    [1, 0, 4, 3],
                    [2, 6, 6, 2],
                    [1, 12, 3, 3],
                    [9, 2, 4, 4],
                    [8, 10, 5, 3],
                ],
            },
        ]
        test_gridvars = {
            "rows": 17,
            "cols": 18,
            "rectangles": [
                [1, 1, 5, 3],
                [3, 7, 4, 2],
                [2, 14, 6, 3],
                [9, 0, 5, 2],
                [8, 6, 7, 3],
                [11, 14, 4, 2],
            ],
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
        background = taskvars["background_color"]
        bridge = taskvars["bridge_color"]
        rows, cols = gridvars["rows"], gridvars["cols"]
        rectangles = gridvars["rectangles"]

        def build_scene():
            foreground = gridvars.get(
                "foreground_color",
                random.choice(
                    [
                        color
                        for color in range(1, 10)
                        if color not in (background, bridge)
                    ]
                ),
            )
            if (
                isinstance(foreground, (bool, np.bool_))
                or not isinstance(foreground, (int, np.integer))
                or int(foreground) not in {
                    color
                    for color in range(1, 10)
                    if color not in (background, bridge)
                }
            ):
                raise ValueError(
                    "foreground_color must be an integer in the original palette"
                )
            foreground = int(foreground)
            placed_rectangles = gridvars.get(
                "placed_rectangles",
                [
                    [
                        min(
                            rows - height,
                            max(0, base_top + random.randint(-1, 1)),
                        ),
                        min(
                            cols - width,
                            max(0, base_left + random.randint(-1, 1)),
                        ),
                        height,
                        width,
                    ]
                    for base_top, base_left, height, width in rectangles
                ],
            )
            if len(placed_rectangles) != len(rectangles):
                raise ValueError("placed rectangle count must match rectangle count")
            grid = np.full((rows, cols), background, dtype=int)
            for (base_top, base_left, base_height, base_width), (
                top,
                left,
                height,
                width,
            ) in zip(rectangles, placed_rectangles):
                if (height, width) != (base_height, base_width):
                    raise ValueError("placed rectangle size must match its base rectangle")
                valid_tops = {
                    min(rows - height, max(0, base_top + delta))
                    for delta in (-1, 0, 1)
                }
                valid_lefts = {
                    min(cols - width, max(0, base_left + delta))
                    for delta in (-1, 0, 1)
                }
                if top not in valid_tops or left not in valid_lefts:
                    raise ValueError("placed rectangle must remain inside its jitter envelope")
                grid[top : top + height, left : left + width] = foreground
            return grid

        def valid_scene(grid):
            objects = find_connected_objects(
                grid,
                diagonal_connectivity=False,
                background=background,
                monochromatic=True,
            )
            if len(objects) != len(rectangles):
                return False
            if not all(len(obj) == obj.height * obj.width for obj in objects):
                return False
            for row in range(rows):
                foreground_cols = np.where(grid[row, :] != background)[0]
                if len(foreground_cols) >= 2:
                    left, right = int(foreground_cols[0]), int(foreground_cols[-1])
                    if np.any(grid[row, left : right + 1] == background):
                        return True
            return False

        return retry(build_scene, valid_scene, max_attempts=100)

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        bridge = taskvars["bridge_color"]
        preserve_border = taskvars["preserve_outer_border"]
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        )
        foreground_by_row = {}
        for obj in objects:
            for row, col, _ in obj:
                foreground_by_row.setdefault(row, []).append(col)
        output = grid.copy()
        for row, columns in foreground_by_row.items():
            if preserve_border and row in (0, grid.shape[0] - 1):
                continue
            left, right = min(columns), max(columns)
            for col in range(left, right + 1):
                if output[row, col] == background:
                    output[row, col] = bridge
        return output
