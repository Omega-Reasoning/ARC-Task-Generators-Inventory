from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Task8e1813beGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains from {vars['minimum_line_count']} through {vars['maximum_line_count']} one-cell-wide parallel colored lines on {color('background_color')}.",
            "2. Every line has a distinct nonempty color, and all lines in one input are either horizontal or vertical.",
            "3. A solid {color('mask_color')} square has side length equal to the number of lines and is surrounded by a {vars['mask_moat_width']}-cell {color('background_color')} moat.",
            "4. The square and its moat interrupt some lines, but each interrupted line retains a visible fragment that preserves its fixed row or column.",
            "5. Line count, orientation, spacing, ordered colors, mask position, and input dimensions vary across examples.",
        ]
        transformation_reasoning_chain = [
            "1. Ignore {color('background_color')} and the {color('mask_color')} square, and group the remaining cells by their distinct line colors.",
            "2. Determine whether the colored groups span rows or columns to identify horizontal or vertical orientation.",
            "3. For horizontal lines, sort their colors by fixed row from top to bottom; for vertical lines, sort by fixed column from left to right.",
            "4. Create a square whose side equals the number of ordered line colors.",
            "5. Fill each output row with its corresponding horizontal-line color, or each output column with its corresponding vertical-line color.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            "background_color": 0,
            "mask_color": random.randint(1, 9),
            "mask_moat_width": 1,
            "minimum_line_count": 3,
            "maximum_line_count": 7,
        }
        train_specs = [
            {"line_count": 3, "orientation": "horizontal"},
            {"line_count": 4, "orientation": "vertical"},
            {"line_count": 5, "orientation": "horizontal"},
            {"line_count": 6, "orientation": "vertical"},
        ]
        random.shuffle(train_specs)

        train = []
        for gridvars in train_specs:
            input_grid = self.create_input(taskvars, gridvars)
            train.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )

        test_input = self.create_input(
            taskvars,
            {
                "line_count": 7,
                "orientation": random.choice(["horizontal", "vertical"]),
            },
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

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        background = taskvars["background_color"]
        mask_color = taskvars["mask_color"]
        moat = taskvars["mask_moat_width"]
        line_count = gridvars["line_count"]
        orientation = gridvars["orientation"]
        spacing = gridvars.get("spacing", random.choice([2, 3]))
        line_start = gridvars.get("line_start", 1)
        line_positions = [
            line_start + index * spacing
            for index in range(line_count)
        ]
        long_dimension = max(
            line_positions[-1] + 2,
            line_count + 2 * moat + 2,
        )
        cross_dimension = max(
            line_count * 2 + 7,
            line_count + 2 * moat + 5,
        )
        if orientation == "horizontal":
            rows, columns = long_dimension, cross_dimension
        else:
            rows, columns = cross_dimension, long_dimension
        rows = gridvars.get("rows", rows)
        columns = gridvars.get("columns", columns)

        available_colors = [
            color
            for color in range(1, 10)
            if color != mask_color
        ]
        line_colors = gridvars.get(
            "line_colors",
            random.sample(available_colors, line_count),
        )
        grid = np.full((rows, columns), background, dtype=int)
        line_cells = set()
        if orientation == "horizontal":
            for position, color in zip(line_positions, line_colors):
                line_cells.update(
                    (position, column, color)
                    for column in range(columns)
                )
        else:
            for position, color in zip(line_positions, line_colors):
                line_cells.update(
                    (row, position, color)
                    for row in range(rows)
                )
        GridObject(line_cells).paste(
            grid,
            overwrite=False,
            background=background,
        )

        def sample_mask_position():
            return (
                random.randint(moat, rows - line_count - moat),
                random.randint(moat, columns - line_count - moat),
            )

        def mask_position_is_valid(position):
            top, left = position
            if orientation == "horizontal":
                interrupted = sum(
                    top - moat
                    <= line_position
                    < top + line_count + moat
                    for line_position in line_positions
                )
            else:
                interrupted = sum(
                    left - moat
                    <= line_position
                    < left + line_count + moat
                    for line_position in line_positions
                )
            return 1 <= interrupted < line_count

        def sample_valid_mask_position():
            try:
                return retry(
                    sample_mask_position,
                    mask_position_is_valid,
                    max_attempts=120,
                )
            except ValueError:
                return (
                    max(moat, (rows - line_count) // 2),
                    max(moat, (columns - line_count) // 2),
                )

        mask_top, mask_left = gridvars.get(
            "mask_position",
            sample_valid_mask_position(),
        )

        grid[
            mask_top - moat : mask_top + line_count + moat,
            mask_left - moat : mask_left + line_count + moat,
        ] = background
        mask = np.full(
            (line_count, line_count),
            mask_color,
            dtype=int,
        )
        GridObject.from_array(
            mask,
            offset=(mask_top, mask_left),
        ).paste(grid, overwrite=False, background=background)
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background = taskvars["background_color"]
        mask_color = taskvars["mask_color"]
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        )
        cells_by_color = {}
        for object_ in objects:
            for row, column, color in object_.cells:
                if color != mask_color:
                    cells_by_color.setdefault(color, set()).add((row, column))
        if not cells_by_color:
            return np.full((1, 1), background, dtype=int)

        horizontal_score = 0
        vertical_score = 0
        for cells in cells_by_color.values():
            row_span = max(row for row, _ in cells) - min(row for row, _ in cells)
            column_span = (
                max(column for _, column in cells)
                - min(column for _, column in cells)
            )
            if column_span > row_span:
                horizontal_score += 1
            else:
                vertical_score += 1
        horizontal = horizontal_score > vertical_score
        if horizontal:
            ordered_colors = sorted(
                cells_by_color,
                key=lambda color: min(
                    row for row, _ in cells_by_color[color]
                ),
            )
        else:
            ordered_colors = sorted(
                cells_by_color,
                key=lambda color: min(
                    column for _, column in cells_by_color[color]
                ),
            )

        line_count = len(ordered_colors)
        output = np.full(
            (line_count, line_count),
            background,
            dtype=int,
        )
        stripe_cells = set()
        for index, color in enumerate(ordered_colors):
            if horizontal:
                stripe_cells.update(
                    (index, column, color)
                    for column in range(line_count)
                )
            else:
                stripe_cells.update(
                    (row, index, color)
                    for row in range(line_count)
                )
        GridObject(stripe_cells).paste(
            output,
            overwrite=False,
            background=background,
        )
        return output
