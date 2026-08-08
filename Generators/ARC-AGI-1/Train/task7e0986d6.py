from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Task7e0986d6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input has between {vars['minimum_grid_dimension']} and {vars['maximum_grid_dimension']} rows and columns and uses {color('background_color')} plus two nonempty colors.",
            "2. The more frequent nonempty color forms one or more solid axis-aligned rectangular fields, and some fields may touch edge-to-edge.",
            "3. The less frequent anomaly color replaces selected cells inside those fields and also appears as isolated distractors outside them.",
            "4. Every anomaly belonging to a field has at least {vars['support_neighbor_threshold']} orthogonally adjacent field-color neighbors in the visible input.",
            "5. Every outside distractor has fewer than {vars['support_neighbor_threshold']} orthogonally adjacent field-color neighbors; dimensions, colors, rectangles, and anomaly positions vary across examples.",
        ]
        transformation_reasoning_chain = [
            "1. Determine the more frequent nonempty field color and the less frequent anomaly color, then copy the input.",
            "2. For each anomaly cell, count its field-color neighbors directly above, below, left, and right.",
            "3. If the count is at least {vars['support_neighbor_threshold']}, recolor that anomaly with the field color to repair its rectangle.",
            "4. Otherwise recolor the anomaly {color('background_color')} to remove an outside distractor.",
            "5. Preserve all original field-color and {color('background_color')} cells.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            "minimum_grid_dimension": 12,
            "maximum_grid_dimension": 20,
            "background_color": 0,
            "support_neighbor_threshold": 2,
        }
        color_pairs = random.sample(
            [
                (field_color, anomaly_color)
                for field_color in range(1, 10)
                for anomaly_color in range(1, 10)
                if field_color != anomaly_color
            ],
            5,
        )
        train_specs = [
            {
                "shape": (12, 14),
                "rectangle_count": 1,
                "touching": False,
                "embedded_per_rectangle": 2,
                "outside_count": 4,
            },
            {
                "shape": (14, 13),
                "rectangle_count": 2,
                "touching": False,
                "embedded_per_rectangle": 1,
                "outside_count": 5,
            },
            {
                "shape": (15, 17),
                "rectangle_count": 3,
                "touching": True,
                "embedded_per_rectangle": 2,
                "outside_count": 6,
            },
            {
                "shape": (17, 16),
                "rectangle_count": 4,
                "touching": False,
                "embedded_per_rectangle": 1,
                "outside_count": 7,
            },
        ]
        for spec, colors in zip(train_specs, color_pairs[:4]):
            spec["field_color"], spec["anomaly_color"] = colors
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

        test_gridvars = {
            "shape": (18, 19),
            "rectangle_count": 5,
            "touching": False,
            "embedded_per_rectangle": 1,
            "outside_count": 8,
            "field_color": color_pairs[4][0],
            "anomaly_color": color_pairs[4][1],
        }
        test_input = self.create_input(taskvars, test_gridvars)
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
        rows, columns = gridvars["shape"]
        rectangle_count = gridvars["rectangle_count"]
        touching = gridvars["touching"]
        field_color = gridvars["field_color"]
        anomaly_color = gridvars["anomaly_color"]
        background = taskvars["background_color"]
        threshold = taskvars["support_neighbor_threshold"]

        def rectangles_overlap(first, second):
            first_top, first_left, first_height, first_width = first
            second_top, second_left, second_height, second_width = second
            return not (
                first_top + first_height <= second_top
                or second_top + second_height <= first_top
                or first_left + first_width <= second_left
                or second_left + second_width <= first_left
            )

        def sample_layout():
            rectangles = []
            if touching and rectangle_count >= 2:
                first_height = gridvars.get(
                    "touching_first_height", random.randint(3, 5)
                )
                first_width = gridvars.get(
                    "touching_first_width", random.randint(3, 5)
                )
                second_height = gridvars.get(
                    "touching_second_height", random.randint(3, 5)
                )
                second_width = gridvars.get(
                    "touching_second_width", random.randint(3, 5)
                )
                horizontal_touch = gridvars.get(
                    "horizontal_touch", random.choice([True, False])
                )
                if horizontal_touch:
                    top = gridvars.get(
                        "touching_top",
                        random.randint(
                            0,
                            rows - max(first_height, second_height),
                        ),
                    )
                    left = gridvars.get(
                        "touching_left",
                        random.randint(0, columns - first_width - second_width),
                    )
                    rectangles.extend(
                        [
                            (top, left, first_height, first_width),
                            (
                                top,
                                left + first_width,
                                second_height,
                                second_width,
                            ),
                        ]
                    )
                else:
                    top = gridvars.get(
                        "touching_top",
                        random.randint(0, rows - first_height - second_height),
                    )
                    left = gridvars.get(
                        "touching_left",
                        random.randint(
                            0,
                            columns - max(first_width, second_width),
                        ),
                    )
                    rectangles.extend(
                        [
                            (top, left, first_height, first_width),
                            (
                                top + first_height,
                                left,
                                second_height,
                                second_width,
                            ),
                        ]
                    )
            while len(rectangles) < rectangle_count:
                rectangle_index = len(rectangles)
                height = gridvars.get(
                    f"rectangle_{rectangle_index}_height",
                    random.randint(3, 5),
                )
                width = gridvars.get(
                    f"rectangle_{rectangle_index}_width",
                    random.randint(3, 6),
                )
                rectangles.append(
                    (
                        gridvars.get(
                            f"rectangle_{rectangle_index}_top",
                            random.randint(0, rows - height),
                        ),
                        gridvars.get(
                            f"rectangle_{rectangle_index}_left",
                            random.randint(0, columns - width),
                        ),
                        height,
                        width,
                    )
                )
            return [
                tuple(int(value) for value in rectangle)
                for rectangle in gridvars.get("rectangles", rectangles)
            ]

        def layout_is_valid(rectangles):
            if len(rectangles) != rectangle_count:
                return False
            for index, rectangle in enumerate(rectangles):
                top, left, height, width = rectangle
                if not (
                    0 <= top
                    and 0 <= left
                    and top + height <= rows
                    and left + width <= columns
                ):
                    return False
                for previous in rectangles[:index]:
                    if rectangles_overlap(rectangle, previous):
                        return False
            return True

        try:
            rectangles = retry(
                sample_layout,
                layout_is_valid,
                max_attempts=160,
            )
        except ValueError:
            rectangles = []
            fallback_positions = [
                (1, 1),
                (1, 7),
                (6, 1),
                (6, 7),
                (11, 1),
            ]
            if touching and rectangle_count >= 2:
                rectangles.extend(
                    [
                        (1, 1, 3, 4),
                        (1, 5, 3, 4),
                    ]
                )
                fallback_positions = [(6, 1), (6, 7), (11, 1)]
            for top, left in fallback_positions:
                if len(rectangles) >= rectangle_count:
                    break
                rectangles.append((top, left, 3, 4))

        full_grid = np.full((rows, columns), background, dtype=int)
        for top, left, height, width in rectangles:
            rectangle_array = np.full(
                (height, width),
                field_color,
                dtype=int,
            )
            GridObject.from_array(
                rectangle_array,
                offset=(top, left),
            ).paste(full_grid, overwrite=False, background=background)

        embedded_per_rectangle = gridvars["embedded_per_rectangle"]

        def sample_embedded_cells():
            selected = set()
            for rectangle_index, (top, left, height, width) in enumerate(
                rectangles
            ):
                rectangle_cells = [
                    (row, column)
                    for row in range(top, top + height)
                    for column in range(left, left + width)
                ]
                selected.update(
                    tuple(int(value) for value in cell)
                    for cell in gridvars.get(
                        f"rectangle_{rectangle_index}_embedded_cells",
                        random.sample(
                            rectangle_cells,
                            min(embedded_per_rectangle, len(rectangle_cells)),
                        ),
                    )
                )
            return {
                tuple(int(value) for value in cell)
                for cell in gridvars.get("embedded_cells", selected)
            }

        def support_count(grid, row, column):
            return sum(
                0 <= row + row_step < rows
                and 0 <= column + column_step < columns
                and grid[row + row_step, column + column_step] == field_color
                for row_step, column_step in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                )
            )

        def embedded_cells_are_valid(selected):
            expected_count = rectangle_count * embedded_per_rectangle
            if len(selected) != expected_count:
                return False
            candidate = full_grid.copy()
            for row, column in selected:
                candidate[row, column] = anomaly_color
            return all(
                support_count(candidate, row, column) >= threshold
                for row, column in selected
            )

        try:
            embedded_cells = retry(
                sample_embedded_cells,
                embedded_cells_are_valid,
                max_attempts=120,
            )
        except ValueError:
            embedded_cells = {
                (top + height // 2, left + width // 2)
                for top, left, height, width in rectangles
            }

        input_grid = full_grid.copy()
        GridObject(
            {
                (row, column, anomaly_color)
                for row, column in embedded_cells
            }
        ).paste(input_grid, overwrite=True, background=background)

        outside_candidates = [
            (row, column)
            for row in range(rows)
            for column in range(columns)
            if (
                input_grid[row, column] == background
                and support_count(input_grid, row, column) < threshold
            )
        ]
        outside_count = min(gridvars["outside_count"], len(outside_candidates))

        def sample_outside_cells():
            return {
                tuple(int(value) for value in cell)
                for cell in gridvars.get(
                    "outside_cells",
                    random.sample(outside_candidates, outside_count),
                )
            }

        def outside_cells_are_valid(selected):
            return (
                len(selected) == outside_count
                and all(
                    support_count(input_grid, row, column) < threshold
                    for row, column in selected
                )
            )

        try:
            outside_cells = retry(
                sample_outside_cells,
                outside_cells_are_valid,
                max_attempts=80,
            )
        except ValueError:
            outside_cells = set(outside_candidates[:outside_count])
        GridObject(
            {
                (row, column, anomaly_color)
                for row, column in outside_cells
            }
        ).paste(input_grid, overwrite=False, background=background)
        return input_grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background = taskvars["background_color"]
        threshold = taskvars["support_neighbor_threshold"]
        output = grid.copy()
        rows, columns = grid.shape

        objects = find_connected_objects(
            grid,
            diagonal_connectivity=True,
            background=background,
            monochromatic=True,
        )
        color_counts = {}
        for object_ in objects:
            for color in object_.colors:
                color_counts[color] = color_counts.get(color, 0) + len(object_)
        if len(color_counts) < 2:
            return output
        ordered_colors = sorted(
            color_counts,
            key=lambda color: color_counts[color],
            reverse=True,
        )
        field_color = ordered_colors[0]
        anomaly_color = ordered_colors[1]

        repair_cells = set()
        erase_coordinates = set()
        for row, column in map(tuple, np.argwhere(grid == anomaly_color)):
            support = sum(
                0 <= row + row_step < rows
                and 0 <= column + column_step < columns
                and grid[row + row_step, column + column_step] == field_color
                for row_step, column_step in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                )
            )
            if support >= threshold:
                repair_cells.add((row, column, field_color))
            else:
                erase_coordinates.add((row, column))

        GridObject(repair_cells).paste(
            output,
            overwrite=True,
            background=background,
        )
        GridObject.from_grid(grid, erase_coordinates).cut(
            output,
            background=background,
        )
        return output
