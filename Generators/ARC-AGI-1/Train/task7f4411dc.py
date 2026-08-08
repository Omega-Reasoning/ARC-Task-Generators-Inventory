from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Task7f4411dcGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input has between {vars['minimum_grid_dimension']} and {vars['maximum_grid_dimension']} rows and columns, with one nonempty color on {color('background_color')}.",
            "2. The colored cells include one or more solid axis-aligned rectangles whose height and width are each at least {vars['membership_block_size']}.",
            "3. Additional cells of the same color appear as isolated distractors or as one-cell protrusions touching a rectangle.",
            "4. No distractor or protrusion completes a new filled {vars['membership_block_size']}x{vars['membership_block_size']} block outside the intended rectangles.",
            "5. Grid dimensions, color, rectangle count and geometry, and noise placement vary across examples.",
        ]
        transformation_reasoning_chain = [
            "1. Examine every {vars['membership_block_size']}x{vars['membership_block_size']} window in the input.",
            "2. Mark all four cells of each window that is completely filled with the single non-{color('background_color')} color.",
            "3. Take the union of all marked cells, so overlapping windows preserve complete solid rectangles of any larger size.",
            "4. Place that retained union on a new {color('background_color')} grid of the input shape.",
            "5. Thus erase every colored cell that belongs to no filled {vars['membership_block_size']}x{vars['membership_block_size']} window.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            "minimum_grid_dimension": 9,
            "maximum_grid_dimension": 20,
            "membership_block_size": 2,
            "background_color": 0,
        }
        object_colors = random.sample(range(1, 10), 5)
        train_specs = [
            {
                "shape": (9, 11),
                "rectangle_count": 1,
                "isolated_count": 4,
                "protrusion_count": 0,
            },
            {
                "shape": (12, 13),
                "rectangle_count": 2,
                "isolated_count": 2,
                "protrusion_count": 3,
            },
            {
                "shape": (14, 15),
                "rectangle_count": 3,
                "isolated_count": 3,
                "protrusion_count": 4,
            },
            {
                "shape": (16, 17),
                "rectangle_count": 4,
                "isolated_count": 5,
                "protrusion_count": 5,
            },
        ]
        for spec, object_color in zip(train_specs, object_colors[:4]):
            spec["object_color"] = object_color
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
                "shape": (18, 19),
                "rectangle_count": 5,
                "isolated_count": 6,
                "protrusion_count": 6,
                "object_color": object_colors[4],
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
        rows, columns = gridvars["shape"]
        rectangle_count = gridvars["rectangle_count"]
        object_color = gridvars["object_color"]
        background = taskvars["background_color"]
        block_size = taskvars["membership_block_size"]

        def rectangles_too_close(first, second):
            first_top, first_left, first_height, first_width = first
            second_top, second_left, second_height, second_width = second
            return not (
                first_top + first_height + 1 <= second_top
                or second_top + second_height + 1 <= first_top
                or first_left + first_width + 1 <= second_left
                or second_left + second_width + 1 <= first_left
            )

        def sample_layout():
            return [
                (
                    random.randint(0, rows - height),
                    random.randint(0, columns - width),
                    height,
                    width,
                )
                for height, width in [
                    (
                        random.randint(block_size, 5),
                        random.randint(block_size, 6),
                    )
                    for _ in range(rectangle_count)
                ]
            ]

        def layout_is_valid(rectangles):
            if len(rectangles) != rectangle_count:
                return False
            for index, rectangle in enumerate(rectangles):
                top, left, height, width = rectangle
                if height < block_size or width < block_size:
                    return False
                if not (
                    0 <= top
                    and 0 <= left
                    and top + height <= rows
                    and left + width <= columns
                ):
                    return False
                if any(
                    rectangles_too_close(rectangle, previous)
                    for previous in rectangles[:index]
                ):
                    return False
            return True

        def sample_rectangles():
            try:
                return retry(
                    sample_layout,
                    layout_is_valid,
                    max_attempts=200,
                )
            except ValueError:
                fallback_positions = [
                    (1, 1),
                    (1, 6),
                    (1, 11),
                    (6, 1),
                    (6, 6),
                ]
                return [
                    (top, left, 2, 3)
                    for top, left in fallback_positions[:rectangle_count]
                ]

        rectangles = [
            tuple(int(value) for value in rectangle)
            for rectangle in gridvars.get("rectangles", sample_rectangles())
        ]
        if not layout_is_valid(rectangles):
            raise ValueError("rectangles do not form a valid separated layout")

        rectangle_grid = np.full((rows, columns), background, dtype=int)
        intended_cells = set()
        for top, left, height, width in rectangles:
            rectangle = np.full(
                (height, width),
                object_color,
                dtype=int,
            )
            GridObject.from_array(
                rectangle,
                offset=(top, left),
            ).paste(rectangle_grid, overwrite=False, background=background)
            intended_cells.update(
                (row, column)
                for row in range(top, top + height)
                for column in range(left, left + width)
            )

        def retained_cells(candidate):
            retained = set()
            for row in range(rows - block_size + 1):
                for column in range(columns - block_size + 1):
                    block = candidate[
                        row : row + block_size,
                        column : column + block_size,
                    ]
                    if np.all(block == object_color):
                        retained.update(
                            (block_row, block_column)
                            for block_row in range(row, row + block_size)
                            for block_column in range(
                                column,
                                column + block_size,
                            )
                        )
            return retained

        protrusion_candidates = [
            (row, column)
            for row in range(rows)
            for column in range(columns)
            if (
                rectangle_grid[row, column] == background
                and any(
                    (row + row_step, column + column_step) in intended_cells
                    for row_step, column_step in (
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                    )
                )
            )
        ]
        isolated_candidates = [
            (row, column)
            for row in range(rows)
            for column in range(columns)
            if (
                rectangle_grid[row, column] == background
                and all(
                    (row + row_step, column + column_step) not in intended_cells
                    for row_step in (-1, 0, 1)
                    for column_step in (-1, 0, 1)
                )
            )
        ]
        protrusion_count = min(
            gridvars["protrusion_count"],
            len(protrusion_candidates),
        )
        isolated_count = min(
            gridvars["isolated_count"],
            len(isolated_candidates),
        )

        def grid_with_noise(noise_cells):
            candidate = rectangle_grid.copy()
            GridObject(
                {
                    (row, column, object_color)
                    for row, column in noise_cells
                }
            ).paste(candidate, overwrite=False, background=background)
            return candidate

        def sample_noise():
            return set(
                random.sample(protrusion_candidates, protrusion_count)
                + random.sample(isolated_candidates, isolated_count)
            )

        def noise_is_valid(noise_cells):
            return (
                len(noise_cells) == protrusion_count + isolated_count
                and retained_cells(grid_with_noise(noise_cells)) == intended_cells
            )

        def sample_valid_noise():
            try:
                return retry(
                    sample_noise,
                    noise_is_valid,
                    max_attempts=160,
                )
            except ValueError:
                fallback_noise = set()
                ordered_candidates = (
                    protrusion_candidates[:protrusion_count]
                    + isolated_candidates[:isolated_count]
                )
                for coordinate in ordered_candidates:
                    candidate_noise = fallback_noise | {coordinate}
                    if (
                        retained_cells(grid_with_noise(candidate_noise))
                        == intended_cells
                    ):
                        fallback_noise = candidate_noise
                return fallback_noise

        noise_cells = {
            tuple(int(value) for value in coordinate)
            for coordinate in gridvars.get("noise_cells", sample_valid_noise())
        }
        if retained_cells(grid_with_noise(noise_cells)) != intended_cells:
            raise ValueError("noise cells alter the intended retained rectangles")
        return grid_with_noise(noise_cells)

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background = taskvars["background_color"]
        block_size = taskvars["membership_block_size"]
        output = np.full_like(grid, background)
        rows, columns = grid.shape

        objects = find_connected_objects(
            grid,
            diagonal_connectivity=True,
            background=background,
            monochromatic=True,
        ).sort_by_size(reverse=True)
        if len(objects) == 0:
            return output
        object_color = next(iter(objects[0].colors))

        retained_coordinates = set()
        for row in range(rows - block_size + 1):
            for column in range(columns - block_size + 1):
                block = grid[
                    row : row + block_size,
                    column : column + block_size,
                ]
                if np.all(block == object_color):
                    retained_coordinates.update(
                        (block_row, block_column)
                        for block_row in range(row, row + block_size)
                        for block_column in range(
                            column,
                            column + block_size,
                        )
                    )
        GridObject.from_grid(grid, retained_coordinates).paste(
            output,
            overwrite=False,
            background=background,
        )
        return output
