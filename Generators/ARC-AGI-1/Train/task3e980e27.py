from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task3e980e27Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains one or more multicolor templates and isolated key cells on a {color('background_color')} grid.",
            "2. Every template has a body color and exactly one key cell, which is either {color('mirror_key_color')} or {color('copy_key_color')}.",
            "3. Each isolated key matches the key color of exactly one template in the same input.",
            "4. Template cells are treated as {vars['object_connectivity']}-way connected, so diagonal body links belong to one object.",
            "5. Body colors, template shapes, key types, and singleton positions vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Associate every isolated key cell with the multicolor template containing the same key color.",
            "2. Keep a {color('copy_key_color')}-keyed template in its original orientation.",
            "3. Reflect a {color('mirror_key_color')}-keyed template horizontally about its key cell.",
            "4. Translate the selected template pose until its key coincides with the isolated key.",
            "5. Paste every template cell there while preserving all original templates and keys.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        mirror_key, copy_key = random.sample(range(1, 10), 2)
        taskvars = {
            "background_color": 0,
            "mirror_key_color": mirror_key,
            "copy_key_color": copy_key,
            "object_connectivity": 8,
        }
        schedules = [
            {"mirror_targets": [(13, 5)], "copy_targets": [(13, 13)]},
            {"mirror_targets": [(13, 13), (20, 5)], "copy_targets": [(20, 21)]},
            {"mirror_targets": [(20, 13)], "copy_targets": [(13, 5), (13, 21)]},
            {"mirror_targets": [(13, 5), (20, 21)], "copy_targets": [(13, 21), (20, 5)]},
        ]
        train = []
        for gridvars in schedules:
            input_grid = self.create_input(taskvars, gridvars)
            train.append({
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            })
        test_input = self.create_input(taskvars, {
            "mirror_targets": [(13, 5), (13, 21), (20, 13)],
            "copy_targets": [(13, 13), (20, 5)],
        })
        return taskvars, {
            "train": train,
            "test": [{
                "input": test_input,
                "output": self.transform_input(test_input, taskvars),
            }],
        }

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        background = taskvars["background_color"]
        mirror_key = taskvars["mirror_key_color"]
        copy_key = taskvars["copy_key_color"]
        body_colors = gridvars.get(
            "body_colors",
            random.sample(
                [
                    color for color in range(1, 10)
                    if color not in (mirror_key, copy_key)
                ],
                2,
            ),
        )

        def make_template(template_name, key_color, body_color):
            def sample_body():
                return np.asarray(
                    gridvars.get(
                        f"{template_name}_body_candidate",
                        create_object(
                            4,
                            4,
                            body_color,
                            contiguity=Contiguity.EIGHT,
                            background=background,
                        ),
                    ),
                    dtype=int,
                )

            def valid_body(body):
                occupied = body == body_color
                count = int(np.count_nonzero(occupied))
                return bool(
                    6 <= count <= 12
                    and np.all(np.any(occupied, axis=0))
                    and np.all(np.any(occupied, axis=1))
                    and not np.array_equal(occupied, np.fliplr(occupied))
                )

            try:
                body = retry(sample_body, valid_body, max_attempts=60)
            except ValueError:
                body = np.array([
                    [body_color, body_color, background, background],
                    [background, body_color, body_color, background],
                    [body_color, body_color, background, background],
                    [body_color, background, background, background],
                ], dtype=int)

            occupied_coords = {
                (int(row), int(col))
                for row, col in np.argwhere(body == body_color)
            }
            boundary = set()
            for row, col in occupied_coords:
                for delta_row, delta_col in (
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1),
                ):
                    candidate = (row + delta_row, col + delta_col)
                    if (
                        candidate not in occupied_coords
                        and (
                            candidate[0] < 0
                            or candidate[0] >= body.shape[0]
                            or candidate[1] < 0
                            or candidate[1] >= body.shape[1]
                        )
                    ):
                        boundary.add(candidate)
            key_row, key_col = gridvars.get(
                f"{template_name}_key_coord",
                random.choice(sorted(boundary)),
            )
            cells = {
                (row, col, body_color)
                for row, col in occupied_coords
            }
            cells.add((key_row, key_col, key_color))
            return {
                tuple(cell)
                for cell in gridvars.get(f"{template_name}_template", cells)
            }

        mirror_template = make_template("mirror", mirror_key, body_colors[0])
        copy_template = make_template("copy", copy_key, body_colors[1])
        rows = int(gridvars.get("rows", 26))
        cols = int(gridvars.get("cols", 26))
        grid = np.full((rows, cols), background, dtype=int)

        for cells, (origin_row, origin_col) in (
            (mirror_template, gridvars.get("mirror_origin", (2, 2))),
            (copy_template, gridvars.get("copy_origin", (2, 17))),
        ):
            if not cells:
                continue
            min_row = min(row for row, _, _ in cells)
            min_col = min(col for _, col, _ in cells)
            for row, col, color in cells:
                grid[
                    origin_row + row - min_row,
                    origin_col + col - min_col,
                ] = color

        for row, col in gridvars["mirror_targets"]:
            grid[row, col] = mirror_key
        for row, col in gridvars["copy_targets"]:
            grid[row, col] = copy_key
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background = taskvars["background_color"]
        mirror_key = taskvars["mirror_key_color"]
        copy_key = taskvars["copy_key_color"]
        diagonal = taskvars["object_connectivity"] == 8
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=diagonal,
            background=background,
            monochromatic=False,
        )
        templates = [obj for obj in objects if len(obj.colors) > 1]
        singleton_keys = [obj for obj in objects if obj.size == 1]
        output = grid.copy()
        for singleton in singleton_keys:
            target_row, target_col, key_color = next(iter(singleton.cells))
            if key_color not in (mirror_key, copy_key):
                continue
            matches = [
                template for template in templates
                if key_color in template.colors
            ]
            if len(matches) != 1:
                continue
            template = matches[0]
            key_cells = [
                (row, col)
                for row, col, color in template.cells
                if color == key_color
            ]
            if len(key_cells) != 1:
                continue
            source_row, source_col = key_cells[0]
            for row, col, color in template.cells:
                relative_row = row - source_row
                relative_col = col - source_col
                if key_color == mirror_key:
                    relative_col = -relative_col
                output[
                    target_row + relative_row,
                    target_col + relative_col,
                ] = color
        return output
