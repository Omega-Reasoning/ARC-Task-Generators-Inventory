from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Task91714a58Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['rows']}x{vars['cols']} grid on {color('background_color')} with sparse cells of many colors.",
            "2. Most foreground cells form small singleton or irregular noise components scattered across the grid.",
            "3. Exactly one color contains a completely filled axis-aligned rectangle larger than every accidental filled rectangle.",
            "4. One or more same-colored noise cells may protrude from an edge of the target and must not become part of the output rectangle.",
            "5. Target dimensions, orientation, position, colors, noise density, and protrusions vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Use the non-{color('background_color')} objects to identify every candidate foreground color.",
            "2. For each color, search successive row bands for contiguous column runs filled throughout the band.",
            "3. Select the maximum-area filled rectangle, trimming attached protrusions and discarding smaller accidental runs.",
            "4. Return a {vars['rows']}x{vars['cols']} {color('background_color')} grid containing only that rectangle at its original position.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            "rows": random.randint(14, 20),
            "cols": random.randint(14, 20),
            "background_color": 0,
        }
        train_specs = [
            {
                "target_height": 2,
                "target_width": 5,
                "noise_density": 0.14,
                "add_protrusion": False,
            },
            {
                "target_height": 5,
                "target_width": 2,
                "noise_density": 0.17,
                "add_protrusion": True,
            },
            {
                "target_height": 3,
                "target_width": 3,
                "noise_density": 0.20,
                "add_protrusion": True,
            },
            {
                "target_height": 4,
                "target_width": 4,
                "noise_density": 0.22,
                "add_protrusion": False,
            },
        ]
        test_spec = {
            "target_height": 3,
            "target_width": 6,
            "noise_density": 0.19,
            "add_protrusion": True,
        }

        def make_pair(gridvars: dict) -> GridPair:
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [make_pair(spec) for spec in train_specs]
        return taskvars, {"train": train, "test": [make_pair(test_spec)]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        background_color = taskvars["background_color"]
        target_height = gridvars["target_height"]
        target_width = gridvars["target_width"]
        noise_density = gridvars["noise_density"]
        add_protrusion = gridvars["add_protrusion"]
        target_color = gridvars.get("target_color", random.randint(1, 9))

        def sample_candidate() -> tuple[np.ndarray, tuple[int, int]]:
            row_top = gridvars.get(
                "row_top",
                random.randint(1, rows - target_height - 1),
            )
            col_left = gridvars.get(
                "col_left",
                random.randint(1, cols - target_width - 1),
            )
            grid = np.full((rows, cols), background_color, dtype=int)
            grid = np.asarray(
                gridvars.get(
                    "noise_grid",
                    random_cell_coloring(
                        grid,
                        list(range(1, 10)),
                        density=noise_density,
                        background=background_color,
                    ),
                ),
                dtype=int,
            )
            if grid.shape != (rows, cols):
                raise ValueError("noise_grid must match the task canvas")
            grid = np.array(grid, dtype=int, copy=True)
            grid[
                row_top : row_top + target_height,
                col_left : col_left + target_width,
            ] = target_color
            if add_protrusion:
                top_offset = gridvars.get(
                    "top_protrusion_offset",
                    random.randrange(target_width),
                )
                bottom_offset = gridvars.get(
                    "bottom_protrusion_offset",
                    random.randrange(target_width),
                )
                left_offset = gridvars.get(
                    "left_protrusion_offset",
                    random.randrange(target_height),
                )
                right_offset = gridvars.get(
                    "right_protrusion_offset",
                    random.randrange(target_height),
                )
                protrusions = [
                    (row_top - 1, col_left + top_offset),
                    (
                        row_top + target_height,
                        col_left + bottom_offset,
                    ),
                    (row_top + left_offset, col_left - 1),
                    (
                        row_top + right_offset,
                        col_left + target_width,
                    ),
                ]
                protrusion_row, protrusion_col = gridvars.get(
                    "protrusion",
                    random.choice(protrusions),
                )
                grid[protrusion_row, protrusion_col] = target_color
            return grid, (row_top, col_left)

        def intended_rectangle_wins(
            candidate: tuple[np.ndarray, tuple[int, int]],
        ) -> bool:
            grid, (row_top, col_left) = candidate
            expected = np.full((rows, cols), background_color, dtype=int)
            expected[
                row_top : row_top + target_height,
                col_left : col_left + target_width,
            ] = target_color
            return bool(
                np.array_equal(
                    self.transform_input(grid, taskvars),
                    expected,
                )
            )

        try:
            grid, _ = retry(
                sample_candidate,
                intended_rectangle_wins,
                max_attempts=50,
            )
            return grid
        except ValueError:
            row_top = (rows - target_height) // 2
            col_left = (cols - target_width) // 2
            grid = np.full((rows, cols), background_color, dtype=int)
            grid[
                row_top : row_top + target_height,
                col_left : col_left + target_width,
            ] = target_color
            if add_protrusion:
                grid[row_top - 1, col_left] = target_color
            noise_colors = [
                color for color in range(1, 10) if color != target_color
            ]
            fallback_positions = [
                (0, 0),
                (0, cols - 1),
                (rows - 1, 0),
                (rows - 1, cols - 1),
                (1, cols // 2),
            ]
            for index, (row, col) in enumerate(fallback_positions):
                grid[row, col] = noise_colors[index % len(noise_colors)]
            return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        background_color = taskvars["background_color"]
        foreground_objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background_color,
            monochromatic=True,
        )
        candidate_colors = set()
        for obj in foreground_objects:
            candidate_colors.update(int(color) for color in obj.colors)

        best_area = 0
        best_rectangle = None
        for color in candidate_colors:
            color_mask = grid == color
            for row_top in range(rows):
                shared_columns = np.ones(cols, dtype=bool)
                for row_bottom in range(row_top, rows):
                    shared_columns = np.logical_and(
                        shared_columns,
                        color_mask[row_bottom, :],
                    )
                    run_start = None
                    for col in range(cols + 1):
                        filled = col < cols and bool(shared_columns[col])
                        if filled and run_start is None:
                            run_start = col
                        elif not filled and run_start is not None:
                            area = (row_bottom - row_top + 1) * (col - run_start)
                            if area > best_area:
                                best_area = area
                                best_rectangle = (
                                    row_top,
                                    row_bottom + 1,
                                    run_start,
                                    col,
                                    color,
                                )
                            run_start = None

        row_top, row_bottom, col_left, col_right, color = best_rectangle
        output = np.full((rows, cols), background_color, dtype=int)
        rectangle = np.full(
            (row_bottom - row_top, col_right - col_left),
            color,
            dtype=int,
        )
        GridObject.from_array(
            rectangle,
            offset=(row_top, col_left),
        ).paste(output, overwrite=True, background=background_color)
        return output
