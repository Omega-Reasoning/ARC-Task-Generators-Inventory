from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random


class Task6cf79266Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['rows']}-row by {vars['cols']}-column grid containing one dense non-background color and scattered {color('background_color')} cells.",
            "2. The dense field color can differ between examples, while {color('background_color')} remains the background.",
            "3. Hidden among the scattered background noise are one or more completely empty {vars['square_size']} by {vars['square_size']} square windows.",
            "4. The number and positions of those complete empty squares vary between examples, and their locations need not align with the grid border.",
            "5. All examples share the same dimensions, target square size, and {color('marker_color')} replacement role.",
        ]
        transformation_reasoning_chain = [
            "1. Copy the input and inspect every {vars['square_size']} by {vars['square_size']} window from top to bottom and left to right.",
            "2. When the current window consists entirely of {color('background_color')} cells, recolor every cell in that square with {color('marker_color')}.",
            "3. Continue the row-major scan on the updated grid, so a later window overlapping an already marked square is not selected again.",
            "4. Preserve the dense field and every scattered background cell that is not part of a selected complete square.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        marker_color = random.choice(range(1, 10))
        taskvars = {
            "rows": random.randint(18, 24),
            "cols": random.randint(18, 24),
            "square_size": 3,
            "background_color": 0,
            "marker_color": marker_color,
        }
        field_colors = random.sample(
            [color for color in range(1, 10) if color != marker_color],
            5,
        )
        train_specs = [
            {"square_count": 1, "field_color": field_colors[0]},
            {"square_count": 2, "field_color": field_colors[1]},
            {"square_count": 1, "field_color": field_colors[2]},
            {"square_count": 2, "field_color": field_colors[3]},
        ]
        test_spec = {"square_count": 3, "field_color": field_colors[4]}

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
        square_size = taskvars["square_size"]
        background_color = taskvars["background_color"]
        marker_color = taskvars["marker_color"]
        field_color = gridvars["field_color"]
        square_count = gridvars["square_count"]

        fallback_positions = [
            (2, 2),
            (2, cols - square_size - 3),
            (rows - square_size - 3, 2),
        ][:square_count]

        def sample_positions() -> list[tuple[int, int]]:
            candidates = [
                (row, col)
                for row in range(1, rows - square_size)
                for col in range(1, cols - square_size)
            ]
            return [
                tuple(int(value) for value in position)
                for position in gridvars.get(
                    "positions",
                    random.sample(candidates, square_count),
                )
            ]

        def well_separated(positions: list[tuple[int, int]]) -> bool:
            return all(
                abs(row_a - row_b) >= square_size + 2
                or abs(col_a - col_b) >= square_size + 2
                for index, (row_a, col_a) in enumerate(positions)
                for row_b, col_b in positions[index + 1 :]
            )

        try:
            positions = retry(
                sample_positions,
                well_separated,
                max_attempts=40,
            )
        except ValueError:
            positions = fallback_positions
        positions = sorted(positions)

        def detected_positions(candidate: np.ndarray) -> list[tuple[int, int]]:
            working = candidate.copy()
            detected = []
            for row in range(rows - square_size + 1):
                for col in range(cols - square_size + 1):
                    if np.all(
                        working[
                            row : row + square_size,
                            col : col + square_size,
                        ]
                        == background_color
                    ):
                        detected.append((row, col))
                        working[
                            row : row + square_size,
                            col : col + square_size,
                        ] = marker_color
            return detected

        def carve_squares(candidate: np.ndarray) -> np.ndarray:
            for row, col in positions:
                candidate[
                    row : row + square_size,
                    col : col + square_size,
                ] = background_color
            return candidate

        def sample_grid() -> np.ndarray:
            candidate = np.full((rows, cols), background_color, dtype=int)
            random_cell_coloring(
                candidate,
                field_color,
                density=gridvars.get(
                    "density",
                    random.uniform(0.56, 0.70),
                ),
                background=background_color,
            )
            noise_cells = [
                tuple(int(value) for value in cell)
                for cell in gridvars.get(
                    "noise_cells",
                    np.argwhere(candidate == background_color).tolist(),
                )
            ]
            candidate.fill(field_color)
            for row, col in noise_cells:
                candidate[row, col] = background_color
            return carve_squares(candidate)

        try:
            return retry(
                sample_grid,
                lambda candidate: detected_positions(candidate) == positions,
                max_attempts=60,
            )
        except ValueError:
            fallback = np.full((rows, cols), field_color, dtype=int)
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    protected = any(
                        square_row - 1 <= row <= square_row + square_size
                        and square_col - 1 <= col <= square_col + square_size
                        for square_row, square_col in positions
                    )
                    if not protected and (3 * row + 5 * col) % 11 == 0:
                        fallback[row, col] = background_color
            return carve_squares(fallback)

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        square_size = taskvars['square_size']
        background_color = taskvars['background_color']
        marker_color = taskvars['marker_color']
        output = grid.copy()
        for row in range(output.shape[0] - square_size + 1):
            for col in range(output.shape[1] - square_size + 1):
                window = output[
                    row : row + square_size,
                    col : col + square_size,
                ]
                if np.all(window == background_color):
                    output[
                        row : row + square_size,
                        col : col + square_size,
                    ] = marker_color
        return output
