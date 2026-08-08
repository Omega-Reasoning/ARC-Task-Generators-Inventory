from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import GridObject


class Task1b60fb0cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each {vars['grid_size']} by {vars['grid_size']} input contains one {color('source_color')} figure on a {color('background_color')} background.",
            "2. The figure is formed by overlaying three consecutive quarter-turn copies of one asymmetric motif.",
            "3. The shared rotation center may be on a cell or between cells and must be inferred from the visible union.",
            "4. One fourth-copy orientation is absent, although different rotated copies may overlap near the center.",
        ]
        transformation_reasoning_chain = [
            "1. Test integer and half-integer centers under the {vars['rotation_order']}-fold quarter-turn geometry.",
            "2. Find the unique center where rotating the {color('source_color')} figure by one, two, or three quarter turns exposes one identical nonempty missing set.",
            "3. Paint that missing fourth-copy set {color('completion_color')}.",
            "4. Preserve every original {color('source_color')} cell and the {color('background_color')} canvas elsewhere.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        source_color, completion_color = random.sample(range(1, 10), 2)
        taskvars = {
            "grid_size": random.randint(13, 17),
            "source_color": source_color,
            "completion_color": completion_color,
            "background_color": 0,
            "rotation_order": 4,
        }

        def make_pair(height: int, width: int, half_center: bool) -> GridPair:
            input_grid = self.create_input(
                taskvars,
                {
                    "motif_height": height,
                    "motif_width": width,
                    "half_center": half_center,
                },
            )
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [
            make_pair(3, 3, False),
            make_pair(3, 4, True),
            make_pair(4, 3, False),
            make_pair(4, 4, True),
        ]
        return taskvars, {"train": train, "test": [make_pair(5, 4, False)]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        size = taskvars["grid_size"]
        source_color = taskvars["source_color"]
        background = taskvars["background_color"]
        height = gridvars["motif_height"]
        width = gridvars["motif_width"]
        center_row_twice = size - 1
        center_col_twice = size - 1
        desired_parity = 1 if gridvars["half_center"] else 0
        if center_row_twice % 2 != desired_parity:
            center_row_twice += 1
            center_col_twice += 1

        def rotate_cells(
            cells: set[tuple[int, int]], turns: int
        ) -> set[tuple[int, int]]:
            rotated = set()
            for row, col in cells:
                delta_row_twice = 2 * row - center_row_twice
                delta_col_twice = 2 * col - center_col_twice
                for _ in range(turns):
                    delta_row_twice, delta_col_twice = (
                        -delta_col_twice,
                        delta_row_twice,
                    )
                rotated.add(
                    (
                        (center_row_twice + delta_row_twice) // 2,
                        (center_col_twice + delta_col_twice) // 2,
                    )
                )
            return rotated

        def qualifying_centers(foreground: set[tuple[int, int]]) -> list:
            centers = []
            for candidate_row_twice in range(2 * size - 1):
                for candidate_col_twice in range(2 * size - 1):
                    missing_sets = []
                    valid = True
                    for turns in range(1, taskvars["rotation_order"]):
                        rotated = set()
                        for row, col in foreground:
                            delta_row_twice = 2 * row - candidate_row_twice
                            delta_col_twice = 2 * col - candidate_col_twice
                            for _ in range(turns):
                                delta_row_twice, delta_col_twice = (
                                    -delta_col_twice,
                                    delta_row_twice,
                                )
                            target_row_twice = candidate_row_twice + delta_row_twice
                            target_col_twice = candidate_col_twice + delta_col_twice
                            if target_row_twice % 2 or target_col_twice % 2:
                                valid = False
                                break
                            target = (
                                target_row_twice // 2,
                                target_col_twice // 2,
                            )
                            if not (
                                0 <= target[0] < size and 0 <= target[1] < size
                            ):
                                valid = False
                                break
                            rotated.add(target)
                        if not valid:
                            break
                        missing_sets.append(rotated - foreground)
                    if (
                        valid
                        and missing_sets
                        and len(missing_sets[0]) >= 4
                        and all(
                            missing == missing_sets[0]
                            for missing in missing_sets[1:]
                        )
                    ):
                        centers.append(
                            (candidate_row_twice, candidate_col_twice)
                        )
            return centers

        center_row = center_row_twice / 2
        center_col = center_col_twice / 2
        bottom = int(np.floor(center_row))
        top = bottom - height + 1
        left = int(round(center_col - (width - 1) / 2))
        omitted_turn = gridvars.get("omitted_turn", 3)

        def sample_grid() -> np.ndarray:
            motif = np.asarray(
                gridvars.get(
                    "motif",
                    create_object(
                        height,
                        width,
                        source_color,
                        contiguity=Contiguity.EIGHT,
                        background=background,
                    ),
                ),
                dtype=int,
            )
            base = {
                (top + int(row), left + int(col))
                for row, col in np.argwhere(motif == source_color)
            }
            rotations = [rotate_cells(base, turns) for turns in range(4)]
            foreground = set().union(*(
                rotation
                for turns, rotation in enumerate(rotations)
                if turns != omitted_turn
            ))
            grid = np.full((size, size), background, dtype=int)
            GridObject(
                {(row, col, source_color) for row, col in foreground}
            ).paste(grid, background=background)
            return grid

        def valid_grid(grid: np.ndarray) -> bool:
            foreground = {
                (int(row), int(col))
                for row, col in np.argwhere(grid == source_color)
            }
            centers = qualifying_centers(foreground)
            return centers == [(center_row_twice, center_col_twice)]

        try:
            return retry(sample_grid, valid_grid, max_attempts=120)
        except ValueError:
            motif = np.full((height, width), background, dtype=int)
            motif[0, :] = source_color
            motif[:, width // 2] = source_color
            motif[-1, max(0, width // 2 - 1)] = source_color
            base = {
                (top + int(row), left + int(col))
                for row, col in np.argwhere(motif == source_color)
            }
            foreground = set()
            for turns in range(4):
                if turns != omitted_turn:
                    foreground.update(rotate_cells(base, turns))
            grid = np.full((size, size), background, dtype=int)
            GridObject(
                {(row, col, source_color) for row, col in foreground}
            ).paste(grid, background=background)
            return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        source_color = taskvars["source_color"]
        completion_color = taskvars["completion_color"]
        rotation_order = taskvars["rotation_order"]
        foreground = {
            (int(row), int(col))
            for row, col in np.argwhere(grid == source_color)
        }
        row_count, col_count = grid.shape
        valid_centers = []
        for center_row_twice in range(2 * row_count - 1):
            for center_col_twice in range(2 * col_count - 1):
                missing_sets = []
                center_valid = True
                for turns in range(1, rotation_order):
                    rotated = set()
                    for row, col in foreground:
                        delta_row_twice = 2 * row - center_row_twice
                        delta_col_twice = 2 * col - center_col_twice
                        for _ in range(turns):
                            delta_row_twice, delta_col_twice = (
                                -delta_col_twice,
                                delta_row_twice,
                            )
                        target_row_twice = center_row_twice + delta_row_twice
                        target_col_twice = center_col_twice + delta_col_twice
                        if target_row_twice % 2 or target_col_twice % 2:
                            center_valid = False
                            break
                        target_row = target_row_twice // 2
                        target_col = target_col_twice // 2
                        if not (
                            0 <= target_row < row_count
                            and 0 <= target_col < col_count
                        ):
                            center_valid = False
                            break
                        rotated.add((target_row, target_col))
                    if not center_valid:
                        break
                    missing_sets.append(rotated - foreground)
                if (
                    center_valid
                    and missing_sets
                    and missing_sets[0]
                    and all(missing == missing_sets[0] for missing in missing_sets[1:])
                ):
                    valid_centers.append(
                        (
                            len(missing_sets[0]),
                            center_row_twice,
                            center_col_twice,
                            missing_sets[0],
                        )
                    )

        if not valid_centers:
            return grid.copy()
        _, _, _, missing = min(valid_centers, key=lambda item: item[:3])
        output = grid.copy()
        for row, col in missing:
            output[row, col] = completion_color
        return output
