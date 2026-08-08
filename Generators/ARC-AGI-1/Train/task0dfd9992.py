from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import random_cell_coloring, retry


class Task0dfd9992Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Every input is a {vars['grid_size']} by {vars['grid_size']} grid formed by repeating a smaller square color tile in both axes.",
            "2. The fundamental tile period varies between examples and need not divide the full grid size.",
            "3. One or more rectangular or irregular regions have been erased to {color('hole_color')}.",
            "4. Outside those holes, repeated copies agree at every coordinate modulo the hidden period.",
            "5. Enough intact repetitions remain to determine every cell of the fundamental tile.",
        ]
        transformation_reasoning_chain = [
            "1. Find the smallest square period for which all visible cells at each modulo-period coordinate agree.",
            "2. Recover the color of every fundamental-tile coordinate from its intact repeated occurrences.",
            "3. Repeat that recovered tile over the full {vars['grid_size']} by {vars['grid_size']} field.",
            "4. Replace every {color('hole_color')} cell while preserving all already visible colors.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            "grid_size": random.randint(21, 25),
            "hole_color": 0,
        }

        def make_pair(period: int) -> GridPair:
            input_grid = self.create_input(taskvars, {"period": period})
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [make_pair(period) for period in (4, 5, 6, 7)]
        return taskvars, {"train": train, "test": [make_pair(9)]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        period = gridvars["period"]
        size = taskvars["grid_size"]
        palette = gridvars.get(
            "palette", random.sample(range(1, 10), random.randint(4, 7))
        )

        def smallest_period(candidate: np.ndarray) -> int:
            for trial in range(1, candidate.shape[0] + 1):
                mapping = {}
                valid = True
                for row in range(candidate.shape[0]):
                    for col in range(candidate.shape[1]):
                        value = int(candidate[row, col])
                        if value == taskvars["hole_color"]:
                            continue
                        key = (row % trial, col % trial)
                        if key in mapping and mapping[key] != value:
                            valid = False
                            break
                        mapping[key] = value
                    if not valid:
                        break
                if valid and len(mapping) == trial * trial:
                    return trial
            return candidate.shape[0]

        def sample_tile() -> np.ndarray:
            tile = np.zeros((period, period), dtype=int)
            return np.asarray(
                gridvars.get(
                    "tile",
                    random_cell_coloring(
                        tile,
                        palette,
                        density=1.0,
                        background=0,
                        overwrite=False,
                    ),
                ),
                dtype=int,
            )

        try:
            tile = retry(
                sample_tile,
                lambda candidate: smallest_period(candidate) == period,
                max_attempts=80,
            )
        except ValueError:
            tile = np.full((period, period), palette[0], dtype=int)
            tile[-1, -1] = palette[1]

        full = np.tile(
            tile,
            (
                (size + period - 1) // period,
                (size + period - 1) // period,
            ),
        )[:size, :size]

        def sample_damage() -> np.ndarray:
            damaged = full.copy()
            damage_count = gridvars.get("damage_count", random.randint(2, 5))
            for damage_index in range(damage_count):
                height = gridvars.get(
                    f"damage_{damage_index}_height",
                    random.randint(2, min(5, size - 1)),
                )
                width = gridvars.get(
                    f"damage_{damage_index}_width",
                    random.randint(2, min(6, size - 1)),
                )
                top = gridvars.get(
                    f"damage_{damage_index}_top", random.randint(0, size - height)
                )
                left = gridvars.get(
                    f"damage_{damage_index}_left", random.randint(0, size - width)
                )
                damaged[top : top + height, left : left + width] = taskvars[
                    "hole_color"
                ]
            return damaged

        try:
            return retry(
                sample_damage,
                lambda candidate: (
                    np.count_nonzero(candidate == taskvars["hole_color"]) >= 8
                    and smallest_period(candidate) == period
                ),
                max_attempts=80,
            )
        except ValueError:
            damaged = full.copy()
            damaged[1:3, 1:4] = taskvars["hole_color"]
            return damaged

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        hole_color = taskvars["hole_color"]
        recovered = None
        period = None
        maximum_period = min(grid.shape)

        for candidate in range(1, maximum_period + 1):
            mapping = {}
            consistent = True
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    value = int(grid[row, col])
                    if value == hole_color:
                        continue
                    key = (row % candidate, col % candidate)
                    if key in mapping and mapping[key] != value:
                        consistent = False
                        break
                    mapping[key] = value
                if not consistent:
                    break
            if consistent and len(mapping) == candidate * candidate:
                period = candidate
                recovered = mapping
                break

        if period is None:
            return grid.copy()
        output = np.zeros_like(grid)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                output[row, col] = recovered[(row % period, col % period)]
        return output
