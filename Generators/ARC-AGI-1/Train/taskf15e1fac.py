from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects


class Taskf15e1facGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. A {color('background_color')} grid contains isolated {color('marker_color')} cells along one outer edge.",
            "2. Isolated {color('pattern_color')} seed cells lie on one perpendicular outer edge.",
            "3. The two occupied edges, dimensions, marker count and spacing, and seed profile vary by example.",
            "4. Neither color occupies the shared corner, so the two boundary roles remain distinct.",
            "5. Every passed marker changes the profile offset by {vars['shift_per_marker']} cell toward the interior from the marker edge.",
        ]
        transformation_reasoning_chain = [
            "1. Read the {color('pattern_color')} boundary cells as a one-dimensional seed profile and sweep it inward from that edge.",
            "2. Treat ordered {color('marker_color')} boundary positions as thresholds along the sweep direction.",
            "3. Copy the profile through each strip, adding {vars['shift_per_marker']} inward offset for every threshold already passed.",
            "4. Clip shifted pattern cells at the far edge and preserve all {color('marker_color')} cells.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        marker, pattern = random.sample(range(1, 10), 2)
        taskvars = {
            "background_color": 0,
            "marker_color": marker,
            "pattern_color": pattern,
            "shift_per_marker": 1,
        }
        train_gridvars = [
            {
                "rows": 17,
                "cols": 12,
                "marker_edge": "left",
                "marker_positions": [4, 10],
                "pattern_edge": "top",
                "pattern_positions": [1, 5, 7, 9],
            },
            {
                "rows": 14,
                "cols": 10,
                "marker_edge": "right",
                "marker_positions": [3, 7, 11],
                "pattern_edge": "top",
                "pattern_positions": [2, 6],
            },
            {
                "rows": 12,
                "cols": 12,
                "marker_edge": "bottom",
                "marker_positions": [4, 8],
                "pattern_edge": "left",
                "pattern_positions": [2, 6, 9],
            },
            {
                "rows": 13,
                "cols": 16,
                "marker_edge": "top",
                "marker_positions": [3, 8, 12],
                "pattern_edge": "right",
                "pattern_positions": [2, 5, 9],
            },
        ]
        test_gridvars = {
            "rows": 15,
            "cols": 19,
            "marker_edge": "bottom",
            "marker_positions": [2, 6, 11, 15],
            "pattern_edge": "right",
            "pattern_positions": [2, 5, 8, 11, 13],
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
        marker = taskvars["marker_color"]
        pattern = taskvars["pattern_color"]
        rows, cols = gridvars["rows"], gridvars["cols"]
        marker_edge = gridvars["marker_edge"]
        pattern_edge = gridvars["pattern_edge"]

        def jittered(values, limit):
            return [
                min(limit - 2, max(1, value + random.randint(-1, 1)))
                for value in values
            ]

        def validate_placed_positions(name, values, base_values, limit):
            if not isinstance(values, (list, tuple)) or len(values) != len(
                base_values
            ):
                raise ValueError(f"{name} must align one-for-one with its base positions")
            placed = []
            for value, base_value in zip(values, base_values):
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                ):
                    raise ValueError(f"{name} entries must be integers")
                position = int(value)
                allowed = {
                    min(limit - 2, max(1, base_value + delta))
                    for delta in (-1, 0, 1)
                }
                if position not in allowed:
                    raise ValueError(
                        f"{name} entries must stay inside their clamped jitter envelopes"
                    )
                placed.append(position)
            return placed

        def build_scene():
            grid = np.full((rows, cols), background, dtype=int)
            marker_limit = rows if marker_edge in ("left", "right") else cols
            pattern_limit = rows if pattern_edge in ("left", "right") else cols
            marker_positions = gridvars.get(
                "placed_marker_positions",
                jittered(
                    gridvars["marker_positions"],
                    marker_limit,
                ),
            )
            pattern_positions = gridvars.get(
                "placed_pattern_positions",
                jittered(
                    gridvars["pattern_positions"],
                    pattern_limit,
                ),
            )
            marker_positions = validate_placed_positions(
                "placed_marker_positions",
                marker_positions,
                gridvars["marker_positions"],
                marker_limit,
            )
            pattern_positions = validate_placed_positions(
                "placed_pattern_positions",
                pattern_positions,
                gridvars["pattern_positions"],
                pattern_limit,
            )
            for position in marker_positions:
                if marker_edge == "top":
                    grid[0, position] = marker
                elif marker_edge == "bottom":
                    grid[rows - 1, position] = marker
                elif marker_edge == "left":
                    grid[position, 0] = marker
                else:
                    grid[position, cols - 1] = marker
            for position in pattern_positions:
                if pattern_edge == "top":
                    grid[0, position] = pattern
                elif pattern_edge == "bottom":
                    grid[rows - 1, position] = pattern
                elif pattern_edge == "left":
                    grid[position, 0] = pattern
                else:
                    grid[position, cols - 1] = pattern
            return grid

        expected_objects = len(gridvars["marker_positions"]) + len(
            gridvars["pattern_positions"]
        )

        def valid_scene(grid):
            objects = find_connected_objects(
                grid,
                diagonal_connectivity=False,
                background=background,
                monochromatic=True,
            )
            return (
                len(objects) == expected_objects
                and int(np.sum(grid == marker))
                == len(gridvars["marker_positions"])
                and int(np.sum(grid == pattern))
                == len(gridvars["pattern_positions"])
            )

        return retry(build_scene, valid_scene, max_attempts=100)

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        marker = taskvars["marker_color"]
        pattern = taskvars["pattern_color"]
        shift = taskvars["shift_per_marker"]
        perceived = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        )
        if len(perceived) == 0:
            return grid.copy()

        candidates = [
            grid.copy(),
            np.rot90(grid, 1),
            np.rot90(grid, 2),
            np.rot90(grid, 3),
            np.fliplr(grid),
            np.flipud(grid),
            grid.T.copy(),
            np.flipud(np.fliplr(grid)).T,
        ]
        canonical = None
        orientation_index = None
        for index, candidate in enumerate(candidates):
            marker_cells = np.argwhere(candidate == marker)
            pattern_cells = np.argwhere(candidate == pattern)
            if len(marker_cells) == 0 or len(pattern_cells) == 0:
                continue
            markers_on_top = all(int(row) == 0 for row, _ in marker_cells)
            pattern_on_left = all(int(col) == 0 for _, col in pattern_cells)
            if markers_on_top and pattern_on_left:
                canonical = candidate
                orientation_index = index
                break
        if canonical is None:
            return grid.copy()

        marker_columns = sorted(
            int(col) for _, col in np.argwhere(canonical == marker)
        )
        seed_rows = sorted(
            int(row) for row, _ in np.argwhere(canonical == pattern)
        )
        canonical_output = canonical.copy()
        for col in range(canonical.shape[1]):
            passed_markers = sum(
                1 for marker_col in marker_columns if marker_col <= col
            )
            row_shift = passed_markers * shift
            for seed_row in seed_rows:
                target_row = seed_row + row_shift
                if (
                    0 <= target_row < canonical.shape[0]
                    and canonical_output[target_row, col] != marker
                ):
                    canonical_output[target_row, col] = pattern

        if orientation_index == 0:
            return canonical_output
        if orientation_index == 1:
            return np.rot90(canonical_output, 3)
        if orientation_index == 2:
            return np.rot90(canonical_output, 2)
        if orientation_index == 3:
            return np.rot90(canonical_output, 1)
        if orientation_index == 4:
            return np.fliplr(canonical_output)
        if orientation_index == 5:
            return np.flipud(canonical_output)
        if orientation_index == 6:
            return canonical_output.T.copy()
        return np.flipud(np.fliplr(canonical_output)).T
