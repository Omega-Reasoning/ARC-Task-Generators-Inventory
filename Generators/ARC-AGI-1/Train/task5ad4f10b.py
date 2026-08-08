from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import parse_objects_by_color


class Task5ad4f10bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Input grids use {color('background_color')} as empty space and contain exactly two non-background colors.",
            "2. One color forms solid equal-sized square blocks aligned to a {vars['output_size']}x{vars['output_size']} lattice.",
            "3. The other color appears as sparse single-cell noise outside and between those blocks.",
            "4. Block color, noise color, block side, lattice occupancy, grid size, and lattice placement vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Group non-background cells by color and identify the repeated-block field by its shared non-unit run length.",
            "2. Infer the block side from the common horizontal and vertical run lengths of the dense color.",
            "3. Divide its bounding lattice into {vars['output_size']}x{vars['output_size']} block positions and record which positions are occupied.",
            "4. Return a {vars['output_size']}x{vars['output_size']} {color('background_color')} grid, coloring occupied positions with the sparse marker color.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"background_color": 0, "output_size": 3}
        schedules = [
            {"block_side": 2, "occupancy": [[1, 0, 1], [0, 1, 0], [1, 1, 0]]},
            {"block_side": 3, "occupancy": [[1, 1, 0], [0, 1, 1], [1, 0, 0]]},
            {"block_side": 4, "occupancy": [[0, 1, 1], [1, 0, 1], [1, 1, 0]]},
            {"block_side": 2, "occupancy": [[1, 0, 0], [1, 1, 1], [0, 0, 1]]},
        ]
        train = []
        for schedule in schedules:
            dense_color, marker_color = random.sample(range(1, 10), 2)
            grid = self.create_input(
                taskvars,
                {
                    **schedule,
                    "dense_color": dense_color,
                    "marker_color": marker_color,
                },
            )
            train.append({"input": grid, "output": self.transform_input(grid, taskvars)})
        dense_color, marker_color = random.sample(range(1, 10), 2)
        test_grid = self.create_input(
            taskvars,
            {
                "block_side": 5,
                "occupancy": [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
                "dense_color": dense_color,
                "marker_color": marker_color,
            },
        )
        return taskvars, {
            "train": train,
            "test": [{"input": test_grid, "output": self.transform_input(test_grid, taskvars)}],
        }

    def create_input(self, taskvars, gridvars):
        background = taskvars["background_color"]
        output_size = taskvars["output_size"]
        block_side = gridvars["block_side"]
        occupancy = np.array(gridvars["occupancy"], dtype=int)
        dense_color = gridvars["dense_color"]
        marker_color = gridvars["marker_color"]
        lattice_span = output_size * block_side
        rows = gridvars.get("rows", random.randint(lattice_span + 5, 30))
        cols = gridvars.get("cols", random.randint(lattice_span + 5, 30))
        top = gridvars.get("top", random.randint(1, rows - lattice_span - 1))
        left = gridvars.get("left", random.randint(1, cols - lattice_span - 1))
        base = np.full((rows, cols), background, dtype=int)
        for block_row, block_col in zip(*np.where(occupancy == 1)):
            row_start = top + int(block_row) * block_side
            col_start = left + int(block_col) * block_side
            base[
                row_start : row_start + block_side,
                col_start : col_start + block_side,
            ] = dense_color
        available = int(np.count_nonzero(base == background))
        noise_count = gridvars.get(
            "noise_count",
            random.randint(8, min(30, max(8, available // 10))),
        )

        def sample_noise():
            candidate = base.copy()
            density = min(1.0, (noise_count + 0.5) / available)
            return random_cell_coloring(
                candidate,
                marker_color,
                density=density,
                background=background,
            )

        def has_isolated_marker(candidate):
            marker_rows, marker_cols = np.where(candidate == marker_color)
            if len(marker_rows) < 2:
                return False
            for row, col in zip(marker_rows, marker_cols):
                if all(
                    not (0 <= row + delta_row < rows and 0 <= col + delta_col < cols)
                    or candidate[row + delta_row, col + delta_col] != marker_color
                    for delta_row, delta_col in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                ):
                    return True
            return False

        try:
            sampled_candidate = retry(sample_noise, has_isolated_marker, max_attempts=40)
        except ValueError:
            sampled_candidate = base.copy()
            placed = []
            for row, col in zip(*np.where(sampled_candidate == background)):
                if all(abs(int(row) - old_row) + abs(int(col) - old_col) > 1 for old_row, old_col in placed):
                    sampled_candidate[int(row), int(col)] = marker_color
                    placed.append((int(row), int(col)))
                    if len(placed) == noise_count:
                        break
        candidate = np.asarray(
            gridvars.get("candidate", sampled_candidate), dtype=int
        ).copy()
        noise_cells = gridvars.get("noise_cells")
        if noise_cells is not None:
            candidate[:, :] = base
            for row, col in noise_cells:
                if not (0 <= row < rows and 0 <= col < cols):
                    raise ValueError("forced noise cell lies outside the grid")
                if candidate[row, col] != background:
                    raise ValueError("forced noise overlaps a dense block")
                candidate[row, col] = marker_color
        return candidate

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        output_size = taskvars["output_size"]
        groups = parse_objects_by_color(grid, background=background)
        if len(groups) != 2:
            return np.full((output_size, output_size), background, dtype=int)
        def maximal_runs(mask):
            lengths = []
            for line in mask:
                current = 0
                for occupied in line:
                    if occupied:
                        current += 1
                    elif current:
                        lengths.append(current)
                        current = 0
                if current:
                    lengths.append(current)
            return lengths

        candidates = []
        for group in groups:
            color = int(next(iter(group.colors)))
            mask = grid == color
            runs = maximal_runs(mask) + maximal_runs(mask.T)
            run_scale = runs[0]
            for length in runs[1:]:
                run_scale = int(np.gcd(run_scale, length))
            candidates.append((run_scale, color))
        candidates.sort(reverse=True)
        dense_color = candidates[0][1]
        marker_color = candidates[1][1]
        dense_mask = grid == dense_color
        runs = maximal_runs(dense_mask) + maximal_runs(dense_mask.T)
        if not runs:
            return np.full((output_size, output_size), background, dtype=int)
        block_side = runs[0]
        for length in runs[1:]:
            block_side = int(np.gcd(block_side, length))
        dense_rows, dense_cols = np.where(dense_mask)
        top = int(np.min(dense_rows))
        left = int(np.min(dense_cols))
        output = np.full((output_size, output_size), background, dtype=int)
        for block_row in range(output_size):
            for block_col in range(output_size):
                row_start = top + block_row * block_side
                col_start = left + block_col * block_side
                region = dense_mask[
                    row_start : row_start + block_side,
                    col_start : col_start + block_side,
                ]
                if region.shape == (block_side, block_side) and np.any(region):
                    output[block_row, block_col] = marker_color
        return output
