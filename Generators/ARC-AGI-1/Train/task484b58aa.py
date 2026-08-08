from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import Contiguity, create_object, retry


class Task484b58aaGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Input grids contain a dense multicolor pattern with {color('missing_color')} cells marking erased values.",
            "2. Complete rows repeat vertically with an example-specific period that is smaller than the grid height.",
            "3. Every non-erased cell agrees with all visible cells at the same column and the same row index modulo that period.",
            "4. Several rectangular or connected patches are erased, but every missing position has a visible periodic witness.",
        ]
        transformation_reasoning_chain = [
            "1. Test positive vertical shifts and select the smallest shift for which all overlapping visible cells agree.",
            "2. Treat rows with equal indices modulo the selected shift as copies of one periodic row.",
            "3. Replace each {color('missing_color')} cell with the visible value from its column and periodic row class.",
            "4. Preserve every originally visible non-{color('missing_color')} cell unchanged.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"missing_color": 0}
        periods = [3, 4, 5, 6]
        train = []
        for index, period in enumerate(periods):
            gridvars = {
                "height": 22 + index,
                "width": random.randint(20, 29),
                "period": period,
                "patches": 2 + index % 3,
            }
            grid = self.create_input(taskvars, gridvars)
            train.append({"input": grid, "output": self.transform_input(grid, taskvars)})
        test_grid = self.create_input(
            taskvars,
            {"height": 29, "width": random.randint(23, 29), "period": 7, "patches": 5},
        )
        return taskvars, {
            "train": train,
            "test": [
                {"input": test_grid, "output": self.transform_input(test_grid, taskvars)}
            ],
        }

    def create_input(self, taskvars, gridvars):
        height = gridvars["height"]
        width = gridvars["width"]
        period = gridvars["period"]
        palette_size = int(gridvars.get(
            "palette_size",
            random.randint(3, 7),
        ))
        palette = list(gridvars.get(
            "palette",
            random.sample(range(1, 10), palette_size),
        ))
        sampled_base_rows = np.asarray(
            [
                [
                    gridvars.get(
                        f"base_{row}_{col}",
                        random.choice(palette),
                    )
                    for col in range(width)
                ]
                for row in range(period)
            ],
            dtype=int,
        )
        routed_base_rows = gridvars.get("base_rows", sampled_base_rows)
        if (
            isinstance(routed_base_rows, (list, tuple))
            and routed_base_rows
            and isinstance(routed_base_rows[0], str)
        ):
            routed_base_rows = [
                [int(value) for value in row]
                for row in routed_base_rows
            ]
        base_rows = np.asarray(routed_base_rows, dtype=int)
        full = np.asarray([base_rows[row % period] for row in range(height)], dtype=int)

        def sample_erased():
            candidate = full.copy()
            for patch_index in range(gridvars["patches"]):
                patch_height = int(gridvars.get(
                    f"patch_{patch_index}_height",
                    random.randint(2, min(6, height // 3)),
                ))
                patch_width = int(gridvars.get(
                    f"patch_{patch_index}_width",
                    random.randint(2, min(7, width // 3)),
                ))
                sampled_mask = create_object(
                    patch_height,
                    patch_width,
                    1,
                    contiguity=Contiguity.FOUR,
                    background=0,
                )
                routed_mask = gridvars.get(
                    f"patch_{patch_index}_mask",
                    sampled_mask,
                )
                if (
                    isinstance(routed_mask, (list, tuple))
                    and routed_mask
                    and isinstance(routed_mask[0], str)
                ):
                    routed_mask = [
                        [int(value) for value in mask_row]
                        for mask_row in routed_mask
                    ]
                mask = np.asarray(routed_mask)
                row = int(gridvars.get(
                    f"patch_{patch_index}_row",
                    random.randint(0, height - patch_height),
                ))
                col = int(gridvars.get(
                    f"patch_{patch_index}_col",
                    random.randint(0, width - patch_width),
                ))
                view = candidate[row : row + patch_height, col : col + patch_width]
                view[mask != 0] = taskvars["missing_color"]
            return candidate

        def witnesses_remain(candidate):
            return np.any(candidate == taskvars["missing_color"]) and all(
                np.any(candidate[residue::period, col] != taskvars["missing_color"])
                for residue in range(period)
                for col in range(width)
            )

        try:
            sampled_erased = retry(sample_erased, witnesses_remain, max_attempts=30)
        except ValueError:
            sampled_erased = full.copy()
            sampled_erased[1:3, 1:4] = taskvars["missing_color"]
        return sampled_erased

    def transform_input(self, grid, taskvars):
        missing = taskvars["missing_color"]
        rows, cols = grid.shape
        period = None
        for shift in range(1, rows):
            upper = grid[:-shift]
            lower = grid[shift:]
            visible = (upper != missing) & (lower != missing)
            if np.any(visible) and np.all(upper[visible] == lower[visible]):
                period = shift
                break
        if period is None:
            return grid.copy()
        output = grid.copy()
        for row in range(rows):
            for col in range(cols):
                if output[row, col] != missing:
                    continue
                for witness_row in range(row % period, rows, period):
                    if grid[witness_row, col] != missing:
                        output[row, col] = grid[witness_row, col]
                        break
        return output
