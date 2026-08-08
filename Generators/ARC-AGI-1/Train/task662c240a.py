from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import get_objects_from_raster


class Task662c240aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a vertical raster of {vars['num_blocks']} consecutive {vars['block_size']}x{vars['block_size']} two-color blocks.",
            "2. Each block uses its own pair of non-background colors and has no delimiter between it and the next block.",
            "3. Exactly two blocks are symmetric across their main diagonal from top-left to bottom-right.",
            "4. The remaining block violates that symmetry in at least one off-diagonal cell pair.",
        ]
        transformation_reasoning_chain = [
            "1. Partition the raster into its {vars['num_blocks']} consecutive {vars['block_size']}x{vars['block_size']} blocks.",
            "2. Compare each block with its transpose to test reflection symmetry across the main diagonal.",
            "3. Return the unique block that is not equal to its transpose.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"block_size": 3, "num_blocks": 3}
        schedules = [
            {"asymmetric_index": 0, "mismatch_pairs": [(0, 1)]},
            {"asymmetric_index": 1, "mismatch_pairs": [(0, 2)]},
            {"asymmetric_index": 2, "mismatch_pairs": [(1, 2)]},
            {"asymmetric_index": 0, "mismatch_pairs": [(0, 2)]},
        ]
        train = []
        for schedule in schedules:
            grid = self.create_input(taskvars, schedule)
            train.append({"input": grid, "output": self.transform_input(grid, taskvars)})
        test_grid = self.create_input(
            taskvars,
            {"asymmetric_index": 2, "mismatch_pairs": [(0, 1), (1, 2)]},
        )
        return taskvars, {
            "train": train,
            "test": [{"input": test_grid, "output": self.transform_input(test_grid, taskvars)}],
        }

    def create_input(self, taskvars, gridvars):
        size = taskvars["block_size"]
        colors = gridvars.get(
            "colors",
            random.sample(range(1, 10), 2 * taskvars["num_blocks"]),
        )

        def sample_symmetric():
            seed = np.zeros((size, size), dtype=int)
            seed = np.asarray(
                gridvars.get(
                    f"symmetric_seed_{index}",
                    random_cell_coloring(
                        seed,
                        1,
                        density=gridvars.get(
                            f"symmetric_density_{index}",
                            random.choice([0.33, 0.44, 0.55, 0.66]),
                        ),
                        background=0,
                    ),
                ),
                dtype=int,
            )
            upper = np.triu(seed)
            return np.maximum(upper, upper.T)

        fallback_patterns = [
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
            np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=int),
        ]
        masks = []
        for index in range(taskvars["num_blocks"]):
            try:
                mask = retry(
                    sample_symmetric,
                    lambda value: np.any(value == 0)
                    and np.any(value == 1)
                    and all(not np.array_equal(value, prior) for prior in masks),
                    max_attempts=40,
                )
            except ValueError:
                mask = fallback_patterns[index].copy()
            masks.append(mask)
        asymmetric_index = gridvars["asymmetric_index"]
        asymmetric = masks[asymmetric_index].copy()
        for row, col in gridvars["mismatch_pairs"]:
            flip_first = gridvars.get(
                f"flip_first_{int(row)}_{int(col)}",
                random.choice([True, False]),
            )
            if flip_first:
                asymmetric[row, col] = 1 - asymmetric[row, col]
            else:
                asymmetric[col, row] = 1 - asymmetric[col, row]
        masks[asymmetric_index] = asymmetric
        blocks = []
        for index, mask in enumerate(masks):
            first_color = colors[2 * index]
            second_color = colors[2 * index + 1]
            blocks.append(np.where(mask == 1, first_color, second_color))
        return np.vstack(blocks)

    def transform_input(self, grid, taskvars):
        block_size = taskvars["block_size"]
        raster = get_objects_from_raster(
            grid,
            block_size,
            block_size,
            has_delimiters=False,
        )
        blocks = [row[0].to_array() for row in raster if row]
        asymmetric = [block for block in blocks if not np.array_equal(block, block.T)]
        if len(asymmetric) != 1:
            return np.zeros((block_size, block_size), dtype=int)
        return asymmetric[0].copy()
