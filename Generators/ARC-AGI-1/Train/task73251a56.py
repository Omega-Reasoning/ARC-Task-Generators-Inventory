from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import GridObject


class Task73251a56Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['grid_size']}x{vars['grid_size']} square whose intact cells form a regular coordinate-dependent color pattern.",
            "2. The pattern uses a consecutive non-background palette from color 1 through the largest visible color, while {color('background_color')} marks erased cells.",
            "3. The most frequent visible color is the zero phase of the cyclic palette.",
            "4. One or more rectangular or connected regions of the pattern have been replaced by {color('background_color')} holes.",
            "5. Enough intact cells remain to identify both the palette and its dominant zero-phase color.",
        ]
        transformation_reasoning_chain = [
            "1. Infer the consecutive palette size and its most frequent visible zero-phase color.",
            "2. Give a missing main-diagonal cell phase {vars['diagonal_phase']}; otherwise compute floor(abs(row-column)/(min(row,column)+{vars['denominator_offset']})).",
            "3. Advance the zero-phase color by that phase, wrapping cyclically through the consecutive palette.",
            "4. Fill every {color('background_color')} hole with its computed color and preserve all originally visible cells.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {
            "background_color": 0,
            "grid_size": random.randint(17, 27),
            "denominator_offset": 2,
            "diagonal_phase": 1,
        }
        palette_sizes = random.sample(range(5, 10), 4)
        train_specs = [
            (palette_sizes[0], random.randint(1, palette_sizes[0]), ["above", "above"]),
            (palette_sizes[1], random.randint(1, palette_sizes[1]), ["below", "below"]),
            (palette_sizes[2], random.randint(1, palette_sizes[2]), ["diagonal"]),
            (
                palette_sizes[3],
                random.randint(1, palette_sizes[3]),
                ["above", "below", "diagonal"],
            ),
        ]
        train = []
        for palette_size, base_color, regions in train_specs:
            input_grid = self.create_input(
                taskvars,
                {
                    "palette_size": palette_size,
                    "base_color": base_color,
                    "regions": regions,
                },
            )
            train.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )
        test_input = self.create_input(
            taskvars,
            {
                "palette_size": 9,
                "base_color": random.randint(1, 9),
                "regions": ["diagonal", "above", "below", "above", "below"],
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

    def create_input(self, taskvars, gridvars):
        background = taskvars["background_color"]
        size = taskvars["grid_size"]
        offset = taskvars["denominator_offset"]
        diagonal_phase = taskvars["diagonal_phase"]
        palette_size = gridvars["palette_size"]
        base_color = gridvars["base_color"]
        regions = gridvars["regions"]
        full = np.empty((size, size), dtype=int)
        for row in range(size):
            for col in range(size):
                if row == col:
                    phase = diagonal_phase
                else:
                    phase = abs(row - col) // (min(row, col) + offset)
                full[row, col] = ((base_color - 1 + phase) % palette_size) + 1

        def sample_mask():
            height = random.randint(2, min(6, size // 3))
            width = random.randint(2, min(6, size // 3))
            return create_object(
                height,
                width,
                1,
                contiguity=Contiguity.FOUR,
                background=0,
            )

        def useful_mask(mask):
            return np.count_nonzero(mask) >= 3

        def build_damage():
            damaged = np.array(full, copy=True)
            for region in regions:
                try:
                    mask = retry(sample_mask, useful_mask, max_attempts=30)
                except ValueError:
                    mask = np.ones((2, 3), dtype=int)
                height, width = mask.shape

                def sample_position():
                    return (
                        random.randint(0, size - height),
                        random.randint(0, size - width),
                    )

                def in_region(position):
                    row, col = position
                    center_row = row + height // 2
                    center_col = col + width // 2
                    if region == "above":
                        return center_col - center_row >= 3
                    if region == "below":
                        return center_row - center_col >= 3
                    return abs(center_row - center_col) <= 1

                try:
                    position = retry(
                        sample_position, in_region, max_attempts=50
                    )
                except ValueError:
                    if region == "above":
                        position = (1, size - width - 2)
                    elif region == "below":
                        position = (size - height - 2, 1)
                    else:
                        middle = size // 2
                        position = (middle - height // 2, middle - width // 2)
                GridObject.from_array(mask, offset=position).cut(
                    damaged, background=background
                )
            return damaged

        def inference_is_unambiguous(damaged):
            visible = damaged[damaged != background]
            if visible.size == 0 or int(np.max(visible)) != palette_size:
                return False
            colors, counts = np.unique(visible, return_counts=True)
            maximum = int(np.max(counts))
            winners = colors[counts == maximum]
            return len(winners) == 1 and int(winners[0]) == base_color

        def sample_valid_damage() -> np.ndarray:
            try:
                return retry(
                    build_damage, inference_is_unambiguous, max_attempts=50
                )
            except ValueError:
                fallback = np.array(full, copy=True)
                fallback_cells = {
                    (row, col, int(fallback[row, col]))
                    for row in range(1, min(3, size))
                    for col in range(size - min(3, size), size - 1)
                }
                GridObject(fallback_cells).cut(fallback, background=background)
                return fallback

        hole_cells = [
            tuple(int(value) for value in coordinate)
            for coordinate in gridvars.get(
                "hole_cells",
                np.argwhere(sample_valid_damage() == background),
            )
        ]
        if any(
            len(coordinate) != 2
            or not 0 <= coordinate[0] < size
            or not 0 <= coordinate[1] < size
            for coordinate in hole_cells
        ):
            raise ValueError("hole_cells must contain in-bounds row-column pairs")
        damaged = np.array(full, copy=True)
        for row, col in hole_cells:
            damaged[row, col] = background
        if not inference_is_unambiguous(damaged):
            raise ValueError("hole_mask makes the base pattern ambiguous")
        return damaged

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        offset = taskvars["denominator_offset"]
        diagonal_phase = taskvars["diagonal_phase"]
        output = np.array(grid, copy=True)
        visible = grid[grid != background]
        if visible.size == 0:
            return output
        palette_size = int(np.max(visible))
        colors, counts = np.unique(visible, return_counts=True)
        base_color = int(colors[int(np.argmax(counts))])
        additions = set()
        for row, col in zip(*np.where(grid == background)):
            row = int(row)
            col = int(col)
            if row == col:
                phase = diagonal_phase
            else:
                phase = abs(row - col) // (min(row, col) + offset)
            color = ((base_color - 1 + phase) % palette_size) + 1
            additions.add((row, col, color))
        if additions:
            GridObject(additions).paste(output)
        return output
