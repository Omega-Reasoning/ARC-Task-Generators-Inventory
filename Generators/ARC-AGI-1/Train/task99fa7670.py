from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects


class Task99fa7670Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is an otherwise {color('background_color')} grid containing one or more isolated colored seed cells.",
            "2. Seeds occupy distinct rows and may have different colors and columns.",
            "3. Every seed has empty space to its {vars['horizontal_direction']} and below it.",
            "4. Grid dimensions, seed count, seed colors, and path lengths vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Order the colored seeds from top to bottom.",
            "2. From each seed, extend its color {vars['horizontal_direction']} through the final column.",
            "3. Turn at that boundary and extend the same color {vars['vertical_direction']} through the final row.",
            "4. Process seeds in order so a lower seed overwrites the shared boundary segment below its own row.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {
            "background_color": 0,
            "horizontal_direction": "right",
            "vertical_direction": "down",
        }
        variant = random.randint(0, 2)
        layouts = [
            {"rows": 5, "cols": 6 + variant, "count": 1},
            {"rows": 7, "cols": 8, "count": 2},
            {"rows": 9, "cols": 7 + variant, "count": 3},
            {"rows": 8, "cols": 10, "count": 2},
        ]
        test_layout = {"rows": 10, "cols": 11 + variant, "count": 4}

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {"input": input_grid, "output": self.transform_input(input_grid, taskvars)}

        return taskvars, {
            "train": [make_pair(layout) for layout in layouts],
            "test": [make_pair(test_layout)],
        }

    def create_input(self, taskvars, gridvars):
        rows, cols, count = gridvars["rows"], gridvars["cols"], gridvars["count"]
        grid = np.full((rows, cols), taskvars["background_color"], dtype=int)
        def sample_natural_seed_rows():
            try:
                return retry(
                    lambda: sorted(random.sample(range(rows - 1), count)),
                    lambda values: len(set(values)) == count,
                    max_attempts=20,
                )
            except ValueError:
                return list(range(count))

        seed_rows = list(
            gridvars.get("seed_rows", sample_natural_seed_rows())
        )
        seed_cols = list(
            gridvars.get(
                "seed_cols",
                [random.randint(0, cols - 3) for _ in range(count)],
            )
        )
        seed_colors = list(
            gridvars.get("seed_colors", random.sample(range(1, 10), count))
        )
        if not (
            len(seed_rows) == len(seed_cols) == len(seed_colors) == count
            and len(set(seed_rows)) == count
            and all(0 <= row < rows - 1 for row in seed_rows)
            and all(0 <= col < cols - 1 for col in seed_cols)
            and all(color in range(1, 10) for color in seed_colors)
        ):
            raise ValueError("seed controls violate the input grammar")
        GridObject(
            {
                (row, col, color)
                for row, col, color in zip(seed_rows, seed_cols, seed_colors)
            }
        ).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        horizontal_direction = taskvars["horizontal_direction"]
        vertical_direction = taskvars["vertical_direction"]
        if horizontal_direction != "right" or vertical_direction != "down":
            return grid.copy()
        output = grid.copy()
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        )
        seeds = sorted(
            [cell for obj in objects for cell in obj.cells],
            key=lambda cell: (cell[0], cell[1]),
        )
        for row, col, color in seeds:
            output[row, col:] = color
            output[row:, -1] = color
        return output
