from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import parse_objects_by_color


class Task9af7a82cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a compact rectangular grid containing several non-{color('background_color')} colors.",
            "2. Every foreground color has a different number of cells, so their frequencies have a strict order.",
            "3. Input dimensions, number of colors, color identities, counts, and spatial arrangements vary between examples.",
            "4. The largest color count determines the output height, and the number of colors determines its width.",
        ]
        transformation_reasoning_chain = [
            "1. Count the cells of every non-{color('background_color')} color.",
            "2. Order the colors by {vars['frequency_order']} frequency.",
            "3. Create one top-aligned output column per ordered color, with height equal to the largest count.",
            "4. Fill each column in its own color for exactly its counted height and leave all lower cells {color('background_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"background_color": 0, "frequency_order": "descending"}
        layouts = [
            {"rows": 4, "cols": 4, "num_colors": 3, "counts": [5, 3, 1]},
            {"rows": 4, "cols": 5, "num_colors": 4, "counts": [6, 4, 2, 1]},
            {"rows": 5, "cols": 4, "num_colors": 2, "counts": [7, 3]},
            {"rows": 5, "cols": 5, "num_colors": 4, "counts": [8, 5, 3, 1]},
        ]
        test_layout = {"rows": 6, "cols": 5, "num_colors": 5, "counts": [9, 7, 5, 3, 1]}

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {"input": input_grid, "output": self.transform_input(input_grid, taskvars)}

        return taskvars, {
            "train": [make_pair(layout) for layout in layouts],
            "test": [make_pair(test_layout)],
        }

    def create_input(self, taskvars, gridvars):
        rows, cols = gridvars["rows"], gridvars["cols"]
        colors = list(
            gridvars.get(
                "colors",
                random.sample(range(1, 10), gridvars["num_colors"]),
            )
        )

        def sample_grid():
            candidate = np.zeros((rows, cols), dtype=int)
            for color, count in zip(colors, gridvars["counts"]):
                remaining = int(np.count_nonzero(candidate == taskvars["background_color"]))
                random_cell_coloring(
                    candidate,
                    color,
                    density=(count + 0.25) / remaining,
                    background=taskvars["background_color"],
                )
            return candidate

        def valid(candidate):
            observed = [int(np.count_nonzero(candidate == color)) for color in colors]
            return observed == gridvars["counts"]

        try:
            sampled_grid = retry(sample_grid, valid, max_attempts=100)
        except ValueError:
            sampled_grid = np.zeros((rows, cols), dtype=int)
            positions = list(range(rows * cols))
            cursor = 0
            for color, count in zip(colors, gridvars["counts"]):
                for index in positions[cursor : cursor + count]:
                    row, col = divmod(index, cols)
                    sampled_grid[row, col] = color
                cursor += count
        input_grid = np.asarray(
            gridvars.get("input_grid", sampled_grid),
            dtype=int,
        )
        if input_grid.shape != (rows, cols):
            raise ValueError("input_grid shape does not match rows and cols")
        if not valid(input_grid):
            raise ValueError("input_grid does not match the requested color counts")
        return input_grid.copy()

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        frequency_order = taskvars["frequency_order"]
        objects = list(parse_objects_by_color(grid, background=background))
        counts = [(next(iter(obj.colors)), len(obj)) for obj in objects]
        counts.sort(key=lambda item: (-item[1], item[0]))
        if frequency_order != "descending":
            counts.reverse()
        if not counts:
            return np.zeros((1, 1), dtype=int)
        output = np.full((counts[0][1], len(counts)), background, dtype=int)
        for col, (color, count) in enumerate(counts):
            output[:count, col] = color
        return output
