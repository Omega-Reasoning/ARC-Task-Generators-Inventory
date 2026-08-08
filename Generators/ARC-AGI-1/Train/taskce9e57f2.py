from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Taskce9e57f2Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains between three and {vars['maximum_bars']} separated vertical bars of {color('bar_color')} on {color('background_color')}.",
            "2. Every bar is one cell wide, is four-connected, and reaches the same bottom baseline.",
            "3. Blank columns separate neighboring bars so they form distinct connected objects.",
            "4. Bar heights vary within and between examples and include both odd and even values.",
            "5. Grid height, bar count, and the ordering of bar heights vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Find each vertical {color('bar_color')} object and measure its height.",
            "2. For each bar, compute floor(height divided by {vars['half_divisor']}).",
            "3. Starting at the common baseline, recolor exactly that many cells of the bar to {color('highlight_color')}.",
            "4. Preserve the upper cells of every bar as {color('bar_color')} and leave all {color('background_color')} cells unchanged.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        bar_color, highlight_color = random.sample(range(1, 10), 2)
        taskvars = {
            "background_color": 0,
            "bar_color": bar_color,
            "highlight_color": highlight_color,
            "half_divisor": 2,
            "maximum_bars": 5,
        }
        train_gridvars = [
            {"rows": 8, "heights": [3, 5, 6]},
            {"rows": 10, "heights": [8, 4, 7, 3]},
            {"rows": 11, "heights": [5, 9, 4]},
            {"rows": 12, "heights": [10, 6, 3, 8]},
        ]
        test_gridvars = {"rows": 13, "heights": [11, 4, 9, 6, 3]}

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
        base_heights = [int(value) for value in gridvars["heights"]]

        def shuffled_heights():
            values = list(base_heights)
            random.shuffle(values)
            return values

        heights = [
            int(value)
            for value in gridvars.get("ordered_heights", shuffled_heights())
        ]
        rows = int(gridvars["rows"])
        if (
            not 3 <= len(heights) <= int(taskvars["maximum_bars"])
            or sorted(heights) != sorted(base_heights)
            or any(not 1 <= height <= rows for height in heights)
        ):
            raise ValueError("ordered_heights must be a valid permutation of heights")
        cols = 2 * len(heights) + 1
        grid = np.full((rows, cols), taskvars["background_color"], dtype=int)
        for index, height in enumerate(heights):
            bar = np.full((height, 1), taskvars["bar_color"], dtype=int)
            GridObject.from_array(
                bar,
                offset=(rows - height, 1 + 2 * index),
            ).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        output = grid.copy()
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=taskvars["background_color"],
            monochromatic=True,
        ).with_color(taskvars["bar_color"])
        for obj in objects:
            lower_count = obj.height // taskvars["half_divisor"]
            rows = sorted({row for row, _, _ in obj.cells}, reverse=True)
            lower_rows = set(rows[:lower_count])
            for row, col, _ in obj.cells:
                if row in lower_rows:
                    output[row, col] = taskvars["highlight_color"]
        return output
