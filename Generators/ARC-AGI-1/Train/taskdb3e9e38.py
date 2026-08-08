from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject
import numpy as np
import random


class Taskdb3e9e38Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains one top-anchored vertical line of {color('spine_color')} cells on an otherwise empty grid.",
            "2. The line column, its length, and the number of empty rows below it vary between examples.",
            "3. The line may be near either side, so a pattern expanded around it can be clipped by a grid boundary.",
            "4. All examples share the {color('spine_color')} spine role and the {color('accent_color')} alternating accent role.",
        ]
        transformation_reasoning_chain = [
            "1. Measure the length of the top-anchored {color('spine_color')} vertical spine.",
            "2. On each spine row, expand horizontally by one cell for every spine cell below that row, clipping the span at the grid edges.",
            "3. Color positions an even horizontal distance from the spine {color('spine_color')} and positions an odd distance {color('accent_color')}.",
            "4. Leave all rows below the original spine empty.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        spine_color, accent_color = random.sample(range(1, 10), 2)
        taskvars = {
            "spine_color": spine_color,
            "accent_color": accent_color,
        }

        def make_pair(length, placement):
            def sample_geometry():
                rows = length + random.randint(1, 3)
                cols = random.randint(max(7, length + 2), min(18, 2 * length + 5))
                if placement == "left":
                    column = random.randint(0, max(0, length // 3))
                elif placement == "right":
                    column = cols - 1 - random.randint(0, max(0, length // 3))
                else:
                    column = random.randint(length // 2, cols - 1 - length // 2)
                return {"rows": rows, "cols": cols, "length": length, "column": column}

            try:
                gridvars = retry(
                    sample_geometry,
                    lambda values: (
                        values["rows"] <= 30
                        and values["cols"] <= 30
                        and 0 <= values["column"] < values["cols"]
                    ),
                    max_attempts=20,
                )
            except ValueError:
                gridvars = {
                    "rows": length + 2,
                    "cols": 2 * length + 1,
                    "length": length,
                    "column": length,
                }
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [
            make_pair(3, "center"),
            make_pair(4, "left"),
            make_pair(5, "right"),
            make_pair(6, "center"),
        ]
        test = [make_pair(7, random.choice(["left", "right"]))]
        return taskvars, {"train": train, "test": test}

    def create_input(self, taskvars, gridvars):
        grid = np.zeros((gridvars["rows"], gridvars["cols"]), dtype=int)
        line = np.full(
            (gridvars["length"], 1), taskvars["spine_color"], dtype=int
        )
        GridObject.from_array(line, offset=(0, gridvars["column"])).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        spine_color = taskvars["spine_color"]
        accent_color = taskvars["accent_color"]
        positions = np.argwhere(grid == spine_color)
        if positions.size == 0:
            return grid.copy()
        column_counts = [
            int(np.count_nonzero(grid[:, col] == spine_color))
            for col in range(grid.shape[1])
        ]
        spine_col = int(np.argmax(column_counts))
        spine_rows = np.where(grid[:, spine_col] == spine_color)[0]
        length = int(np.max(spine_rows)) + 1
        output = grid.copy()
        for row in range(length):
            radius = length - row - 1
            left = max(0, spine_col - radius)
            right = min(grid.shape[1] - 1, spine_col + radius)
            for col in range(left, right + 1):
                output[row, col] = (
                    spine_color
                    if abs(col - spine_col) % 2 == 0
                    else accent_color
                )
        return output
