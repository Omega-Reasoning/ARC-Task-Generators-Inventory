from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects


class Taskbdad9b1fGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains one short horizontal segment of {color('horizontal_color')} and one short vertical segment of {color('vertical_color')}.",
            "2. The two fragments occupy different rows and columns and do not intersect in the input.",
            "3. Each fragment has enough visible cells for its orientation and supporting row or column to be unambiguous.",
            "4. Fragment positions, offsets, and lengths vary while the three color roles remain fixed within the episode.",
            "5. All other cells are empty.",
        ]
        transformation_reasoning_chain = [
            "1. Find the row of the {color('horizontal_color')} fragment and the column of the {color('vertical_color')} fragment.",
            "2. Fill that entire row with {color('horizontal_color')} from the left boundary to the right boundary.",
            "3. Fill that entire column with {color('vertical_color')} from the top boundary to the bottom boundary.",
            "4. Color the unique crossing cell {color('crossing_color')}.",
            "5. Preserve empty cells outside the completed row and column.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        horizontal_color, vertical_color, crossing_color = random.sample(range(1, 10), 3)
        taskvars = {
            "horizontal_color": horizontal_color,
            "vertical_color": vertical_color,
            "crossing_color": crossing_color,
        }

        def make_pair(fragment_length, spacing, horizontal_relation, vertical_relation):
            gridvars = {
                "rows": random.randint(6 + spacing, 10 + spacing),
                "cols": random.randint(6 + spacing, 10 + spacing),
                "fragment_length": fragment_length,
                "minimum_spacing": spacing,
                "horizontal_relation": horizontal_relation,
                "vertical_relation": vertical_relation,
            }
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [
            make_pair(2, 1, "left", "above"),
            make_pair(2, 2, "right", "above"),
            make_pair(3, 1, "left", "below"),
            make_pair(3, 2, "right", "below"),
        ]
        test = [make_pair(4, 3, random.choice(("left", "right")), random.choice(("above", "below")))]
        return taskvars, {"train": train, "test": test}

    def create_input(self, taskvars, gridvars):
        rows, cols = gridvars["rows"], gridvars["cols"]
        length = gridvars["fragment_length"]
        spacing = gridvars["minimum_spacing"]

        def sample_fragments():
            horizontal_row = random.randrange(rows)
            horizontal_start = random.randint(0, cols - length)
            vertical_col = random.randrange(cols)
            vertical_start = random.randint(0, rows - length)
            return horizontal_row, horizontal_start, vertical_col, vertical_start

        def separated(values):
            horizontal_row, horizontal_start, vertical_col, vertical_start = values
            horizontal_cols = range(horizontal_start, horizontal_start + length)
            vertical_rows = range(vertical_start, vertical_start + length)
            separated_fragments = (
                all(abs(vertical_col - col) >= spacing for col in horizontal_cols)
                and all(abs(horizontal_row - row) >= spacing for row in vertical_rows)
            )
            if gridvars["horizontal_relation"] == "left":
                horizontal_relation = vertical_col < horizontal_start
            else:
                horizontal_relation = vertical_col >= horizontal_start + length
            if gridvars["vertical_relation"] == "above":
                vertical_relation = horizontal_row < vertical_start
            else:
                vertical_relation = horizontal_row >= vertical_start + length
            return separated_fragments and horizontal_relation and vertical_relation

        def sample_natural_fragments():
            try:
                return retry(
                    sample_fragments,
                    separated,
                    max_attempts=100,
                )
            except ValueError:
                if gridvars["horizontal_relation"] == "left":
                    horizontal_start, vertical_col = spacing, 0
                else:
                    horizontal_start, vertical_col = 0, length - 1 + spacing
                if gridvars["vertical_relation"] == "above":
                    horizontal_row, vertical_start = 0, spacing
                else:
                    vertical_start, horizontal_row = 0, length - 1 + spacing
                return horizontal_row, horizontal_start, vertical_col, vertical_start

        fragments = tuple(
            int(value)
            for value in gridvars.get("fragments", sample_natural_fragments())
        )
        if len(fragments) != 4 or not separated(fragments):
            raise ValueError("fragments do not satisfy the requested relations")
        horizontal_row, horizontal_start, vertical_col, vertical_start = fragments
        if not (
            0 <= horizontal_row < rows
            and 0 <= horizontal_start <= cols - length
            and 0 <= vertical_col < cols
            and 0 <= vertical_start <= rows - length
        ):
            raise ValueError("fragments do not fit the requested grid")
        grid = np.zeros((rows, cols), dtype=int)
        grid[
            horizontal_row,
            horizontal_start : horizontal_start + length,
        ] = taskvars["horizontal_color"]
        grid[
            vertical_start : vertical_start + length,
            vertical_col,
        ] = taskvars["vertical_color"]
        return grid

    def transform_input(self, grid, taskvars):
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=True,
        )
        horizontal_row = None
        vertical_col = None
        for obj in objects:
            if taskvars["horizontal_color"] in obj.colors and obj.width >= obj.height:
                horizontal_row = int(next(iter(obj.cells))[0])
            if taskvars["vertical_color"] in obj.colors and obj.height >= obj.width:
                vertical_col = int(next(iter(obj.cells))[1])
        if horizontal_row is None or vertical_col is None:
            return grid.copy()
        output = grid.copy()
        output[horizontal_row, :] = taskvars["horizontal_color"]
        output[:, vertical_col] = taskvars["vertical_color"]
        output[horizontal_row, vertical_col] = taskvars["crossing_color"]
        return output
