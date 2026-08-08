from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import retry


class Task941d9a10Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a square grid partitioned by complete {color('divider_color')} rows and columns.",
            "2. The dividers create odd numbers of horizontal and vertical empty compartments, so each axis has a unique middle compartment.",
            "3. Compartment heights and widths may be unequal, but every non-divider cell begins {color('background_color')}.",
            "4. The numbers and locations of dividers vary between examples while the four color roles are shared.",
        ]
        transformation_reasoning_chain = [
            "1. Find the full {color('divider_color')} rows and columns and enumerate the compartments between them.",
            "2. Fill the upper-left compartment {color('first_color')}.",
            "3. Fill the unique middle row-band and middle column-band compartment {color('middle_color')}.",
            "4. Fill the lower-right compartment {color('last_color')} and preserve all dividers and other empty compartments.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        divider_color, first_color, middle_color, last_color = random.sample(range(1, 10), 4)
        taskvars = {
            "background_color": 0,
            "divider_color": divider_color,
            "first_color": first_color,
            "middle_color": middle_color,
            "last_color": last_color,
        }
        layouts = [
            {"row_bands": 3, "col_bands": 3},
            {"row_bands": 5, "col_bands": 3},
            {"row_bands": 3, "col_bands": 5},
            {"row_bands": 3, "col_bands": 3},
        ]
        test_layout = {"row_bands": 5, "col_bands": 5}

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {"input": input_grid, "output": self.transform_input(input_grid, taskvars)}

        return taskvars, {
            "train": [make_pair(layout) for layout in layouts],
            "test": [make_pair(test_layout)],
        }

    def create_input(self, taskvars, gridvars):
        size = 10

        def sample_dividers(number_of_bands):
            return sorted(random.sample(range(1, size - 1), number_of_bands - 1))

        def valid(dividers):
            boundaries = [-1] + dividers + [size]
            return all(boundaries[index + 1] - boundaries[index] >= 2 for index in range(len(boundaries) - 1))

        def sample_valid_dividers(number_of_bands):
            try:
                return retry(
                    lambda: sample_dividers(number_of_bands),
                    valid,
                    max_attempts=80,
                )
            except ValueError:
                return [1, 3, 5, 7] if number_of_bands == 5 else [3, 6]

        row_dividers = list(gridvars.get(
            "row_dividers",
            sample_valid_dividers(gridvars["row_bands"]),
        ))
        col_dividers = list(gridvars.get(
            "col_dividers",
            sample_valid_dividers(gridvars["col_bands"]),
        ))
        for axis, dividers, number_of_bands in (
            ("row", row_dividers, gridvars["row_bands"]),
            ("column", col_dividers, gridvars["col_bands"]),
        ):
            if (
                len(dividers) != number_of_bands - 1
                or dividers != sorted(set(dividers))
                or any(not isinstance(value, int) for value in dividers)
                or any(value not in range(1, size - 1) for value in dividers)
                or not valid(dividers)
            ):
                raise ValueError(
                    f"{axis}_dividers must define the requested number of nonempty bands"
                )
        grid = np.full((size, size), taskvars["background_color"], dtype=int)
        grid[row_dividers, :] = taskvars["divider_color"]
        grid[:, col_dividers] = taskvars["divider_color"]
        return grid

    def transform_input(self, grid, taskvars):
        divider = taskvars["divider_color"]
        output = grid.copy()
        divider_rows = [
            row for row in range(grid.shape[0]) if np.all(grid[row, :] == divider)
        ]
        divider_cols = [
            col for col in range(grid.shape[1]) if np.all(grid[:, col] == divider)
        ]
        row_starts = [0] + [row + 1 for row in divider_rows]
        row_stops = divider_rows + [grid.shape[0]]
        col_starts = [0] + [col + 1 for col in divider_cols]
        col_stops = divider_cols + [grid.shape[1]]
        row_bands = list(zip(row_starts, row_stops))
        col_bands = list(zip(col_starts, col_stops))
        selected = [
            (0, 0, taskvars["first_color"]),
            (len(row_bands) // 2, len(col_bands) // 2, taskvars["middle_color"]),
            (len(row_bands) - 1, len(col_bands) - 1, taskvars["last_color"]),
        ]
        for row_index, col_index, color in selected:
            row_start, row_stop = row_bands[row_index]
            col_start, col_stop = col_bands[col_index]
            output[row_start:row_stop, col_start:col_stop] = color
        return output
