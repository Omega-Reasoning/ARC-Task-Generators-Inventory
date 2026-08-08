from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.transformation_library import parse_objects_by_color
import numpy as np
import random


class Task2dc579daGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a square grid partitioned into {vars['panel_count']} equal square panels by one horizontal and one vertical separator of width {vars['separator_width']}.",
            "2. Every panel is otherwise filled with the same example-specific color.",
            "3. Exactly one panel contains one cell of a rare color different from both the panel fill and separator colors.",
            "4. Panel size, colors, the rare cell position, and the panel containing it vary between examples.",
            "5. The separator crosses the full grid and leaves all four panels the same size.",
        ]
        transformation_reasoning_chain = [
            "1. Locate the full separator row and column of width {vars['separator_width']} and split the input into {vars['panel_count']} equal panels.",
            "2. Identify the rare cell whose color differs from the common panel-fill color and the separator color.",
            "3. Select the unique panel that contains that rare cell.",
            "4. Return that entire panel unchanged as the output grid.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        separator_width = random.choice([1, 2])
        taskvars = {
            "separator_width": separator_width,
            "panel_count": 4,
        }
        quadrants = [0, 1, 2, 3]
        random.shuffle(quadrants)
        train = []
        for index, quadrant in enumerate(quadrants):
            panel_size = 2 + index
            colors = random.sample(range(1, 10), 3)
            gridvars = {
                "panel_size": panel_size,
                "quadrant": quadrant,
                "panel_color": colors[0],
                "separator_color": colors[1],
                "rare_color": colors[2],
                "rare_row": random.randrange(panel_size),
                "rare_col": random.randrange(panel_size),
            }
            input_grid = self.create_input(taskvars, gridvars)
            train.append({
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            })

        test_size = random.choice([6, 7])
        test_colors = random.sample(range(1, 10), 3)
        test_vars = {
            "panel_size": test_size,
            "quadrant": random.randrange(4),
            "panel_color": test_colors[0],
            "separator_color": test_colors[1],
            "rare_color": test_colors[2],
            "rare_row": random.randrange(test_size),
            "rare_col": random.randrange(test_size),
        }
        test_input = self.create_input(taskvars, test_vars)
        return taskvars, {
            "train": train,
            "test": [{
                "input": test_input,
                "output": self.transform_input(test_input, taskvars),
            }],
        }

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        panel_size = gridvars["panel_size"]
        separator_width = taskvars["separator_width"]
        side = 2 * panel_size + separator_width
        grid = np.full(
            (side, side),
            gridvars["panel_color"],
            dtype=int,
        )
        grid[
            panel_size:panel_size + separator_width,
            :,
        ] = gridvars["separator_color"]
        grid[
            :,
            panel_size:panel_size + separator_width,
        ] = gridvars["separator_color"]

        quadrant = gridvars["quadrant"]
        row_offset = 0 if quadrant < 2 else panel_size + separator_width
        col_offset = 0 if quadrant % 2 == 0 else panel_size + separator_width
        grid[
            row_offset + gridvars["rare_row"],
            col_offset + gridvars["rare_col"],
        ] = gridvars["rare_color"]
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        separator_width = taskvars["separator_width"]
        colors = [int(color) for color in np.unique(grid)]

        separator_color = None
        separator_rows = None
        separator_cols = None
        for color in colors:
            full_rows = np.where(np.all(grid == color, axis=1))[0]
            full_cols = np.where(np.all(grid == color, axis=0))[0]
            if len(full_rows) == separator_width and len(full_cols) == separator_width:
                separator_color = color
                separator_rows = full_rows
                separator_cols = full_cols
                break

        if separator_color is None:
            return grid.copy()

        counts = {
            color: int(np.count_nonzero(grid == color))
            for color in colors
            if color != separator_color
        }
        panel_color = max(counts, key=counts.get)
        objects = parse_objects_by_color(grid, background=panel_color)
        rare_objects = [
            obj for obj in objects
            if separator_color not in obj.colors
        ]
        rare_object = min(rare_objects, key=lambda obj: obj.size)
        rare_row, rare_col = next(iter(rare_object.coords))

        row_ranges = [
            (0, int(separator_rows[0])),
            (int(separator_rows[-1]) + 1, grid.shape[0]),
        ]
        col_ranges = [
            (0, int(separator_cols[0])),
            (int(separator_cols[-1]) + 1, grid.shape[1]),
        ]
        for row_start, row_stop in row_ranges:
            for col_start, col_stop in col_ranges:
                if (
                    row_start <= rare_row < row_stop
                    and col_start <= rare_col < col_stop
                ):
                    return grid[row_start:row_stop, col_start:col_stop].copy()
        return grid.copy()
