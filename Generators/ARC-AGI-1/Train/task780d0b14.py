from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task780d0b14Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is divided by complete {vars['divider_width']}-cell-wide {color('background_color')} rows and columns into a rectangular array of panels.",
            "2. Every panel contains cells of one non-{color('background_color')} panel color scattered over {color('background_color')} holes.",
            "3. The divider lines are the only complete {color('background_color')} rows and columns, so their intersections determine all panel boundaries.",
            "4. Panel-row heights, panel-column widths, and the number of panel rows and columns vary between examples.",
            "5. Panel colors may repeat, while their identities, fill densities, and hole positions vary independently.",
        ]
        transformation_reasoning_chain = [
            "1. Locate the complete {color('background_color')} divider rows and columns and split the grid into its ordered panel rectangles.",
            "2. Within each panel, find all non-{color('background_color')} connected objects while ignoring the scattered holes.",
            "3. Recover the single panel color shared by those objects and represent that entire panel with one cell.",
            "4. Preserve panel order to return the compressed matrix of colors, with one output row and column per panel row and column.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            "background_color": 0,
            "divider_width": 1,
        }
        train_specs = [
            {
                "panel_rows": 2,
                "panel_cols": 2,
                "density_range": (0.55, 0.68),
            },
            {
                "panel_rows": 2,
                "panel_cols": 3,
                "density_range": (0.64, 0.78),
            },
            {
                "panel_rows": 3,
                "panel_cols": 2,
                "density_range": (0.72, 0.88),
            },
            {
                "panel_rows": 3,
                "panel_cols": 3,
                "density_range": (0.58, 0.84),
            },
        ]
        test_spec = {
            "panel_rows": 4,
            "panel_cols": 3,
            "density_range": (0.60, 0.88),
        }

        def make_pair(gridvars: dict) -> GridPair:
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [make_pair(spec) for spec in train_specs]
        return taskvars, {"train": train, "test": [make_pair(test_spec)]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        background_color = taskvars["background_color"]
        divider_width = taskvars["divider_width"]
        panel_rows = gridvars["panel_rows"]
        panel_cols = gridvars["panel_cols"]
        density_low, density_high = gridvars["density_range"]
        row_heights = [
            int(value)
            for value in gridvars.get(
                "row_heights",
                [random.randint(3, 6) for _ in range(panel_rows)],
            )
        ]
        col_widths = [
            int(value)
            for value in gridvars.get(
                "col_widths",
                [random.randint(3, 6) for _ in range(panel_cols)],
            )
        ]
        total_rows = sum(row_heights) + divider_width * (panel_rows - 1)
        total_cols = sum(col_widths) + divider_width * (panel_cols - 1)
        grid = np.full(
            (total_rows, total_cols),
            background_color,
            dtype=int,
        )

        palette = [
            color for color in range(1, 10) if color != background_color
        ]
        panel_colors = [
            int(value)
            for value in gridvars.get(
                "panel_colors",
                [random.choice(palette) for _ in range(panel_rows * panel_cols)],
            )
        ]
        if len(set(panel_colors)) == 1:
            panel_colors[-1] = gridvars.get(
                "replacement_panel_color",
                random.choice(
                    [color for color in palette if color != panel_colors[0]]
                ),
            )

        row_start = 0
        color_index = 0
        for row_height in row_heights:
            col_start = 0
            for col_width in col_widths:
                panel_color = panel_colors[color_index]
                density = gridvars.get(
                    f"panel_{color_index}_density",
                    random.uniform(density_low, density_high),
                )

                def sample_panel() -> np.ndarray:
                    candidate = np.full(
                        (row_height, col_width),
                        background_color,
                        dtype=int,
                    )
                    return np.asarray(
                        gridvars.get(
                            f"panel_{color_index}",
                            random_cell_coloring(
                                candidate,
                                panel_color,
                                density=density,
                                background=background_color,
                            ),
                        ),
                        dtype=int,
                    )

                def panel_is_unambiguous(candidate: np.ndarray) -> bool:
                    occupied = candidate != background_color
                    return bool(
                        np.all(np.any(occupied, axis=0))
                        and np.all(np.any(occupied, axis=1))
                        and np.any(candidate == background_color)
                    )

                try:
                    panel = retry(
                        sample_panel,
                        panel_is_unambiguous,
                        max_attempts=60,
                    )
                except ValueError:
                    panel = np.full(
                        (row_height, col_width),
                        panel_color,
                        dtype=int,
                    )
                    panel[0, 0] = background_color

                grid[
                    row_start : row_start + row_height,
                    col_start : col_start + col_width,
                ] = panel
                col_start += col_width + divider_width
                color_index += 1
            row_start += row_height + divider_width
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background_color = taskvars["background_color"]
        divider_rows = set(
            int(row) for row in np.where(np.all(grid == background_color, axis=1))[0]
        )
        divider_cols = set(
            int(col) for col in np.where(np.all(grid == background_color, axis=0))[0]
        )

        row_spans = []
        start = 0
        for row in range(grid.shape[0] + 1):
            if row == grid.shape[0] or row in divider_rows:
                if start < row:
                    row_spans.append((start, row))
                start = row + 1

        col_spans = []
        start = 0
        for col in range(grid.shape[1] + 1):
            if col == grid.shape[1] or col in divider_cols:
                if start < col:
                    col_spans.append((start, col))
                start = col + 1

        output = np.full(
            (len(row_spans), len(col_spans)),
            background_color,
            dtype=int,
        )
        for panel_row, (row_start, row_stop) in enumerate(row_spans):
            for panel_col, (col_start, col_stop) in enumerate(col_spans):
                panel = grid[row_start:row_stop, col_start:col_stop]
                objects = find_connected_objects(
                    panel,
                    diagonal_connectivity=False,
                    background=background_color,
                    monochromatic=True,
                )
                color_counts = {}
                for obj in objects:
                    for _, _, color in obj.cells:
                        color_counts[int(color)] = color_counts.get(int(color), 0) + 1
                output[panel_row, panel_col] = max(
                    color_counts,
                    key=color_counts.get,
                )
        return output
