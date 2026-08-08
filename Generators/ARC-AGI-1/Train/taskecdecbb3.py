from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects


class Taskecdecbb3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each {color('background_color')} input contains one or more parallel full-length {color('divider_color')} lines.",
            "2. The lines are all horizontal or all vertical within an example, but orientation varies across examples.",
            "3. One or more isolated {color('marker_color')} cells lie between dividers or outside their outer span.",
            "4. Grid dimensions and the counts, spacings, and placements of lines and markers vary.",
            "5. Intersection emphasis has neighborhood radius {vars['intersection_radius']}.",
        ]
        transformation_reasoning_chain = [
            "1. Detect the orientation and ordered positions of the full-length {color('divider_color')} lines.",
            "2. For each {color('marker_color')} cell between two lines, draw a perpendicular marker-colored segment between its two bracketing dividers.",
            "3. For a marker outside the line span, draw only from that marker to its nearest divider.",
            "4. At every crossed divider, preserve the {color('marker_color')} center and fill its radius-{vars['intersection_radius']} eight-neighbor ring with {color('divider_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        background, divider, marker = 0, *random.sample(range(1, 10), 2)
        taskvars = {
            "background_color": background,
            "divider_color": divider,
            "marker_color": marker,
            "intersection_radius": 1,
        }
        train_gridvars = [
            {
                "rows": 13,
                "cols": 13,
                "orientation": "horizontal",
                "lines": [6],
                "markers": [[2, 2], [10, 8]],
            },
            {
                "rows": 13,
                "cols": 18,
                "orientation": "vertical",
                "lines": [3, 14],
                "markers": [[4, 8]],
            },
            {
                "rows": 17,
                "cols": 12,
                "orientation": "horizontal",
                "lines": [7, 13],
                "markers": [[2, 8], [10, 4]],
            },
            {
                "rows": 15,
                "cols": 16,
                "orientation": "vertical",
                "lines": [4, 10],
                "markers": [[2, 1], [7, 7], [12, 13]],
            },
        ]
        test_gridvars = {
            "rows": 19,
            "cols": 18,
            "orientation": "horizontal",
            "lines": [4, 10, 15],
            "markers": [[1, 2], [7, 7], [12, 13], [17, 4]],
        }

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
        background = taskvars["background_color"]
        divider = taskvars["divider_color"]
        marker = taskvars["marker_color"]
        rows, cols = gridvars["rows"], gridvars["cols"]
        orientation = gridvars["orientation"]
        lines = gridvars["lines"]
        marker_bases = gridvars["markers"]

        def build_scene():
            grid = np.full((rows, cols), background, dtype=int)
            if orientation == "horizontal":
                for row in lines:
                    grid[row, :] = divider
            else:
                for col in lines:
                    grid[:, col] = divider
            for index, (base_row, base_col) in enumerate(marker_bases):
                offset = gridvars.get(
                    f"marker_{index}_offset",
                    random.randint(-1, 1),
                )
                if orientation == "horizontal":
                    row = base_row
                    col = min(cols - 2, max(1, base_col + offset))
                else:
                    row = min(rows - 2, max(1, base_row + offset))
                    col = base_col
                grid[row, col] = marker
            return grid

        def valid_scene(grid):
            objects = find_connected_objects(
                grid,
                diagonal_connectivity=True,
                background=background,
                monochromatic=True,
            )
            marker_objects = [obj for obj in objects if marker in obj.colors]
            marker_cells = int(np.sum(grid == marker))
            return (
                marker_cells == len(marker_bases)
                and len(marker_objects) == len(marker_bases)
                and all(len(obj) == 1 for obj in marker_objects)
            )

        return retry(build_scene, valid_scene, max_attempts=40)

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        divider = taskvars["divider_color"]
        marker = taskvars["marker_color"]
        radius = taskvars["intersection_radius"]
        rows, cols = grid.shape
        horizontal_lines = [
            row for row in range(rows) if np.all(grid[row, :] == divider)
        ]
        vertical_lines = [
            col for col in range(cols) if np.all(grid[:, col] == divider)
        ]
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        )
        marker_cells = sorted(
            (row, col)
            for obj in objects
            if marker in obj.colors
            for row, col, _ in obj
        )
        output = grid.copy()
        intersections = []

        if horizontal_lines:
            for row, col in marker_cells:
                above = [line for line in horizontal_lines if line < row]
                below = [line for line in horizontal_lines if line > row]
                start = max(above) if above else row
                end = min(below) if below else row
                if not above:
                    end = min(below)
                if not below:
                    start = max(above)
                for current_row in range(start, end + 1):
                    output[current_row, col] = marker
                intersections.extend(
                    (line, col)
                    for line in horizontal_lines
                    if start <= line <= end
                )
        elif vertical_lines:
            for row, col in marker_cells:
                left = [line for line in vertical_lines if line < col]
                right = [line for line in vertical_lines if line > col]
                start = max(left) if left else col
                end = min(right) if right else col
                if not left:
                    end = min(right)
                if not right:
                    start = max(left)
                for current_col in range(start, end + 1):
                    output[row, current_col] = marker
                intersections.extend(
                    (row, line)
                    for line in vertical_lines
                    if start <= line <= end
                )
        else:
            return output

        for row, col in intersections:
            for row_offset in range(-radius, radius + 1):
                for col_offset in range(-radius, radius + 1):
                    if row_offset == 0 and col_offset == 0:
                        continue
                    neighbor_row = row + row_offset
                    neighbor_col = col + col_offset
                    if 0 <= neighbor_row < rows and 0 <= neighbor_col < cols:
                        output[neighbor_row, neighbor_col] = divider
        for row, col in intersections:
            output[row, col] = marker
        return output
