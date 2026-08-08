from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Taskaf902bf9Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is an empty grid containing one or more groups of four {color('corner_color')} cells.",
            "2. Within each group, the four cells occupy the corners of an axis-aligned rectangle.",
            "3. Rectangle interiors are empty and different rectangles do not create ambiguous extra corner quartets.",
            "4. Rectangle count, position, height, and width vary between examples."
        ]
        transformation_reasoning_chain = [
            "1. Find row and column pairs whose four intersections are all {color('corner_color')}.",
            "2. Treat each such quartet as the corners of a rectangle.",
            "3. Fill every cell strictly inside each rectangle with {color('fill_color')}.",
            "4. Preserve all {color('corner_color')} cells and all background outside rectangle interiors."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        corner_color, fill_color = random.sample(range(1, 10), 2)
        taskvars = {'corner_color': corner_color, 'fill_color': fill_color}
        train_gridvars = [
            {'rows': 10, 'cols': 11, 'sizes': [(3, 3)]},
            {'rows': 12, 'cols': 12, 'sizes': [(4, 6)]},
            {'rows': 14, 'cols': 15, 'sizes': [(3, 5), (5, 3)]},
            {'rows': 16, 'cols': 14, 'sizes': [(6, 4), (3, 6)]},
        ]
        test_gridvars = {'rows': 18, 'cols': 17, 'sizes': [(5, 7), (4, 4)]}
        train = []
        for gridvars in train_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        test_input = self.create_input(taskvars, test_gridvars)
        return taskvars, {
            'train': train,
            'test': [{'input': test_input, 'output': self.transform_input(test_input, taskvars)}],
        }

    def create_input(self, taskvars, gridvars):
        rows, cols = gridvars['rows'], gridvars['cols']
        corner_color = taskvars['corner_color']

        def sample_grid():
            candidate = np.zeros((rows, cols), dtype=int)
            used_rows = set()
            used_cols = set()
            boxes = []
            for height, width in gridvars['sizes']:
                placed = False
                for _ in range(80):
                    row0 = random.randint(0, rows - height)
                    col0 = random.randint(0, cols - width)
                    row1, col1 = row0 + height - 1, col0 + width - 1
                    separated = all(
                        row1 < old_row0 - 1
                        or old_row1 < row0 - 1
                        or col1 < old_col0 - 1
                        or old_col1 < col0 - 1
                        for old_row0, old_row1, old_col0, old_col1 in boxes
                    )
                    if row0 not in used_rows and row1 not in used_rows and col0 not in used_cols and col1 not in used_cols and separated:
                        for row, col in ((row0, col0), (row0, col1), (row1, col0), (row1, col1)):
                            candidate[row, col] = corner_color
                        used_rows.update((row0, row1))
                        used_cols.update((col0, col1))
                        boxes.append((row0, row1, col0, col1))
                        placed = True
                        break
                if not placed:
                    return None
            return candidate

        def sample_natural_grid():
            try:
                return retry(sample_grid, lambda value: value is not None, max_attempts=40)
            except ValueError:
                candidate = np.zeros((rows, cols), dtype=int)
                anchors = [(1, 1), (rows // 2 + 1, cols // 2 + 1)]
                for (height, width), (row0, col0) in zip(gridvars['sizes'], anchors):
                    row1, col1 = row0 + height - 1, col0 + width - 1
                    for row, col in ((row0, col0), (row0, col1), (row1, col0), (row1, col1)):
                        candidate[row, col] = corner_color
                return candidate

        corner_grid = np.asarray(
            gridvars.get("corner_grid", sample_natural_grid()),
            dtype=int,
        )
        if "corner_grid" in gridvars:
            if corner_grid.shape != (rows, cols) or not set(
                int(value) for value in np.unique(corner_grid)
            ).issubset({0, corner_color}):
                raise ValueError("corner_grid has the wrong shape or palette")
            points = set(
                (int(row), int(col))
                for row, col in np.argwhere(corner_grid == corner_color)
            )
            point_rows = sorted({row for row, _ in points})
            point_cols = sorted({col for _, col in points})
            rectangles = []
            for row_index, row0 in enumerate(point_rows):
                for row1 in point_rows[row_index + 1 :]:
                    for col_index, col0 in enumerate(point_cols):
                        for col1 in point_cols[col_index + 1 :]:
                            if {
                                (row0, col0),
                                (row0, col1),
                                (row1, col0),
                                (row1, col1),
                            }.issubset(points):
                                rectangles.append((row1 - row0 + 1, col1 - col0 + 1))
            if (
                len(points) != 4 * len(gridvars["sizes"])
                or sorted(rectangles) != sorted(tuple(size) for size in gridvars["sizes"])
            ):
                raise ValueError("corner_grid does not encode exactly the requested rectangles")
        return corner_grid.copy()

    def transform_input(self, grid, taskvars):
        corner_color = taskvars['corner_color']
        fill_color = taskvars['fill_color']
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=True,
        ).with_color(corner_color)
        points = set()
        for obj in objects:
            points.update(obj.coords)
        rows = sorted({row for row, _ in points})
        cols = sorted({col for _, col in points})
        output = grid.copy()
        for first_row_index in range(len(rows)):
            for second_row_index in range(first_row_index + 1, len(rows)):
                row0, row1 = rows[first_row_index], rows[second_row_index]
                for first_col_index in range(len(cols)):
                    for second_col_index in range(first_col_index + 1, len(cols)):
                        col0, col1 = cols[first_col_index], cols[second_col_index]
                        corners = {(row0, col0), (row0, col1), (row1, col0), (row1, col1)}
                        if corners.issubset(points):
                            output[row0 + 1:row1, col0 + 1:col1] = fill_color
        return output
