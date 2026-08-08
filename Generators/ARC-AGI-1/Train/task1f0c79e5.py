from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task1f0c79e5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is an otherwise empty grid containing one compact 2x2 clue.",
            "2. Every clue cell is either the fixed direction-marker color {color('marker_color')} or one other paint color.",
            "3. The positions of the {color('marker_color')} cells within the 2x2 clue select one or more outward diagonal directions.",
            "4. The clue location, paint color, and selected nonempty subset of directions vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Locate the 2x2 clue and identify its paint color as the non-{color('marker_color')} foreground color.",
            "2. Read every {color('marker_color')} corner as the corresponding outward diagonal direction from the clue.",
            "3. Recolor all four cells of the clue with the paint color.",
            "4. In every selected direction, repeat the filled 2x2 square one diagonal step at a time until the grid boundary clips further copies.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = gridvars['rows']
        cols = gridvars['cols']
        directions = list(gridvars['directions'])
        paint_color = gridvars['paint_color']

        def sample_position():
            return (
                gridvars.get('sampled_row', random.randint(2, rows - 4)),
                gridvars.get('sampled_col', random.randint(2, cols - 4)),
            )

        if 'position' in gridvars:
            row0, col0 = gridvars['position']
        else:
            row0, col0 = retry(
                sample_position,
                lambda pos: 1 < pos[0] < rows - 3 and 1 < pos[1] < cols - 3,
            )

        grid = np.zeros((rows, cols), dtype=int)
        grid[row0:row0 + 2, col0:col0 + 2] = paint_color
        corners = [
            (row0, col0),
            (row0, col0 + 1),
            (row0 + 1, col0 + 1),
            (row0 + 1, col0),
        ]
        for index in directions:
            row, col = corners[index]
            grid[row, col] = taskvars['marker_color']
        return grid

    def transform_input(self, grid, taskvars):
        marker_color = taskvars['marker_color']
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=False,
        )
        if len(objects) == 0:
            return np.array(grid, copy=True)

        clue = max(objects, key=lambda obj: obj.size)
        row_slice, col_slice = clue.bounding_box
        row0, col0 = row_slice.start, col_slice.start
        row1, col1 = row_slice.stop - 1, col_slice.stop - 1
        paint_values = [
            int(grid[row, col])
            for row in range(row0, row1 + 1)
            for col in range(col0, col1 + 1)
            if int(grid[row, col]) not in (0, marker_color)
        ]
        if not paint_values:
            return np.array(grid, copy=True)
        paint_color = paint_values[0]

        directions = []
        for row in range(row0, row1 + 1):
            for col in range(col0, col1 + 1):
                if int(grid[row, col]) == marker_color:
                    delta_row = -1 if row == row0 else 1
                    delta_col = -1 if col == col0 else 1
                    directions.append((delta_row, delta_col))

        output = np.zeros_like(grid)
        rows, cols = grid.shape
        for delta_row, delta_col in directions:
            for step in range(max(rows, cols) + 2):
                top = row0 + step * delta_row
                left = col0 + step * delta_col
                if top >= rows or left >= cols or top + 1 < 0 or left + 1 < 0:
                    break
                for row in range(top, top + 2):
                    for col in range(left, left + 2):
                        if 0 <= row < rows and 0 <= col < cols:
                            output[row, col] = paint_color
        return output

    def create_grids(self):
        marker_color = random.randint(1, 9)
        taskvars = {'marker_color': marker_color}
        base = random.randrange(4)
        direction_sets = [
            (base,),
            ((base + 2) % 4,),
            (base, (base + 1) % 4),
            (base, (base + 1) % 4, (base + 2) % 4),
            (base, (base + 2) % 4),
        ]
        train = []
        test = []
        for index, directions in enumerate(direction_sets):
            rows = random.randint(8, 14)
            cols = random.randint(8, 14)
            paint_color = random.choice(
                [color for color in range(1, 10) if color != marker_color]
            )
            gridvars = {
                'rows': rows,
                'cols': cols,
                'directions': directions,
                'paint_color': paint_color,
            }
            input_grid = self.create_input(taskvars, gridvars)
            pair = GridPair(
                input=input_grid,
                output=self.transform_input(input_grid, taskvars),
            )
            if index < 4:
                train.append(pair)
            else:
                test.append(pair)
        return taskvars, TrainTestData(train=train, test=test)
