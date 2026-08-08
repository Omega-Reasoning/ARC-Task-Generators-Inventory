from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects
from typing import Dict, Any, Tuple
import numpy as np
import random


class Taskcdecee7fGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a sparse {color('background_color')} grid containing at most {vars['output_size']} squared isolated colored points.",
            "2. No two colored points share an input column, while their row coordinates are otherwise irregular.",
            "3. Point colors may repeat, so horizontal position rather than color determines their ordering.",
            "4. The number of points varies and may leave part of the compact output unused.",
        ]
        transformation_reasoning_chain = [
            "1. Collect all non-{color('background_color')} points and sort them by input column from left to right.",
            "2. Pack successive groups of {vars['output_size']} colors into a {vars['output_size']}x{vars['output_size']} output.",
            "3. Fill the first output row {vars['first_row_direction']} and reverse the direction on every following row.",
            "4. Leave output positions after the final point {color('background_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        background_color = taskvars['background_color']
        rows = gridvars['rows']
        cols = gridvars['cols']
        point_count = gridvars['point_count']
        palette_size = gridvars.get(
            'palette_size',
            random.randint(2, min(4, point_count - 1)),
        )
        palette = gridvars.get(
            'palette',
            random.sample(
                [
                    color
                    for color in range(1, 10)
                    if color != background_color
                ],
                palette_size,
            ),
        )

        def sample_positions():
            columns = gridvars.get(
                'columns',
                random.sample(range(cols), point_count),
            )
            point_rows = gridvars.get(
                'point_rows',
                [random.randrange(rows) for _ in columns],
            )
            return list(zip(point_rows, columns))

        try:
            positions = retry(
                sample_positions,
                lambda value: all(
                    abs(row_a - row_b) + abs(col_a - col_b) != 1
                    for index, (row_a, col_a) in enumerate(value)
                    for row_b, col_b in value[index + 1:]
                ),
                max_attempts=50,
            )
        except ValueError:
            positions = [((2 * index) % rows, index) for index in range(point_count)]

        point_colors = gridvars.get(
            'point_colors',
            [random.choice(palette) for _ in positions],
        )
        cells = {
            (row, col, color)
            for (row, col), color in zip(positions, point_colors)
        }
        grid = np.full((rows, cols), background_color, dtype=int)
        GridObject(cells).paste(grid, overwrite=True, background=background_color)
        return grid

    def transform_input(self, grid, taskvars):
        background_color = taskvars['background_color']
        output_size = taskvars['output_size']
        first_row_direction = taskvars['first_row_direction']
        points = []
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background_color,
            monochromatic=True,
        )
        for obj in objects:
            for row, col, color in obj.cells:
                points.append((int(col), int(row), int(color)))
        points.sort(key=lambda item: (item[0], item[1]))

        output = np.full((output_size, output_size), background_color, dtype=int)
        for index, (_, _, color) in enumerate(points[:output_size * output_size]):
            output_row = index // output_size
            offset = index % output_size
            forward = output_row % 2 == 0
            if first_row_direction != 'left_to_right':
                forward = not forward
            output_col = offset if forward else output_size - 1 - offset
            output[output_row, output_col] = color
        return output

    def create_grids(self):
        taskvars = {
            'background_color': 0,
            'output_size': 3,
            'first_row_direction': 'left_to_right',
        }
        train_gridvars = [
            {'rows': 8, 'cols': 10, 'point_count': 5},
            {'rows': 10, 'cols': 11, 'point_count': 6},
            {'rows': 11, 'cols': 12, 'point_count': 7},
            {'rows': 12, 'cols': 13, 'point_count': 8},
        ]
        test_gridvars = [{'rows': 14, 'cols': 14, 'point_count': 9}]
        train = []
        test = []
        for gridvars in train_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train.append({'input': input_grid, 'output': output_grid})
        for gridvars in test_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            test.append({'input': input_grid, 'output': output_grid})
        return taskvars, {'train': train, 'test': test}
