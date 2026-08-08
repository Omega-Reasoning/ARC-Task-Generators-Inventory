from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import parse_objects_by_color
import numpy as np
import random


class Task1f876c06Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input contains several colors on an otherwise empty grid.",
            "2. Each foreground color appears in exactly {vars['endpoint_count']} isolated endpoint cells.",
            "3. The two cells of each color lie on one 45-degree diagonal, so their row and column separations are equal.",
            "4. Pair colors, diagonal slopes, segment lengths, and locations vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Group the nonempty cells by color and pair the {vars['endpoint_count']} endpoints of each color.",
            "2. Determine the unit diagonal step connecting each same-colored endpoint pair.",
            "3. Fill every cell on the inclusive diagonal segment between the endpoints with their color.",
            "4. Combine all completed colored segments on an otherwise empty grid of the same dimensions.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = gridvars['rows']
        cols = gridvars['cols']
        segment_count = gridvars['segment_count']
        grid = np.zeros((rows, cols), dtype=int)
        occupied = set()
        colors = gridvars.get(
            'colors', random.sample(range(1, 10), segment_count)
        )

        for segment_index, color in enumerate(colors):
            def sample_segment():
                length = gridvars.get(
                    f'segment_{segment_index}_length',
                    random.randint(2, min(6, rows - 1, cols - 1)),
                )
                delta_row = gridvars.get(
                    f'segment_{segment_index}_delta_row',
                    random.choice([-1, 1]),
                )
                delta_col = gridvars.get(
                    f'segment_{segment_index}_delta_col',
                    random.choice([-1, 1]),
                )
                row = gridvars.get(
                    f'segment_{segment_index}_row',
                    random.randint(length, rows - 1)
                    if delta_row < 0
                    else random.randint(0, rows - 1 - length),
                )
                col = gridvars.get(
                    f'segment_{segment_index}_col',
                    random.randint(length, cols - 1)
                    if delta_col < 0
                    else random.randint(0, cols - 1 - length),
                )
                path = [
                    (row + step * delta_row, col + step * delta_col)
                    for step in range(length + 1)
                ]
                return row, col, delta_row, delta_col, length, path

            segment = retry(
                sample_segment,
                lambda value: not any(cell in occupied for cell in value[5]),
            )
            row, col, delta_row, delta_col, length, path = segment
            occupied.update(path)
            grid[row, col] = color
            grid[row + length * delta_row, col + length * delta_col] = color
        return grid

    def transform_input(self, grid, taskvars):
        endpoint_count = taskvars['endpoint_count']
        output = np.array(grid, copy=True)
        color_groups = parse_objects_by_color(grid, background=0)
        for group in color_groups:
            coords = sorted(group.coords)
            if len(coords) != endpoint_count:
                continue
            (row0, col0), (row1, col1) = coords
            row_distance = row1 - row0
            col_distance = col1 - col0
            if abs(row_distance) != abs(col_distance) or row_distance == 0:
                continue
            delta_row = 1 if row_distance > 0 else -1
            delta_col = 1 if col_distance > 0 else -1
            color = int(grid[row0, col0])
            for step in range(abs(row_distance) + 1):
                output[row0 + step * delta_row, col0 + step * delta_col] = color
        return output

    def create_grids(self):
        taskvars = {'endpoint_count': 2}
        train = []
        for segment_count in (2, 3, 4, 5):
            gridvars = {
                'rows': random.randint(12, 18),
                'cols': random.randint(12, 18),
                'segment_count': segment_count,
            }
            input_grid = self.create_input(taskvars, gridvars)
            train.append(GridPair(input=input_grid, output=self.transform_input(input_grid, taskvars)))
        gridvars = {
            'rows': random.randint(16, 22),
            'cols': random.randint(16, 22),
            'segment_count': 6,
        }
        test_input = self.create_input(taskvars, gridvars)
        test = [GridPair(input=test_input, output=self.transform_input(test_input, taskvars))]
        return taskvars, TrainTestData(train=train, test=test)
