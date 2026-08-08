from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task2bcee788Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. A {vars['grid_size']} by {vars['grid_size']} empty grid contains one connected monochromatic payload object.",
            "2. Cells of {color('marker_color')} sit immediately outside one payload edge and copy the occupied cells along that boundary.",
            "3. The payload color, asymmetric shape, position, and marked side vary between examples.",
            "4. The marker can be above, below, left, or right of the payload and indicates the side on which a reflected copy is missing.",
        ]
        transformation_reasoning_chain = [
            "1. Separate the payload cells from the adjacent {color('marker_color')} boundary marker.",
            "2. Infer the horizontal or vertical reflection axis halfway between the payload edge and marker cells.",
            "3. Reflect every payload cell across that axis, using the payload color and replacing the marker where the copy overlaps it.",
            "4. Preserve the original and reflected payload, then replace every remaining cell with {color('fill_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        marker_color = taskvars['marker_color']
        payload_color = gridvars['payload_color']
        height = gridvars['height']
        width = gridvars['width']
        side = gridvars['side']

        shape = np.asarray(
            gridvars.get(
                'shape',
                retry(
                    lambda: create_object(
                        height,
                        width,
                        payload_color,
                        contiguity=Contiguity.FOUR,
                        background=0,
                    ),
                    lambda value: (
                        np.any(value[0, :] != 0)
                        and np.any(value[-1, :] != 0)
                        and np.any(value[:, 0] != 0)
                        and np.any(value[:, -1] != 0)
                        and np.count_nonzero(value) < height * width
                    ),
                ),
            ),
            dtype=int,
        )
        grid = np.zeros((grid_size, grid_size), dtype=int)
        if side == 'right':
            top = gridvars.get(
                'top', random.randint(1, grid_size - height - 1)
            )
            left = gridvars.get(
                'left', random.randint(1, grid_size - 2 * width)
            )
        elif side == 'left':
            top = gridvars.get(
                'top', random.randint(1, grid_size - height - 1)
            )
            left = gridvars.get(
                'left', random.randint(width, grid_size - width - 1)
            )
        elif side == 'bottom':
            top = gridvars.get(
                'top', random.randint(1, grid_size - 2 * height)
            )
            left = gridvars.get(
                'left', random.randint(1, grid_size - width - 1)
            )
        else:
            top = gridvars.get(
                'top', random.randint(height, grid_size - height - 1)
            )
            left = gridvars.get(
                'left', random.randint(1, grid_size - width - 1)
            )

        for row in range(height):
            for col in range(width):
                if shape[row, col] != 0:
                    grid[top + row, left + col] = payload_color
        if side == 'right':
            for row in range(height):
                if shape[row, width - 1] != 0:
                    grid[top + row, left + width] = marker_color
        elif side == 'left':
            for row in range(height):
                if shape[row, 0] != 0:
                    grid[top + row, left - 1] = marker_color
        elif side == 'bottom':
            for col in range(width):
                if shape[height - 1, col] != 0:
                    grid[top + height, left + col] = marker_color
        else:
            for col in range(width):
                if shape[0, col] != 0:
                    grid[top - 1, left + col] = marker_color
        return grid

    def transform_input(self, grid, taskvars):
        marker_color = taskvars['marker_color']
        fill_color = taskvars['fill_color']
        objects = list(find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=True,
        ))
        payload_objects = [
            obj for obj in objects if marker_color not in obj.colors
        ]
        marker_objects = [
            obj for obj in objects if marker_color in obj.colors
        ]
        if len(payload_objects) == 0 or len(marker_objects) == 0:
            return np.array(grid, copy=True)
        payload = max(payload_objects, key=lambda obj: obj.size)
        marker = max(marker_objects, key=lambda obj: obj.size)
        payload_cells = [
            (int(row), int(col), int(color))
            for row, col, color in payload.cells
        ]
        marker_rows = [int(row) for row, col, color in marker.cells]
        marker_cols = [int(col) for row, col, color in marker.cells]
        payload_rows = [row for row, col, color in payload_cells]
        payload_cols = [col for row, col, color in payload_cells]
        orientation = None
        axis_sum = None
        if max(marker_cols) < min(payload_cols):
            orientation = 'vertical'
            axis_sum = max(marker_cols) + min(payload_cols)
        elif min(marker_cols) > max(payload_cols):
            orientation = 'vertical'
            axis_sum = min(marker_cols) + max(payload_cols)
        elif max(marker_rows) < min(payload_rows):
            orientation = 'horizontal'
            axis_sum = max(marker_rows) + min(payload_rows)
        elif min(marker_rows) > max(payload_rows):
            orientation = 'horizontal'
            axis_sum = min(marker_rows) + max(payload_rows)
        if orientation is None:
            return np.array(grid, copy=True)
        output = np.full(grid.shape, fill_color, dtype=int)
        rows, cols = grid.shape
        for row, col, color in payload_cells:
            output[row, col] = color
            if orientation == 'vertical':
                reflected_row = row
                reflected_col = axis_sum - col
            else:
                reflected_row = axis_sum - row
                reflected_col = col
            if 0 <= reflected_row < rows and 0 <= reflected_col < cols:
                output[reflected_row, reflected_col] = color
        return output

    def create_grids(self):
        grid_size = random.randint(10, 14)
        colors = retry(
            lambda: random.sample(range(1, 10), 7),
            lambda values: len(set(values)) == 7,
        )
        marker_color, fill_color = colors[:2]
        taskvars = {
            'grid_size': grid_size,
            'marker_color': marker_color,
            'fill_color': fill_color,
        }
        sides = ['right', 'top', 'left', 'bottom']
        random.shuffle(sides)
        dimensions = [(2, 3), (3, 2), (3, 3), (2, 4)]
        train = []
        for payload_color, side, (height, width) in zip(
            colors[2:6], sides, dimensions
        ):
            input_grid = self.create_input(taskvars, {
                'payload_color': payload_color,
                'side': side,
                'height': height,
                'width': width,
            })
            train.append(GridPair(
                input=input_grid,
                output=self.transform_input(input_grid, taskvars),
            ))
        test_input = self.create_input(taskvars, {
            'payload_color': colors[6],
            'side': random.choice(sides),
            'height': 4,
            'width': 3,
        })
        test = [GridPair(
            input=test_input,
            output=self.transform_input(test_input, taskvars),
        )]
        return taskvars, TrainTestData(train=train, test=test)
