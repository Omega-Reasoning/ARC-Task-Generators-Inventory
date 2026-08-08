from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects
from typing import Dict, Any, Tuple
import numpy as np
import random


class Taskce22a75aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is an implicit array of square macro-cells with side length {vars['block_size']}.",
            "2. Selected macro-cells contain one {color('marker_color')} marker, while all other cells use {color('background_color')}.",
            "3. Marker coordinates differ by multiples of {vars['block_size']} and identify their containing macro-cells.",
            "4. The macro-grid dimensions, number of markers, and selected-cell mask vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. For every {color('marker_color')} cell, identify its containing {vars['block_size']}x{vars['block_size']} macro-cell.",
            "2. Create a {color('background_color')} output with the same shape as the input.",
            "3. Fill every selected macro-cell completely with {color('output_color')}.",
            "4. Leave every unselected macro-cell entirely {color('background_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        background_color = taskvars['background_color']
        marker_color = taskvars['marker_color']
        block_size = taskvars['block_size']
        block_rows = gridvars['block_rows']
        block_cols = gridvars['block_cols']
        marker_count = gridvars['marker_count']

        def sample_blocks():
            return random.sample(
                [
                    (block_row, block_col)
                    for block_row in range(block_rows)
                    for block_col in range(block_cols)
                ],
                marker_count,
            )

        try:
            selected = gridvars.get(
                'selected_blocks',
                retry(
                    sample_blocks,
                    lambda value: (
                        len(set(value)) == marker_count
                        and len({row for row, _ in value}) >= min(2, block_rows)
                        and len({col for _, col in value}) >= min(2, block_cols)
                    ),
                    max_attempts=30,
                ),
            )
        except ValueError:
            selected = gridvars.get('selected_blocks', sample_blocks())
        selected = [tuple(map(int, value)) for value in selected]
        if not (
            len(selected) == marker_count
            and len(set(selected)) == marker_count
            and all(
                0 <= row < block_rows and 0 <= col < block_cols
                for row, col in selected
            )
            and len({row for row, _ in selected}) >= min(2, block_rows)
            and len({col for _, col in selected}) >= min(2, block_cols)
        ):
            raise ValueError('selected_blocks violate the macro-grid grammar')

        grid = np.full(
            (block_rows * block_size, block_cols * block_size),
            background_color,
            dtype=int,
        )
        local_offset = block_size // 2
        cells = {
            (
                block_row * block_size + local_offset,
                block_col * block_size + local_offset,
                marker_color,
            )
            for block_row, block_col in selected
        }
        GridObject(cells).paste(grid, overwrite=True, background=background_color)
        return grid

    def transform_input(self, grid, taskvars):
        background_color = taskvars['background_color']
        marker_color = taskvars['marker_color']
        output_color = taskvars['output_color']
        block_size = taskvars['block_size']
        output = np.full(grid.shape, background_color, dtype=int)
        markers = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background_color,
            monochromatic=True,
        ).with_color(marker_color)
        filled_block = np.full((block_size, block_size), output_color, dtype=int)
        for marker in markers:
            for row, col, _ in marker.cells:
                row_start = (int(row) // block_size) * block_size
                col_start = (int(col) // block_size) * block_size
                GridObject.from_array(
                    filled_block,
                    offset=(row_start, col_start),
                ).paste(output, overwrite=True, background=background_color)
        return output

    def create_grids(self):
        marker_color, output_color = random.sample(range(1, 10), 2)
        taskvars = {
            'background_color': 0,
            'marker_color': marker_color,
            'output_color': output_color,
            'block_size': random.randint(2, 4),
        }
        train_gridvars = [
            {'block_rows': 2, 'block_cols': 3, 'marker_count': 2},
            {'block_rows': 3, 'block_cols': 2, 'marker_count': 3},
            {'block_rows': 3, 'block_cols': 3, 'marker_count': 3},
            {'block_rows': 2, 'block_cols': 4, 'marker_count': 4},
        ]
        test_gridvars = [
            {'block_rows': 4, 'block_cols': 3, 'marker_count': 5}
        ]
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
