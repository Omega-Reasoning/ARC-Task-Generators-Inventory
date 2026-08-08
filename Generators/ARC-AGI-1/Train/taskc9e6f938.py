from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry, create_object, Contiguity
from typing import Dict, Any, Tuple
import numpy as np
import random


class Taskc9e6f938Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a small {color('background_color')} raster containing a sparse {color('object_color')} pattern.",
            "2. The pattern may have several disconnected cells and is generally asymmetric from left to right.",
            "3. Input height and width can vary, but the same {vars['reflection_axis']} reflection rule applies.",
            "4. The complete input raster, including its empty cells, forms the left half of the output.",
        ]
        transformation_reasoning_chain = [
            "1. Copy the complete input as the left half of the output.",
            "2. Reflect the input across its {vars['reflection_axis']} axis, reversing the column order.",
            "3. Append the reflected copy directly to the right of the original, preserving the row count and doubling the width.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        background_color = taskvars['background_color']
        object_color = taskvars['object_color']
        rows = gridvars['rows']
        cols = gridvars['cols']

        def render_object_cells(values):
            cells = [tuple(map(int, value)) for value in values]
            if len(cells) != len(set(cells)):
                raise ValueError('object_cells must be unique')
            if any(not (0 <= row < rows and 0 <= col < cols) for row, col in cells):
                raise ValueError('object_cells must lie inside the raster')
            pattern = np.full((rows, cols), background_color, dtype=int)
            for row, col in cells:
                pattern[row, col] = object_color
            return pattern

        def sample_object_cells():
            pattern = create_object(
                rows,
                cols,
                object_color,
                contiguity=Contiguity.NONE,
                background=background_color,
            )
            return [tuple(map(int, cell)) for cell in np.argwhere(pattern == object_color)]

        def valid_object_cells(values):
            try:
                pattern = render_object_cells(values)
            except (TypeError, ValueError):
                return False
            return bool(
                1 <= int(np.count_nonzero(pattern != background_color)) < pattern.size
                and not np.array_equal(pattern, np.fliplr(pattern))
            )

        try:
            object_cells = gridvars.get(
                'object_cells',
                retry(
                    sample_object_cells,
                    valid_object_cells,
                    max_attempts=30,
                ),
            )
        except ValueError:
            object_cells = gridvars.get(
                'object_cells',
                [(0, 0), (rows - 1, max(0, cols - 2))],
            )
        if not valid_object_cells(object_cells):
            raise ValueError('object_cells do not form a valid asymmetric pattern')
        return render_object_cells(object_cells)

    def transform_input(self, grid, taskvars):
        reflection_axis = taskvars['reflection_axis']
        source = np.array(grid, copy=True)
        if reflection_axis == 'vertical':
            reflected = np.fliplr(source)
        else:
            reflected = np.flipud(source)
        return np.concatenate((source, reflected), axis=1)

    def create_grids(self):
        taskvars = {
            'background_color': 0,
            'object_color': random.randint(1, 9),
            'reflection_axis': 'vertical',
        }
        train_gridvars = [
            {'rows': 3, 'cols': 3},
            {'rows': 3, 'cols': 4},
            {'rows': 4, 'cols': 3},
            {'rows': 4, 'cols': 5},
        ]
        test_gridvars = [{'rows': 5, 'cols': 6}]
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
