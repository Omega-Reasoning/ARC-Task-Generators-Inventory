from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry, create_object, Contiguity
from Framework.transformation_library import GridObject
from typing import Dict, Any, Tuple
import numpy as np
import random


class Taskcce03e0dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a small {color('background_color')} raster containing {color('selector_color')} cells and {color('content_color')} cells.",
            "2. The complete input raster is the pattern that can be copied into output blocks.",
            "3. Every input coordinate addresses one output block with the same height and width as the input.",
            "4. Selector count and arrangement vary between examples while role colors stay fixed within the episode.",
        ]
        transformation_reasoning_chain = [
            "1. Create a background output with input-height squared rows and input-width squared columns.",
            "2. For every {color('selector_color')} input cell, place a complete unchanged copy of the input in its corresponding output block.",
            "3. Leave the block for every non-{color('selector_color')} cell entirely {color('background_color')}.",
            "4. Preserve all source colors and local source coordinates inside every copied block.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        background_color = taskvars['background_color']
        selector_color = taskvars['selector_color']
        content_color = taskvars['content_color']
        rows = gridvars['rows']
        cols = gridvars['cols']
        selector_count = gridvars['selector_count']

        def normalize_positions(values, name):
            if not isinstance(values, (list, tuple)):
                raise ValueError(f'{name} must be a coordinate list')
            positions = []
            for value in values:
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(f'{name} entries must be row/column pairs')
                row, col = value
                if (
                    isinstance(row, (bool, np.bool_))
                    or isinstance(col, (bool, np.bool_))
                    or not isinstance(row, (int, np.integer))
                    or not isinstance(col, (int, np.integer))
                ):
                    raise ValueError(f'{name} coordinates must be integers')
                position = (int(row), int(col))
                if not (0 <= position[0] < rows and 0 <= position[1] < cols):
                    raise ValueError(f'{name} coordinates must lie inside the raster')
                positions.append(position)
            if len(positions) != len(set(positions)):
                raise ValueError(f'{name} coordinates must be unique')
            return positions

        def render_source_positions(values):
            if not isinstance(values, dict) or set(values) != {
                'selector_positions',
                'content_positions',
            }:
                raise ValueError(
                    'source_positions must contain exactly selector_positions '
                    'and content_positions'
                )
            selector_positions = normalize_positions(
                values['selector_positions'], 'selector_positions'
            )
            content_positions = normalize_positions(
                values['content_positions'], 'content_positions'
            )
            if set(selector_positions) & set(content_positions):
                raise ValueError('selector and content positions must be disjoint')
            source = np.full((rows, cols), background_color, dtype=int)
            for row, col in selector_positions:
                source[row, col] = selector_color
            for row, col in content_positions:
                source[row, col] = content_color
            return source

        def sample_source_positions():
            sampled = create_object(
                rows,
                cols,
                [selector_color, content_color],
                contiguity=Contiguity.NONE,
                background=background_color,
            )
            return {
                'selector_positions': [
                    tuple(map(int, value))
                    for value in np.argwhere(sampled == selector_color)
                ],
                'content_positions': [
                    tuple(map(int, value))
                    for value in np.argwhere(sampled == content_color)
                ],
            }

        def valid_source_positions(values):
            try:
                source = render_source_positions(values)
            except (TypeError, ValueError):
                return False
            return bool(
                int(np.count_nonzero(source == selector_color)) == selector_count
                and int(np.count_nonzero(source == content_color)) >= 1
                and int(np.count_nonzero(source != background_color)) < source.size
            )

        def fallback_source_positions():
            fallback = np.full((rows, cols), background_color, dtype=int)
            coordinates = [
                (row, col)
                for row in range(rows)
                for col in range(cols)
            ]
            for row, col in coordinates[:selector_count]:
                fallback[row, col] = selector_color
            content_index = min(selector_count, len(coordinates) - 1)
            fallback[coordinates[content_index]] = content_color
            return {
                'selector_positions': [
                    tuple(map(int, value))
                    for value in np.argwhere(fallback == selector_color)
                ],
                'content_positions': [
                    tuple(map(int, value))
                    for value in np.argwhere(fallback == content_color)
                ],
            }

        try:
            source_positions = gridvars.get(
                'source_positions',
                retry(
                    sample_source_positions,
                    valid_source_positions,
                    max_attempts=80,
                ),
            )
        except ValueError:
            source_positions = gridvars.get(
                'source_positions', fallback_source_positions()
            )
        if not valid_source_positions(source_positions):
            raise ValueError('source_positions violate the original source predicate')
        return render_source_positions(source_positions)

    def transform_input(self, grid, taskvars):
        background_color = taskvars['background_color']
        selector_color = taskvars['selector_color']
        source = np.array(grid, copy=True)
        rows, cols = source.shape
        output = np.full((rows * rows, cols * cols), background_color, dtype=int)
        local_pattern = np.where(source == background_color, 0, source)
        for row, col in np.argwhere(source == selector_color):
            GridObject.from_array(
                local_pattern,
                offset=(int(row) * rows, int(col) * cols),
            ).paste(output, overwrite=True, background=background_color)
        return output

    def create_grids(self):
        selector_color, content_color = random.sample(range(1, 10), 2)
        taskvars = {
            'background_color': 0,
            'selector_color': selector_color,
            'content_color': content_color,
        }
        train_gridvars = [
            {'rows': 3, 'cols': 3, 'selector_count': 1},
            {'rows': 3, 'cols': 3, 'selector_count': 2},
            {'rows': 3, 'cols': 3, 'selector_count': 3},
            {'rows': 3, 'cols': 3, 'selector_count': 4},
        ]
        test_gridvars = [{'rows': 4, 'cols': 4, 'selector_count': 5}]
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
