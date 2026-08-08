from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry, random_cell_coloring
from Framework.transformation_library import find_connected_objects
from typing import Dict, Any, Tuple
import numpy as np
import random


class Taskc8f0f002Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a fully colored rectangular grid containing {color('source_color')} and two unchanged colors.",
            "2. Cells of {color('source_color')} may be isolated or connected and occur at varying coordinates.",
            "3. The two distractor colors are {color('unchanged_color_1')} and {color('unchanged_color_2')}.",
            "4. The grid dimensions and the counts of all three input colors vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Find every cell whose color is {color('source_color')}, including all disconnected components.",
            "2. Recolor each such cell {color('target_color')} at the same coordinate.",
            "3. Preserve every {color('unchanged_color_1')} and {color('unchanged_color_2')} cell and keep the original grid shape.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        source_color = taskvars['source_color']
        unchanged_color_1 = taskvars['unchanged_color_1']
        unchanged_color_2 = taskvars['unchanged_color_2']
        rows = gridvars['rows']
        cols = gridvars['cols']
        source_block = gridvars['source_block']

        def sample_grid():
            grid = np.full((rows, cols), unchanged_color_1, dtype=int)
            random_cell_coloring(
                grid,
                [source_color, unchanged_color_2],
                density=gridvars.get(
                    "density",
                    random.uniform(0.45, 0.75),
                ),
                background=unchanged_color_1,
                overwrite=False,
            )
            grid = np.asarray(gridvars.get("grid", grid), dtype=int)
            if source_block and rows >= 2 and cols >= 2:
                grid[:2, :2] = source_color
            return grid

        try:
            return retry(
                sample_grid,
                lambda value: (
                    int(np.count_nonzero(value == source_color)) >= 2
                    and set(int(color) for color in np.unique(value))
                    == {source_color, unchanged_color_1, unchanged_color_2}
                ),
                max_attempts=30,
            )
        except ValueError:
            fallback = np.full((rows, cols), unchanged_color_1, dtype=int)
            fallback[0, 0] = source_color
            fallback[-1, -1] = source_color
            fallback[0, -1] = unchanged_color_2
            return fallback

    def transform_input(self, grid, taskvars):
        source_color = taskvars['source_color']
        target_color = taskvars['target_color']
        output = np.array(grid, copy=True)
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=target_color,
            monochromatic=True,
        )
        objects.with_color(source_color).color_all(target_color).paste(
            output,
            overwrite=True,
            background=target_color,
        )
        return output

    def create_grids(self):
        source_color, target_color, unchanged_color_1, unchanged_color_2 = random.sample(
            range(1, 10),
            4,
        )
        taskvars = {
            'source_color': source_color,
            'target_color': target_color,
            'unchanged_color_1': unchanged_color_1,
            'unchanged_color_2': unchanged_color_2,
        }
        train_gridvars = [
            {'rows': 3, 'cols': 4, 'source_block': False},
            {'rows': 3, 'cols': 6, 'source_block': True},
            {'rows': 4, 'cols': 5, 'source_block': False},
            {'rows': 5, 'cols': 4, 'source_block': True},
        ]
        test_gridvars = [{'rows': 5, 'cols': 7, 'source_block': False}]
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
