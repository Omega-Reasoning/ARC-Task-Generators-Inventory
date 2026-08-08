from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random


class Taskb1948b0aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Every cell belongs to either the {color('source_color')} pattern or the {color('background_color')} background.",
            "2. Input dimensions and the spatial arrangement of the two colors vary between examples.",
            "3. Both colors occur in every nontrivial input.",
            "4. The color roles stay fixed across one generated task episode."
        ]
        transformation_reasoning_chain = [
            "1. Copy the input grid without changing its dimensions or arrangement.",
            "2. Replace every {color('source_color')} cell with {color('target_color')}.",
            "3. Leave every {color('background_color')} cell unchanged.",
            "4. Return the resulting cellwise color substitution."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        source_color, target_color, background_color = random.sample(range(1, 10), 3)
        taskvars = {
            'source_color': source_color,
            'target_color': target_color,
            'background_color': background_color,
        }
        train_gridvars = [
            {'rows': 3, 'cols': 5, 'density': 0.35},
            {'rows': 5, 'cols': 4, 'density': 0.5},
            {'rows': 4, 'cols': 7, 'density': 0.65},
            {'rows': 6, 'cols': 6, 'density': 0.42},
        ]
        test_gridvars = {'rows': 7, 'cols': 5, 'density': 0.58}
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
        source_color = taskvars['source_color']
        background_color = taskvars['background_color']

        def sample_grid():
            grid = np.full((rows, cols), background_color, dtype=int)
            source_mask = gridvars.get(
                'source_mask',
                random_cell_coloring(
                    grid,
                    source_color,
                    density=gridvars['density'],
                    background=background_color,
                    overwrite=False,
                ) == source_color,
            )
            return np.where(source_mask, source_color, background_color)

        try:
            return retry(
                sample_grid,
                lambda value: 2 <= np.count_nonzero(value == source_color) < value.size,
                max_attempts=20,
            )
        except ValueError:
            grid = np.full((rows, cols), background_color, dtype=int)
            for index in range(min(rows, cols)):
                grid[index, index] = source_color
            return grid

    def transform_input(self, grid, taskvars):
        source_color = taskvars['source_color']
        target_color = taskvars['target_color']
        output = grid.copy()
        output[grid == source_color] = target_color
        return output
