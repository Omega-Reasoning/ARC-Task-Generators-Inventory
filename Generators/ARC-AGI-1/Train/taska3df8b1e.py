from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.transformation_library import GridObject
import numpy as np
import random


class Taska3df8b1eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input has {vars['canvas_height']} rows and {color('background_color')} background.",
            "2. Its width varies across examples.",
            "3. A single {color('path_color')} seed occupies the bottom-left cell.",
            "4. Every other input cell is background.",
        ]
        transformation_reasoning_chain = [
            "1. Begin at the bottom-left {color('path_color')} seed and initially move up-right.",
            "2. Add one {color('path_color')} cell after each one-row upward step.",
            "3. Reverse horizontal direction at either side boundary while continuing upward.",
            "4. Stop after placing one path cell in every row of the {vars['canvas_height']}-row grid.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            'canvas_height': random.randint(10, 15),
            'background_color': 0,
            'path_color': random.randint(1, 9),
        }
        train = []
        for width in [2, 3, 4, 5]:
            input_grid = self.create_input(taskvars, {'width': width})
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        test_input = self.create_input(taskvars, {'width': 6})
        return taskvars, {'train': train, 'test': [{'input': test_input, 'output': self.transform_input(test_input, taskvars)}]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        grid = np.full(
            (taskvars['canvas_height'], gridvars['width']),
            taskvars['background_color'],
            dtype=int,
        )
        seed = GridObject({(
            taskvars['canvas_height'] - 1,
            0,
            taskvars['path_color'],
        )})
        seed.paste(grid, background=taskvars['background_color'])
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background = taskvars['background_color']
        path_color = taskvars['path_color']
        seed = np.argwhere(grid == path_color)[0]
        row = int(seed[0])
        col = int(seed[1])
        direction = 1
        cells = set()
        while row >= 0:
            cells.add((row, col, path_color))
            next_col = col + direction
            if next_col < 0 or next_col >= grid.shape[1]:
                direction *= -1
                next_col = col + direction
            col = next_col
            row -= 1
        output = np.full_like(grid, background)
        GridObject(cells).paste(output, background=background)
        return output
