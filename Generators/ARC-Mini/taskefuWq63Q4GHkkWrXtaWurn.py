from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object
import numpy as np
import random

class TaskefuWq63Q4GHkkWrXtaWurnGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "All input grids are squares.",
            "Let n be the height and width of an input grid, where n can only be (3,5,7,9).",
            "They contain a {color('object_color')} border with an empty (0) interior."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling in the empty (0) interior cells by concentric squares.",
            "Each output grid retains the same border color as the input grid, while the inner concentric squares are colored according to the following pattern.",
            "{color('object_color1')} → {color('object_color2')} → {color('object_color3')} → {color('object_color4')}",
            "Each concentric square 1 cell wide."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        n = gridvars['n']
        object_color = taskvars['object_color']

        grid = np.zeros((n, n), dtype=int)
        grid[0, :] = object_color
        grid[-1, :] = object_color
        grid[:, 0] = object_color
        grid[:, -1] = object_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()
        n = grid.shape[0]
        colors = [
            taskvars['object_color1'],
            taskvars['object_color2'],
            taskvars['object_color3'],
            taskvars['object_color4']
        ]

        layer = 0
        while layer < n // 2:
            color = colors[layer % len(colors)]
            output_grid[layer+1:n-layer-1, layer+1:n-layer-1] = color
            layer += 1

        return output_grid

    def create_grids(self) -> tuple:
        taskvars = {
            'object_color': random.randint(1, 9),
            'object_color1': random.randint(1, 9),
            'object_color2': random.randint(1, 9),
            'object_color3': random.randint(1, 9),
            'object_color4': random.randint(1, 9)
        }

        while len(set(taskvars.values())) < 5:
            taskvars = {
                'object_color': random.randint(1, 9),
                'object_color1': random.randint(1, 9),
                'object_color2': random.randint(1, 9),
                'object_color3': random.randint(1, 9),
                'object_color4': random.randint(1, 9)
            }

        train_sizes = [5, 9]
        test_size = random.choice([3, 7])

        train_data = []
        for size in train_sizes:
            gridvars = {'n': size}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})

        test_gridvars = {'n': test_size}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)

        test_data = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_data, 'test': test_data}

