from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity
from transformation_library import GridObject
import numpy as np
import random

class TasktaskfJMaEvLe8WKAAjgDkS6kbSGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids are of size nxn.",
            "Each input grid contains i same-colored cells (blue, green, or orange). The remaining cells are empty (0).",
            "The position of the colored cells can vary from example to example."
        ]
        transformation_chain = [
            "Output grids are constructed by initializing a zero-filled grid of the same size as the input grid.",
            "Each output grid contains, a 4-way connected object, that is an ixi square which touches the top-left corner of the grid.",
            "Colors change based on input: blue to red, green to cyan, orange to maroon."
        ]
        super().__init__(observation_chain, transformation_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        size = taskvars['size']
        color = taskvars['color']
        i = taskvars['i']

        grid = np.zeros((size, size), dtype=int)
        
        for _ in range(i):
            r, c = np.random.randint(0, size, 2)
            grid[r, c] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        i = taskvars['i']
        color_map = {1: 2, 3: 8, 7: 9}
        new_color = color_map[taskvars['color']]

        output_grid = np.zeros_like(grid)
        output_grid[:i, :i] = new_color
        
        return output_grid

    def create_grids(self) -> tuple:
        taskvars = {
            'size': random.randint(8, 30),
            'i': random.randint(1, 6),
            'color': random.choice([1, 3, 7])
        }

        train_pairs = [
            {
                'input': (input_grid := self.create_input(taskvars, {})),
                'output': self.transform_input(input_grid, taskvars)
            }
            for _ in range(random.randint(3, 6))
        ]

        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_pairs, 'test': test_pairs}

