from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
import numpy as np
import random


class Task28e73c20Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a square grid containing only empty background cells.",
            "2. The side length varies between examples and can be odd or even.",
            "3. Larger grids permit more nested turns of the same construction.",
        ]
        transformation_reasoning_chain = [
            "1. Start at the top-left cell and draw the full top edge in {color('spiral_color')}.",
            "2. Continue clockwise around the boundary, leaving a one-cell opening below the starting corner.",
            "3. Turn inward and repeat the rectangular path every two cells so adjacent coils remain separated by one empty cell.",
            "4. Stop when the remaining center can no longer support another turn.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        return np.zeros((gridvars['size'], gridvars['size']), dtype=int)

    def transform_input(self, grid, taskvars):
        spiral_color = taskvars['spiral_color']
        rows, cols = grid.shape
        output = np.array(grid, copy=True)
        layer = 0
        while True:
            top = 2 * layer
            right = cols - 1 - 2 * layer
            bottom = rows - 1 - 2 * layer
            left = 2 * layer
            top_start = max(0, left - 2)
            if top_start > right or top >= rows:
                break
            for col in range(top_start, right + 1):
                output[top, col] = spiral_color
            if top > bottom:
                break
            for row in range(top, bottom + 1):
                output[row, right] = spiral_color
            if bottom > top:
                for col in range(left, right + 1):
                    output[bottom, col] = spiral_color
            if left < right:
                for row in range(top + 2, bottom + 1):
                    output[row, left] = spiral_color
            layer += 1
        return output

    def create_grids(self):
        taskvars = {'spiral_color': random.randint(1, 9)}
        train_sizes = retry(
            lambda: random.sample(range(6, 18), 4),
            lambda sizes: (
                any(size % 2 == 0 for size in sizes)
                and any(size % 2 == 1 for size in sizes)
            ),
        )
        train = []
        for size in sorted(train_sizes):
            input_grid = self.create_input(taskvars, {'size': size})
            train.append(GridPair(
                input=input_grid,
                output=self.transform_input(input_grid, taskvars),
            ))
        test_size = random.randint(max(train_sizes) + 1, 24)
        test_input = self.create_input(taskvars, {'size': test_size})
        test = [GridPair(
            input=test_input,
            output=self.transform_input(test_input, taskvars),
        )]
        return taskvars, TrainTestData(train=train, test=test)
