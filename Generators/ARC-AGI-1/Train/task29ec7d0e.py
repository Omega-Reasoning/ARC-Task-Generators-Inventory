from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
import numpy as np
import random


class Task29ec7d0eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['grid_size']} by {vars['grid_size']} crop of a periodic multiplication-color table with irregular cells or rectangular patches erased to empty.",
            "2. Its nonempty values are the consecutive colors 1 through an example-specific table order.",
            "3. Before erasure, the cell at zero-based row r and column c is one plus r times c modulo that table order.",
            "4. Table order and erased locations vary, while enough cells remain to show every table color.",
        ]
        transformation_reasoning_chain = [
            "1. Infer the table order as the largest visible nonempty color value.",
            "2. For every zero-based coordinate (r, c), compute 1 + ((r times c) modulo the table order).",
            "3. Write those values over the whole grid to restore the complete periodic table and all erased cells.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        order = gridvars['order']
        complete = np.zeros((grid_size, grid_size), dtype=int)
        for row in range(grid_size):
            for col in range(grid_size):
                complete[row, col] = 1 + ((row * col) % order)

        def erase_patches():
            candidate = np.array(complete, copy=True)
            for patch_index in range(gridvars['patch_count']):
                height = gridvars.get(
                    f'patch_{patch_index}_height', random.randint(1, 4)
                )
                width = gridvars.get(
                    f'patch_{patch_index}_width', random.randint(1, 4)
                )
                top = gridvars.get(
                    f'patch_{patch_index}_top',
                    random.randint(0, grid_size - height),
                )
                left = gridvars.get(
                    f'patch_{patch_index}_left',
                    random.randint(0, grid_size - width),
                )
                candidate[top:top + height, left:left + width] = 0
            for cell_index in range(grid_size // 2):
                row = gridvars.get(
                    f'cell_{cell_index}_row', random.randrange(grid_size)
                )
                col = gridvars.get(
                    f'cell_{cell_index}_col', random.randrange(grid_size)
                )
                candidate[row, col] = 0
            return candidate

        return retry(
            erase_patches,
            lambda candidate: (
                np.any(candidate == 0)
                and set(range(1, order + 1)).issubset(
                    set(int(value) for value in np.unique(candidate))
                )
            ),
        )

    def transform_input(self, grid, taskvars):
        order = int(np.max(grid))
        if order <= 0:
            return np.array(grid, copy=True)
        rows, cols = grid.shape
        output = np.zeros((rows, cols), dtype=int)
        for row in range(rows):
            for col in range(cols):
                output[row, col] = 1 + ((row * col) % order)
        return output

    def create_grids(self):
        taskvars = {'grid_size': random.randint(16, 22)}
        train_orders = random.sample(range(3, 9), 4)
        train = []
        for index, order in enumerate(train_orders):
            input_grid = self.create_input(taskvars, {
                'order': order,
                'patch_count': 4 + index,
            })
            train.append(GridPair(
                input=input_grid,
                output=self.transform_input(input_grid, taskvars),
            ))
        test_input = self.create_input(taskvars, {
            'order': 9,
            'patch_count': 8,
        })
        test = [GridPair(
            input=test_input,
            output=self.transform_input(test_input, taskvars),
        )]
        return taskvars, TrainTestData(train=train, test=test)
