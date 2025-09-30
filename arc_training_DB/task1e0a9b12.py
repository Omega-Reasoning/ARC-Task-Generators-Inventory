from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import GridObject, find_connected_objects
import numpy as np
import random

class ARCTask1e0a9b12Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}",
            "Each column has different number of cells of the same color(1-9) and the remaining cells are empty(0).",
            "Each column has different color cells."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First copy the input grid.",
            "For each column, translate all the colored cells such that they are stacked on one another and do not overlap each other; the bottom of the colored stack should always touch the bottom edge of the output grid."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        x_values = taskvars['x']

        grid = np.zeros((rows, rows), dtype=int)
        used_colors = set()

        for col in range(rows):
            color = random.choice([c for c in range(1, 10) if c not in used_colors])
            used_colors.add(color)
            x = x_values[col]
            indices = random.sample(range(rows), x)
            for idx in indices:
                grid[idx, col] = color

        return grid

    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        output_grid = grid.copy()

        for col in range(cols):
            non_empty_cells = [grid[row, col] for row in range(rows) if grid[row, col] != 0]
            for row in range(rows):
                output_grid[row, col] = 0

            for i, value in enumerate(non_empty_cells):
                output_grid[rows - len(non_empty_cells) + i, col] = value

        return output_grid

    def create_grids(self):
        rows = random.randint(2, 9)
        
        train_data = []
        for _ in range(random.randint(3, 4)):
            x = [random.randint(1, rows-1) for _ in range(rows)]
            taskvars = {'rows': rows, 'x': x}
            
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})

        x = [random.randint(1, rows-1) for _ in range(rows)]
        taskvars = {'rows': rows, 'x': x}
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_data, 'test': test_data}

