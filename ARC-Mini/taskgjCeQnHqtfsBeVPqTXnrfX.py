from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object
from transformation_library import GridObject
import numpy as np
import random

class TaskgjCeQnHqtfsBeVPqTXnrfXGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid has the first row and column filled with {color('object_color')} color and the remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "To construct the output grid, create an empty (0) grid of the same size as the input and color the entire last row and column with the same color as in the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        size = gridvars['grid_size']
        object_color = taskvars['object_color']
        grid = np.zeros((size, size), dtype=int)
        grid[0, :] = object_color
        grid[:, 0] = object_color
        return grid

    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        output_grid = np.zeros_like(grid)
        output_grid[-1, :] = object_color
        output_grid[:, -1] = object_color
        return output_grid

    def create_grids(self):
        taskvars = {
            'object_color': random.randint(1, 9)
        }
        
        train_sizes = random.sample(range(5, 30), 3)
        test_size = random.choice([size for size in range(5, 30) if size not in train_sizes])
        
        grids = {'grid_size': train_sizes + [test_size]}
        
        train_examples = []
        for size in train_sizes:
            gridvars = {'grid_size': size}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append(GridPair(input=input_grid, output=output_grid))

        test_gridvars = {'grid_size': test_size}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_examples, test=test_examples)

