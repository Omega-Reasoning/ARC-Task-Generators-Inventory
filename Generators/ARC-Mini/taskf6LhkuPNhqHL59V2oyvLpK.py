import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object
from transformation_library import find_connected_objects

class Taskf6LhkuPNhqHL59V2oyvLpKGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain a completely filled grid border of {color('object_color')} color, with the interior arranged in a checkerboard pattern made of {color('cell_color1')} and {color('cell_color2')} cells.",
            "The checkerboard pattern alternates {color('cell_color1')} and {color('cell_color2')} cells across both rows and columns."
        ]

        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and removing (setting to 0) any {color('cell_color1')} and {color('cell_color2')} cells that are not connected to a {color('object_color')} cell."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        size = gridvars['grid_size']
        grid = np.zeros((size, size), dtype=int)
        
        # Set border with object color
        grid[0, :] = taskvars['object_color']
        grid[-1, :] = taskvars['object_color']
        grid[:, 0] = taskvars['object_color']
        grid[:, -1] = taskvars['object_color']
        
        # Fill interior with checkerboard pattern
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                if (i + j) % 2 == 0:
                    grid[i, j] = taskvars['cell_color1']
                else:
                    grid[i, j] = taskvars['cell_color2']
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()
        size = grid.shape[0]
        
        # Find cells that do not touch the object border
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                if grid[i, j] in (taskvars['cell_color1'], taskvars['cell_color2']):
                    neighbors = [grid[i-1, j], grid[i+1, j], grid[i, j-1], grid[i, j+1]]
                    if taskvars['object_color'] not in neighbors:
                        output_grid[i, j] = 0  # Set to empty
        
        return output_grid
    
    def create_grids(self):
        num_train_examples = random.randint(3, 4)
        grid_sizes = random.sample(range(6, 31), num_train_examples + 1)
        
        # Select three distinct colors
        colors = random.sample(range(1, 10), 3)
        taskvars = {
            'object_color': colors[0],
            'cell_color1': colors[1],
            'cell_color2': colors[2]
        }
        
        train_pairs = []
        for size in grid_sizes[:-1]:
            gridvars = {'grid_size': size}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        test_gridvars = {'grid_size': grid_sizes[-1]}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, TrainTestData(train=train_pairs, test=[GridPair(input=test_input, output=test_output)])


