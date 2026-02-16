import numpy as np
import random
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry

class TaskfGQvY4ECGiqPNkpCgwjFqjGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain one {color('cell_color1')}, one {color('cell_color2')}, and one {color('cell_color3')} cell, with all cells separated by empty (0) cells."
        ]

        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and extending each colored cell diagonally from all four corners until it reaches another colored cell or the grid boundary.",
            "The extension starts with {color('cell_color1')} cell, followed by {color('cell_color2')} cell, and ends with {color('cell_color3')} cell."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        colors = [taskvars['cell_color1'], taskvars['cell_color2'], taskvars['cell_color3']]
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        def valid_placement(coords):
            for r, c in coords:
                if any(
                    0 <= r + dr < grid_size and 0 <= c + dc < grid_size and grid[r + dr, c + dc] != 0
                    for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0)
                ):
                    return False
            return True
        
        placed_cells = []
        for color in colors:
            position = retry(
                lambda: (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)),
                lambda pos: valid_placement([pos])
            )
            grid[position] = color
            placed_cells.append(position)
        
        return grid

    def transform_input(self, grid, taskvars):
        grid_size = grid.shape[0]
        output_grid = grid.copy()
        colors = [taskvars['cell_color1'], taskvars['cell_color2'], taskvars['cell_color3']]
        
        for color in colors:
            object_cells = np.argwhere(grid == color)
            for r, c in object_cells:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nr, nc = r, c
                    while 0 <= nr + dr < grid_size and 0 <= nc + dc < grid_size and output_grid[nr + dr, nc + dc] == 0:
                        nr += dr
                        nc += dc
                        output_grid[nr, nc] = color
        
        return output_grid

    def create_grids(self):
        taskvars = {
            'grid_size': random.randint(5, 30),
            'cell_color1': random.randint(1, 9),
            'cell_color2': random.randint(1, 9),
            'cell_color3': random.randint(1, 9)
        }
        
        while len(set([taskvars['cell_color1'], taskvars['cell_color2'], taskvars['cell_color3']])) < 3:
            taskvars['cell_color2'] = random.randint(1, 9)
            taskvars['cell_color3'] = random.randint(1, 9)
        
        train_pairs = []
        for _ in range(random.randint(3, 4)):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_pairs, 'test': test_pairs}


