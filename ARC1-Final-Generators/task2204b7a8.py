from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring
from transformation_library import find_connected_objects
from typing import Tuple

class Task2204b7a8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "In the input grid either the first row and last row are colored with color color_1(between 1 and 9) and color_2(between 1 and 9) or the first and the last column.",
            "A random number of cells are placed in between the colored rows or columns with color {color('color3')}.",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "If the columns are colored, find the distance of each cell from both of these columns.",
            "Color the cells of {color('color3')} with either color_1 or color_2 based on the minimum distance between the cell and the columns.",
            "Follow the above two steps in case of rows as well."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        grid = np.zeros((rows, rows), dtype=int)
        
        if gridvars['orientation'] == 'row':
            grid[0, :] = gridvars['color1']
            grid[-1, :] = gridvars['color2']
            # Filter positions to exclude cells equidistant from both rows
            positions = [(r, c) for r in range(1, rows - 1) for c in range(1, rows - 1) 
                        if abs(r - 0) != abs(r - (rows - 1))]  # Skip middle row if rows is odd
        else:
            grid[:, 0] = gridvars['color1']
            grid[:, -1] = gridvars['color2']
            # Filter positions to exclude cells equidistant from both columns
            positions = [(r, c) for r in range(1, rows - 1) for c in range(1, rows - 1)
                        if abs(c - 0) != abs(c - (rows - 1))]  # Skip middle column if rows is odd

        if positions:  # Only proceed if there are valid positions
            num_cells = random.randint(1, min(len(positions), rows))
            random.shuffle(positions)

            for i in range(num_cells):
                r, c = positions[i]
                grid[r, c] = taskvars['color3']

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        rows, cols = grid.shape
        output_grid = grid.copy()
        
        # Determine orientation and colors from the input grid
        if np.all(grid[0, :] != 0) and np.all(grid[-1, :] != 0):
            orientation = 'row'
            color1 = grid[0, 0]
            color2 = grid[-1, 0]
        else:
            orientation = 'col'
            color1 = grid[0, 0]
            color2 = grid[0, -1]
        
        if orientation == 'row':
            for r in range(1, rows - 1):
                for c in range(cols):
                    if grid[r, c] == taskvars['color3']:
                        dist1 = abs(r - 0)
                        dist2 = abs(r - (rows - 1))
                        output_grid[r, c] = color1 if dist1 <= dist2 else color2
        else:
            for r in range(rows):
                for c in range(1, cols - 1):
                    if grid[r, c] == taskvars['color3']:
                        dist1 = abs(c - 0)
                        dist2 = abs(c - (cols - 1))
                        output_grid[r, c] = color1 if dist1 <= dist2 else color2

        return output_grid

    def create_grids(self) -> Tuple[dict, TrainTestData]:
        rows = random.randint(8, 20)
        color3 = random.randint(1, 9)  # Only color3 is fixed for the task

        taskvars = {
            'rows': rows,
            'color3': color3,
        }

        gridvars = {}
        train_data = []
        for _ in range(random.randint(3, 4)):
            # Choose new colors and orientation for each example
            color1, color2 = random.sample([i for i in range(1, 10) if i != color3], 2)
            gridvars['orientation'] = random.choice(['row', 'col'])
            gridvars['color1'] = color1
            gridvars['color2'] = color2
            input_grid = self.create_input(taskvars,gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))

        # Choose new colors and orientation for test example
        color1, color2 = random.sample([i for i in range(1, 10) if i != color3], 2)
        gridvars['orientation'] = random.choice(['row', 'col'])
        gridvars['color1'] = color1
        gridvars['color2'] = color2
        test_input = self.create_input(taskvars,gridvars)
        test_output = self.transform_input(test_input, taskvars)

        test_data = [GridPair(input=test_input, output=test_output)]

        return taskvars, {'train': train_data, 'test': test_data}


