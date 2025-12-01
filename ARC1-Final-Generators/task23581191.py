from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import find_connected_objects
import numpy as np
import random

class Task23581191Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "Two cells are randomly placed in the input grid of color {color('color1')} and {color('color2')}.",
            "The remaining cells of the input grid are empty(0)."
        ]
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First copy the input grid to the output grid.",
            "If a colored cell occupies position (i,j), the ith row and jth column are filled with the same cell color.",
            "If the row of one colored cell crosses the column of the other colored cell, the intersection point is colored {color('color3')} and vice versa",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        grid = np.zeros((rows, rows), dtype=int)
        
        # Pick first position
        row1 = random.randint(0, rows-1)
        col1 = random.randint(0, rows-1)
        
        # Pick second position ensuring different row and column
        row2 = random.choice([i for i in range(rows) if i != row1])
        col2 = random.choice([i for i in range(rows) if i != col1])
        
        grid[row1, col1] = taskvars['color1']
        grid[row2, col2] = taskvars['color2']
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        rows, cols = grid.shape
        output_grid = grid.copy()
        
        # Extract positions of color1 and color2
        color1_pos = sorted([(i, j) for i in range(rows) for j in range(cols) if grid[i, j] == taskvars['color1']])
        color2_pos = sorted([(i, j) for i in range(rows) for j in range(cols) if grid[i, j] == taskvars['color2']])
        
        print(color2_pos)
        # Fill rows and columns for color1
        for (i, j) in color1_pos:
            output_grid[i, :] = taskvars['color1']
            output_grid[:, j] = taskvars['color1']

        ix, jx = color2_pos[0][0], color2_pos[0][1]
        for j in range(cols):
            if output_grid[ix, j] != taskvars['color1']:
                output_grid[ix, j] = taskvars['color2']
            else:
                output_grid[ix, j] = taskvars['color3']

        for i in range(rows):
            if output_grid[i, jx] != taskvars['color1']:
                output_grid[i, jx] = taskvars['color2']
            else:
                output_grid[i, jx] = taskvars['color3']



        return output_grid

    def create_grids(self) -> tuple:
        taskvars = {
            'rows': random.randint(9, 20),
            'color1': random.randint(1, 9),
            'color2': random.randint(1, 9),
            'color3': random.randint(1, 9),
        }

        # Ensure all colors are different
        while len({taskvars['color1'], taskvars['color2'], taskvars['color3']}) < 3:
            taskvars['color2'] = random.randint(1, 9)
            taskvars['color3'] = random.randint(1, 9)

        train_test_data = self.create_grids_default(nr_train_examples=random.randint(3, 4), nr_test_examples=1, taskvars=taskvars)
        return taskvars, train_test_data

