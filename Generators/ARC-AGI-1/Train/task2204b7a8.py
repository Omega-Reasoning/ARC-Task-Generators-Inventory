from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, random_cell_coloring
from Framework.transformation_library import find_connected_objects
from typing import Tuple

class Task2204b7a8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input is a square grid of size {vars['rows']} x {vars['rows']}.",
            "Exactly two opposite boundaries are filled: either the first and last ROW, or the first and last COLUMN.",
            "Those two boundaries are filled with two distinct colors (color_1 and color_2).",
            "A small random set of interior cells (not on the boundary) are filled with a third color {color('color3')}.",
            "All other cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "For each interior cell that is color {color('color3')}, compute its distance to the two filled boundaries (rows or columns).",
            "Replace each {color('color3')} cell with the color of the nearer boundary (color_1 or color_2)."
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
            # Ensure at least 4 interior cells are filled in each grid.
            max_possible = min(len(positions), rows)
            num_cells = random.randint(4, max_possible)
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
       
        # Choose an even grid size between 8 and 20 (inclusive)
        rows = random.choice(list(range(8, 21, 2)))
        color3 = random.randint(1, 9)  # Only color3 is fixed for the task

        taskvars = {
            'rows': rows,
            'color3': color3,
        }

        gridvars = {}
        train_data = []

        # Ensure at least one training example uses 'row' orientation and
        # at least one uses 'col' orientation. Fill the remaining examples
        # randomly so the dataset always contains both kinds of boundary fills.
        train_count = random.randint(3, 4)
        # Start with both orientations present, then add random choices for remaining
        orientations = ['row', 'col']
        if train_count > 2:
            orientations += [random.choice(['row', 'col']) for _ in range(train_count - 2)]
        random.shuffle(orientations)

        for orient in orientations:
            color1, color2 = random.sample([i for i in range(1, 10) if i != color3], 2)
            gridvars['orientation'] = orient
            gridvars['color1'] = color1
            gridvars['color2'] = color2
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))

        # For the test example, if both orientations are already present in train,
        # pick a random orientation. Otherwise, pick the missing one to ensure
        # the overall dataset contains both types.
        present_orients = set(orientations)
        if len(present_orients) == 2:
            test_orient = random.choice(['row', 'col'])
        else:
            test_orient = ('col' if 'row' in present_orients else 'row')

        color1, color2 = random.sample([i for i in range(1, 10) if i != color3], 2)
        gridvars['orientation'] = test_orient
        gridvars['color1'] = color1
        gridvars['color2'] = color2
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)

        test_data = [GridPair(input=test_input, output=test_output)]

        return taskvars, {'train': train_data, 'test': test_data}


