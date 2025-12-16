from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import GridObject, find_connected_objects
import numpy as np
import random

class Task1e0a9b12Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['rows']}.",
            "Each column contains between 0 and {vars['rows'] - 1} single-colored cells (values 1-9) placed at random rows; some columns may be empty.",
            "The color within a column is consistent (all colored cells in a column have the same value) and they vary between columns."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "For each column in the grid, all colored cells (non-zero values) are moved down to the lowest available rows such that all appear as vertical stacks at the bottom of the column.",
            "Empty cells (value 0) fill the remaining spaces at the top of each column."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        # generate per-column counts internally (remove task variable 'x')
        # allow columns to be empty (0) but ensure NO column is completely filled
        # so max filled cells per column is rows-1
        x_values = [random.randint(0, max(0, rows - 1)) for _ in range(rows)]

        # ensure we can request certain columns to keep at least one empty cell below
        movable_cols = gridvars.get('movable_cols', []) if gridvars is not None else []

        grid = np.zeros((rows, rows), dtype=int)
        used_colors = set()

        for col in range(rows):
            x = x_values[col]

            # For columns we want to be "movable", avoid filling the bottom row
            if col in movable_cols:
                # sample from rows-1 so bottom row remains empty and colored cells can move down
                max_pop = max(0, rows - 1)
                x = min(x, max_pop)
                indices = random.sample(range(max_pop), x) if x > 0 else []
            else:
                # ensure we never fill an entire column
                x = min(x, max(0, rows - 1))
                indices = random.sample(range(rows), x) if x > 0 else []

            if not indices:
                # empty column (no colored cells)
                continue

            # pick a color for this column; prefer unused colors but allow reuse when necessary
            available_colors = [c for c in range(1, 10) if c not in used_colors]
            if available_colors:
                color = random.choice(available_colors)
                used_colors.add(color)
            else:
                color = random.choice(range(1, 10))

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
        # require at least 3 rows so we can guarantee 3 columns with empty cells below
        # allow larger grids (up to 30x30). Some columns may be empty.
        rows = random.randint(5, 30)
        
        train_data = []
        for _ in range(random.randint(3, 4)):
            taskvars = {'rows': rows}

            # pick 3 columns that will keep at least one empty cell below colored cells
            movable_cols = random.sample(range(rows), 3)

            input_grid = self.create_input(taskvars, {'movable_cols': movable_cols})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})

        taskvars = {'rows': rows}

        movable_cols = random.sample(range(rows), 3)

        test_input = self.create_input(taskvars, {'movable_cols': movable_cols})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_data, 'test': test_data}

