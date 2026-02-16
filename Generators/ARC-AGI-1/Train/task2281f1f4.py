import numpy as np
import random
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry

class task2281f1f4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input is a square grid with size {vars['rows']}×{vars['rows']}.",
            "Some cells on the top row (row 0) and some cells on the rightmost column (the last column) are filled with color {color('cell_color')}.",
            "All other cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Start from a copy of the input grid for the output.",
            "For each cell (i, j) with i > 0 and j < n-1 (where n is the grid size): if the cell at the top row (0, j) is color {color('cell_color')} AND the cell at the rightmost column (i, n-1) is also color {color('cell_color')}, then set the output cell at (i, j) to color {color('output_color')}",
            "Leave all other cells unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cell_color = taskvars['cell_color']
        grid = np.zeros((rows, rows), dtype=int)
        
        num_first_row = random.randint(1, (3 * rows) // 4)
        num_last_col = random.randint(1, (3 * rows) // 4)
        
        first_row_indices = random.sample(range(0, rows-1), num_first_row)
        last_col_indices = random.sample(range(1, rows), num_last_col)
        
        for j in first_row_indices:
            grid[0, j] = cell_color
        
        for i in last_col_indices:
            grid[i, rows - 1] = cell_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        rows = len(grid)
        output_grid = grid.copy()
        
        # Find cell_color from the first row or last column
        cell_color = None
        for j in range(rows):
            if grid[0, j] != 0:
                cell_color = grid[0, j]
                break
        if cell_color is None:
            for i in range(rows):
                if grid[i, rows-1] != 0:
                    cell_color = grid[i, rows-1]
                    break
        
        output_color = taskvars['output_color']
        
        # Apply the transformation
        for i in range(1, rows):
            for j in range(0, rows - 1):
                if grid[0, j] == cell_color and grid[i, rows - 1] == cell_color:
                    output_grid[i, j] = output_color
                    
        return output_grid
    
    def create_grids(self):
        cell_color, output_color = random.sample(range(1, 10), 2)
        taskvars = {
            'rows': random.randint(5, 30),
            'cell_color': cell_color,
            'output_color': output_color
        }
        
        train_examples = [
            {
                'input': (input_grid := self.create_input(taskvars, {})),
                'output': self.transform_input(input_grid, taskvars)
            }
            for _ in range(random.randint(3, 4))
        ]
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
