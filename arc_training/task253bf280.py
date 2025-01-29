from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class FillBetweenCellsTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}",
            "Random numbers of cells of color {color('cell_color')} are placed in the input grid.",
            "There are at most 2 cells placed along any particular row or column.",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "For each row, if there are exactly two cells of color {color('cell_color')}, fill all cells between them with color {color('fill_color')}.",
            "For each column, if there are exactly two cells of color {color('cell_color')}, fill all cells between them with color {color('fill_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        cell_color = taskvars['cell_color']
        grid = np.zeros((rows, cols), dtype=int)
        
        num_cells = random.randint(2, min(rows, cols))
        positions = set()
        
        # Keep track of cells per row and column
        row_counts = [0] * rows
        col_counts = [0] * cols
        
        while len(positions) < num_cells:
            r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
            # Only add position if row and column haven't reached max of 2 cells
            if row_counts[r] < 2 and col_counts[c] < 2 and (r, c) not in positions:
                positions.add((r, c))
                row_counts[r] += 1
                col_counts[c] += 1
        
        for r, c in positions:
            grid[r, c] = cell_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        fill_color = taskvars['fill_color']
        output_grid = np.copy(grid)
        
        rows, cols = grid.shape
        for r in range(rows):
            filled_positions = [c for c in range(cols) if grid[r, c] != 0]
            if len(filled_positions) == 2:
                for c in range(filled_positions[0] + 1, filled_positions[1]):
                    output_grid[r, c] = fill_color
        
        for c in range(cols):
            filled_positions = [r for r in range(rows) if grid[r, c] != 0]
            if len(filled_positions) == 2:
                for r in range(filled_positions[0] + 1, filled_positions[1]):
                    output_grid[r, c] = fill_color
        
        return output_grid
    
    def create_grids(self):
        taskvars = {
            'rows': random.randint(10, 30),
            'cols': random.randint(10, 30),
            'cell_color': random.randint(1, 9),
            'fill_color': random.randint(1, 9)
        }
        while taskvars['cell_color'] == taskvars['fill_color']:
            taskvars['fill_color'] = random.randint(1, 9)
        
        num_train_examples = random.randint(3, 4)
        num_test_examples = 1
        
        return taskvars, self.create_grids_default(num_train_examples, num_test_examples, taskvars)
