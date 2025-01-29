from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class ARCTask29c11459Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "Only two cells are placed along a single row, one cell is of color {color('cell1')} and another cell with color {color('cell2')}.",
            "The first cell is placed in the first column and the second cell is placed in the last column.",
            "The remaining cells are empty(0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "The middle cell along the same row where the two cells are placed should be colored {color('cell3')}.",
            "All the cells between the first cell and the middle cell should be colored {color('cell1')} and the cells between middle cell and second cell should be colored {color('cell2')}."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        num_pattern_rows = random.randint(1, 3)  # Randomly choose 1-3 rows to have the pattern
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Get random unique row indices
        available_rows = list(range(rows))
        pattern_rows = random.sample(available_rows, num_pattern_rows)
        
        # Place cells in each selected row
        for row in pattern_rows:
            grid[row, 0] = taskvars['cell1']
            grid[row, cols - 1] = taskvars['cell2']
        
        return grid

    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Find all rows that have the pattern (non-zero value in first column)
        pattern_rows = np.where(grid[:, 0] != 0)[0]
        
        mid_col = grid.shape[1] // 2
        
        # Apply transformation to each pattern row
        for row in pattern_rows:
            output_grid[row, mid_col] = taskvars['cell3']
            output_grid[row, 1:mid_col] = taskvars['cell1']
            output_grid[row, mid_col + 1:-1] = taskvars['cell2']
        
        return output_grid

    def create_grids(self):
        taskvars = {
            'rows': random.randint(10, 21),
            'cols': random.choice([i for i in range(5, 31, 2)]),
            'cell1': random.randint(1, 9),
            'cell2': random.randint(1, 9),
            'cell3': random.randint(1, 9)
        }

        while taskvars['cell1'] == taskvars['cell2'] or taskvars['cell2'] == taskvars['cell3'] or taskvars['cell1'] == taskvars['cell3']:
            taskvars['cell2'] = random.randint(1, 9)
            taskvars['cell3'] = random.randint(1, 9)

        train_test_data = self.create_grids_default(nr_train_examples=random.randint(2, 3), nr_test_examples=1, taskvars=taskvars)

        return taskvars, train_test_data
