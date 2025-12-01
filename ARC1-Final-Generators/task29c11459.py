from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task29c11459Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "Only two cells are placed along a single row, each with different colors.",
            "The first cell is placed in the first column and the second cell is placed in the last column.",
            "The remaining cells are empty(0).",
            "Different rows may have different color combinations."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "For each row with the pattern:",
            "- The middle cell along the same row should be colored {color('cell3')}.",
            "- All the cells between the first cell and the middle cell should be colored with the same color as the first cell.",
            "- All the cells between the middle cell and the last cell should be colored with the same color as the last cell."
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
        
        # Place cells in each selected row with varying colors
        for row in pattern_rows:
            cell1 = random.randint(1, 9)
            cell2 = random.randint(1, 9)
            while cell2 == cell1:
                cell2 = random.randint(1, 9)
            
            # Make sure the colors are different from the fixed middle color
            while cell1 == taskvars['cell3']:
                cell1 = random.randint(1, 9)
            while cell2 == taskvars['cell3'] or cell2 == cell1:
                cell2 = random.randint(1, 9)
            
            grid[row, 0] = cell1
            grid[row, cols - 1] = cell2
        
        return grid

    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Find all rows that have the pattern (non-zero value in first column)
        pattern_rows = np.where(grid[:, 0] != 0)[0]
        
        mid_col = grid.shape[1] // 2
        
        # Apply transformation to each pattern row
        for row in pattern_rows:
            cell1 = grid[row, 0]  # Read the color from the grid
            cell2 = grid[row, -1]  # Read the color from the grid
            
            # Use the fixed middle color from taskvars
            output_grid[row, mid_col] = taskvars['cell3']
            output_grid[row, 1:mid_col] = cell1
            output_grid[row, mid_col + 1:-1] = cell2
        
        return output_grid

    def create_grids(self):
        taskvars = {
            'rows': random.randint(10, 21),
            'cols': random.choice([i for i in range(5, 31, 2)]),
            'cell3': random.randint(1, 9)  # Fixed middle color across all examples
        }

        train_test_data = self.create_grids_default(nr_train_examples=random.randint(2, 3), nr_test_examples=1, taskvars=taskvars)

        return taskvars, train_test_data