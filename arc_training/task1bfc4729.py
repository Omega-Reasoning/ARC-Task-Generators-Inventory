from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class ARCTask1bfc4729Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}",
            "First a cell {color('cell1')} is placed randomly in the first half of the rows in the input grid.",
            "A second cell {color('cell2')} is placed on the second half of the rows in input grid symmetrical to the first cell(but can be placed in a different column than the first cell), i.e. if the first cell is placed at ith row then the second cell is placed at row {vars['rows']} - i (the row indexing starts from 0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same dimensions as the input grid.",
            "The input grid is copied to the output grid.",
            "The row which has a cell with color {color('cell1')} and the 1st row is completely filled with {color('cell1')}.",
            "The row which has a cell with color {color('cell2')} and the last row is completely filled with {color('cell2')}.",
            "The first column and the last column are filled with color {color('cell1')} from 1st row to the middle row (floor({vars['rows']}/2)).",
            "The first column and the last column are filled with color {color('cell2')} from (middle + 1)th row to the last row."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        cell1 = taskvars['cell1']
        cell2 = taskvars['cell2']

        grid = np.zeros((rows, rows), dtype=int)

        # Place the first cell
        row1 = random.randint(0, rows // 2 - 1)
        col1 = random.randint(0, rows - 1)
        grid[row1, col1] = cell1

        # Place the second cell symmetrically
        row2 = rows - row1 - 1
        col2 = random.randint(0, rows - 1)
        grid[row2, col2] = cell2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        cell1 = taskvars['cell1']
        cell2 = taskvars['cell2']

        output_grid = grid.copy()

        # Find the rows with cell1 and cell2
        row1, _ = np.argwhere(grid == cell1)[0]
        row2, _ = np.argwhere(grid == cell2)[0]

        # Fill rows completely
        output_grid[row1, :] = cell1
        output_grid[0, :] = cell1
        output_grid[row2, :] = cell2
        output_grid[-1, :] = cell2

        # Fill first and last columns
        mid_row = rows // 2
        output_grid[:mid_row, 0] = cell1
        output_grid[:mid_row, -1] = cell1
        output_grid[mid_row:, 0] = cell2
        output_grid[mid_row:, -1] = cell2

        return output_grid

    def create_grids(self) -> tuple:
        rows = random.choice(range(8, 31, 2))
        
        # Create train and test examples
        nr_train = random.randint(3, 4)
        nr_test = 1
        train_test_data = TrainTestData({"train": [], "test": []})

        # Base taskvars
        taskvars = {
            'rows': rows,
        }

        # Generate examples
        train_examples = []
        for _ in range(nr_train):
            # Generate unique colors for each example
            cell1 = random.randint(1, 9)
            cell2 = random.randint(1, 9)
            while cell1 == cell2:
                cell2 = random.randint(1, 9)
            
            taskvars['cell1'] = cell1
            taskvars['cell2'] = cell2
            
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append(GridPair({"input": input_grid, "output": output_grid}))

        # Generate test examples
        test_examples = []
        for _ in range(nr_test):
            cell1 = random.randint(1, 9)
            cell2 = random.randint(1, 9)
            while cell1 == cell2:
                cell2 = random.randint(1, 9)
            
            taskvars['cell1'] = cell1
            taskvars['cell2'] = cell2
            
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_examples.append(GridPair({"input": input_grid, "output": output_grid}))

        return taskvars, TrainTestData({"train": train_examples, "test": test_examples})

