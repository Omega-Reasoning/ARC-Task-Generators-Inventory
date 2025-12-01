from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task22eb0ac0Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}",
            "A maximum of {vars['rows']//2} cells are placed in the first and the last column.",
            "The cells are placed in alternate rows starting from the 2nd row till the last row, both in the first and the last column.",
            "Each of these cells are colored randomly (between 1-9).",
            "The remaining cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First copy the input grid to the output grid.",
            "If the cell color in the first column at position (i,0) is the same as the last column cell (i,row-1), then this row is filled with the color of the cell."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        grid = np.zeros((rows, rows), dtype=int)
        max_cells = rows // 2
        num_cells = random.randint(1, max_cells)  # Random number of cells up to max_cells
        
        # Get available row positions
        available_rows = list(range(1, rows, 2))
        selected_rows = random.sample(available_rows, num_cells)
        
        # Ensure at least one pair has the same color
        first_row = selected_rows[0]
        color = random.randint(1, 9)
        grid[first_row, 0] = color
        grid[first_row, rows - 1] = color
        
        # Fill remaining positions with random colors
        for i in selected_rows[1:]:
            grid[i, 0] = random.randint(1, 9)
            grid[i, rows - 1] = random.randint(1, 9)

        return grid

    def transform_input(self, grid, taskvars):
        rows = taskvars['rows']
        output_grid = grid.copy()

        for i in range(rows):
            if grid[i, 0] == grid[i, rows - 1] and grid[i, 0] != 0:
                output_grid[i, :] = grid[i, 0]

        return output_grid

    def create_grids(self):
        rows = random.randint(10, 18)
        taskvars = {'rows': rows}

        train_pairs = []
        for _ in range(random.randint(3, 4)):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)

        return taskvars, TrainTestData(train=train_pairs, test=[GridPair(input=test_input, output=test_output)])

