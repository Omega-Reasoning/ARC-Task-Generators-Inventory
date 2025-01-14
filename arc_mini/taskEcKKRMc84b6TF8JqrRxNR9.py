# my_arcagi_task_generator.py

import numpy as np
import random

# Mandatory imports per specifications
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData



class TaskEcKKRMc84b6TF8JqrRxNR9Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialize the input reasoning chain
        input_reasoning_chain = [
            "Input grids are square and have different sizes.",
            "They only contain several multi-colored (1-9) cells in the first column.",
            "The remaining cells are empty (0)."
        ]
        
        # 2) Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and expanding each colored cell diagonally down to the right, using the same color, until it reaches the edge of the matrix."
        ]
        
        # 3) Call super().__init__(...)
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, 
                     taskvars: dict, 
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
          * The grid is square, of random size (5..20) or as given in gridvars['size'].
          * It has several multi-colored cells (1..9) in the first column.
          * The rest of the cells are 0 (empty).
          * At least two empty cells remain in the first column.
        """
        size = gridvars.get('size', random.randint(5, 20))
        grid = np.zeros((size, size), dtype=int)

        # Decide how many colored cells we want in the first column
        # Must leave at least 2 empty rows in the first column.
        # So maximum colored cells = size - 2
        max_colored = size - 2
        if max_colored < 1:
            max_colored = 1  # fallback, though size≥5 ensures this won't happen
        num_colored = random.randint(3, max_colored)

        # Randomly choose which row positions in the first column to color
        possible_rows = list(range(size))
        random.shuffle(possible_rows)
        chosen_rows = possible_rows[:num_colored]

        # Place each chosen cell with a random color from 1..9
        for r in chosen_rows:
            color = random.randint(1, 9)
            grid[r, 0] = color

        return grid

    def transform_input(self, 
                        grid: np.ndarray, 
                        taskvars: dict) -> np.ndarray:
        """
        Transform the input grid by:
          1) Copying the input grid to the output grid.
          2) For each colored cell in the input, expand its color diagonally
             down-right until reaching the edge of the grid.
        """
        out_grid = grid.copy()
        rows, cols = grid.shape

        # For each cell in the input that is colored (1..9),
        # fill the diagonal in the output with the same color
        for r in range(rows):
            for c in range(cols):
                color = grid[r, c]
                if color != 0:
                    # Expand along the diagonal while in bounds
                    i = 0
                    while (r + i) < rows and (c + i) < cols:
                        out_grid[r + i, c + i] = color
                        i += 1
        return out_grid

    def create_grids(self):
        """
        Create 3-4 train grids and 1 test grid, each with a different size.
        Return the task variables (empty here) plus the train/test data.
        """
        task_variables = {}  # We have no named variables to substitute in the template
        train_test_data: TrainTestData = {'train': [], 'test': []}

        # We want 3-4 train examples
        nr_train = random.randint(3, 4)
        used_sizes = set()

        # Create train examples
        for _ in range(nr_train):
            # Choose a unique size for each example
            while True:
                size = random.randint(5, 20)
                if size not in used_sizes:
                    used_sizes.add(size)
                    break
            
            input_grid = self.create_input(taskvars=task_variables, gridvars={'size': size})
            output_grid = self.transform_input(input_grid, taskvars=task_variables)
            train_test_data['train'].append(GridPair(input=input_grid, output=output_grid))

        # Create one test example with yet another size
        while True:
            size = random.randint(5, 20)
            if size not in used_sizes:
                used_sizes.add(size)
                break

        test_input_grid = self.create_input(taskvars=task_variables, gridvars={'size': size})
        test_output_grid = self.transform_input(test_input_grid, taskvars=task_variables)
        train_test_data['test'].append(GridPair(input=test_input_grid, output=test_output_grid))

        return task_variables, train_test_data


