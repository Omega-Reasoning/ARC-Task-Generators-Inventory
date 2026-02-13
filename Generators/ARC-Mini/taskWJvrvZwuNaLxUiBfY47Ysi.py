# my_arc_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskWJvrvZwuNaLxUiBfY47YsiGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x3.",
            "Each input grid has the first row completely filled with same-colored (1-9) cells and a single differently colored (1-9) cell positioned in the middle of the second row.",
            "All other cells are empty (0)."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling the entire middle column with cells that match the color of the middle cell in the second row."
        ]

        # 3) Call parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Creates an input grid of size (rows x 3).
        - The first row is completely filled with the same color (gridvars['color_row']).
        - The second row, middle column is a different color (gridvars['color_middle']).
        - The rest of the cells are 0 (empty).
        """
        rows = taskvars['rows']
        color_row = gridvars['color_row']
        color_middle = gridvars['color_middle']

        # Create empty grid
        grid = np.zeros((rows, 3), dtype=int)

        # Fill the entire first row with color_row
        grid[0, :] = color_row

        # Put color_middle in the middle cell of the second row (row=1, col=1)
        if rows > 1:  # Safeguard, though rows>=5 by design
            grid[1, 1] = color_middle

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Implements the transformation reasoning chain:
        1) Copy the grid.
        2) Fill the entire middle column (col=1) with the color of the cell at (row=1, col=1).
        """
        output = grid.copy()
        # Color of the middle cell on the second row
        # (Safe to assume row=1 and col=1 exist because rows>=5, width=3)
        color_middle = grid[1, 1]
        # Fill the entire middle column
        output[:, 1] = color_middle
        return output
    def create_grids(self):
        """
        Creates 3 training examples and 1 test example.
        Invariants:
          - rows is between 5 and 30
        Constraints:
          - color_row != color_middle for each grid
          - The color pairs vary across the training examples
          - The test color pair is different from the training pairs
        """

        # Choose a single number of rows for all examples (can also be varied per example if desired)
        rows = random.randint(5, 30)

        # We will produce 3 train pairs and 1 test pair
        # We need 4 distinct color pairs (color_row, color_middle)
        # each color is in [1..9], color_row != color_middle
        all_colors = list(range(1, 10))

        def pick_distinct_pair(used_pairs):
            # Repeatedly pick a random color pair until it's not in used_pairs
            while True:
                c1 = random.choice(all_colors)
                c2 = random.choice(all_colors)
                if c1 != c2 and (c1, c2) not in used_pairs:
                    return c1, c2

        # Collect distinct pairs for 3 train and 1 test
        used = set()
        color_pairs = []
        for _ in range(4):
            pair = pick_distinct_pair(used)
            used.add(pair)
            color_pairs.append(pair)

        # Prepare structure
        train_data = []
        test_data = []

        # Create 3 training examples
        for i in range(3):
            c1, c2 = color_pairs[i]
            gridvars = {
                'color_row': c1,
                'color_middle': c2
            }
            input_grid = self.create_input({'rows': rows}, gridvars)
            output_grid = self.transform_input(input_grid, {'rows': rows})
            train_data.append(GridPair(input=input_grid, output=output_grid))

        # Create 1 test example
        c1_test, c2_test = color_pairs[3]
        test_gridvars = {
            'color_row': c1_test,
            'color_middle': c2_test
        }
        test_input = self.create_input({'rows': rows}, test_gridvars)
        test_output = self.transform_input(test_input, {'rows': rows})
        test_data.append(GridPair(input=test_input, output=test_output))

        # The dictionary of variables used in the reasoning chains
        # We only have {vars['rows']} in the chain
        taskvars = {
            'rows': rows
        }

        # Combine into the final structure
        train_test_data = TrainTestData(train=train_data, test=test_data)

        return taskvars, train_test_data



