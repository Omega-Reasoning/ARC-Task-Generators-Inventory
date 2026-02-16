# filename: diagonal_extension_task.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task2Kfzy4hxyK4Wm6wcQfYvCCGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain (list of strings)
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They only contain one {color('cell_color1')} and one {color('cell_color2')} diagonal line, each consisting of three cells, with the lines being parallel to each other.",
            "The two diagonal lines follow the main diagonal direction, extending from top-left to bottom-right.",
            "The {color('cell_color1')} diagonal line is always positioned above the {color('cell_color2')} diagonal line."
        ]

        # 2) The transformation reasoning chain (list of strings)
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and extending each diagonal line either towards the top-left or bottom-right, depending on their color.",
            "The {color('cell_color1')} diagonal line is extended towards the top-left corner, while the {color('cell_color2')} diagonal line is extended towards the bottom-right corner, until they reach the grid boundary."
        ]

        # 3) Call the superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Creates an input grid with two separate and parallel diagonal lines of length 3.
        The diagonals:
        - Must be distinct (not part of the same diagonal line)
        - Must be strictly within the grid (not touching the borders)
        - Must be positioned parallel to each other
        """
        grid_size = taskvars['grid_size']
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Define allowed row positions (ensuring enough space)
        min_row = 2
        max_row = grid_size - 5  # Ensuring space for another diagonal below

        # Select a row for the first diagonal (ensuring space above and below)
        row1 = random.randint(min_row, max_row)
        col1 = random.randint(2, grid_size - 5)  # Avoiding border collisions

        # Place the first diagonal (color1)
        for i in range(3):
            grid[row1 + i, col1 + i] = color1

        # The second diagonal must be strictly parallel
        row2 = row1 + 2  # Always maintaining exactly 2 rows of distance
        col2 = col1  # Ensuring the same column spacing

        # Place the second diagonal (color2)
        for i in range(3):
            grid[row2 + i, col2 + i] = color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transforms the input grid by extending:
        - The `cell_color1` diagonal towards the **top-left** corner
        - The `cell_color2` diagonal towards the **bottom-right** corner
        """
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']

        output_grid = np.copy(grid)

        # Extend color1 diagonal (top-left direction)
        coords_color1 = np.argwhere(output_grid == color1)
        if len(coords_color1) > 0:
            top_left_idx = min(coords_color1, key=lambda x: (x[0], x[1]))
            r, c = top_left_idx
            while r > 0 and c > 0:
                r -= 1
                c -= 1
                output_grid[r, c] = color1

        # Extend color2 diagonal (bottom-right direction)
        coords_color2 = np.argwhere(output_grid == color2)
        if len(coords_color2) > 0:
            bottom_right_idx = max(coords_color2, key=lambda x: (x[0], x[1]))
            r, c = bottom_right_idx
            max_r, max_c = output_grid.shape[0] - 1, output_grid.shape[1] - 1
            while r < max_r and c < max_c:
                r += 1
                c += 1
                output_grid[r, c] = color2

        return output_grid

    def create_grids(self):
        """
        Creates the train and test grids.
        - grid_size is randomly chosen between 7 and 30
        - Two distinct colors (cell_color1, cell_color2) are assigned
        - Ensures a high variety of diagonal placements across train examples
        """
        grid_size = random.randint(7, 30)
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)

        taskvars = {
            'grid_size': grid_size,
            'cell_color1': color1,
            'cell_color2': color2
        }

        num_train = random.choice([3, 4])
        num_test = 1

        train = []
        for _ in range(num_train):
            inp = self.create_input(taskvars, {})
            outp = self.transform_input(inp, taskvars)
            train.append(GridPair(input=inp, output=outp))

        test = []
        for _ in range(num_test):
            inp = self.create_input(taskvars, {})
            outp = self.transform_input(inp, taskvars)
            test.append(GridPair(input=inp, output=outp))

        data = TrainTestData(train=train, test=test)

        return taskvars, data


