# my_arc_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random


class TaskUmfctq9Jjj7wsQokCE4trUGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are square and can have different sizes.",
            "They contain one {color('cell_color1')} cell, one {color('cell_color2')} cell, and empty (0) cells.",
            "The {color('cell_color1')} cell is placed in the top-left corner, while the {color('cell_color2')} cell is positioned in the bottom-right corner of the grid."
        ]
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling all empty cells, except the border cells, with {color('cell_color2')}.",
            "This forms a {color('cell_color2')} rectangle, which is diagonally connected to the {color('cell_color1')} cell from its top-right edge and to a {color('cell_color2')} cell at the bottom-right."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid that is square with one cell_color1 in the top-left corner
        and one cell_color2 in the bottom-right corner. All other cells are 0 (empty).
        """
        size = gridvars["size"]
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]

        # Create an empty grid
        grid = np.zeros((size, size), dtype=int)

        # Place cell_color1 at top-left
        grid[0, 0] = cell_color1

        # Place cell_color2 at bottom-right
        grid[size - 1, size - 1] = cell_color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid by filling all interior empty cells (non-border cells) 
        with cell_color2.
        """
        cell_color2 = taskvars["cell_color2"]

        # Copy the grid
        output_grid = grid.copy()

        # Fill all interior cells that are empty (0) with cell_color2
        rows, cols = output_grid.shape
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if output_grid[r, c] == 0:
                    output_grid[r, c] = cell_color2

        return output_grid

    def create_grids(self):
        """
        Create 3-4 train examples and 1 test example. Each grid must have a unique size.
        Randomly assign cell_color1 and cell_color2 (1-9) ensuring they are different.
        """

        # Randomly select two distinct colors from [1, 9]
        cell_color1, cell_color2 = random.sample(range(1, 10), 2)

        # Task-wide variables
        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2
        }

        # Number of training examples: randomly pick between 3 or 4
        nr_train = random.choice([3, 4])

        # Collect distinct sizes. We need nr_train + 1 distinct sizes.
        possible_sizes = list(range(5, 31))
        random.shuffle(possible_sizes)

        chosen_sizes = possible_sizes[: (nr_train + 1)]

        # Build train set
        train_data = []
        for i in range(nr_train):
            gridvars = {"size": chosen_sizes[i]}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))

        # Build test set (just one test example)
        test_data = []
        test_gridvars = {"size": chosen_sizes[-1]}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_data.append(GridPair(input=test_input, output=test_output))

        train_test_data = TrainTestData(train=train_data, test=test_data)
        return taskvars, train_test_data


