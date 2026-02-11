from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optionally, you may import the transformation library, if you need to:
# from transformation_library import find_connected_objects, ...

# Optionally, you may also import from input_library if you need random object helpers:
# from input_library import ...

class TaskNAcaAjFvaFB5bNtmqdCgR4Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain:
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each input grid contains two colored (1-9) cells, with one on the main diagonal (top-left to bottom-right) and the other on the diagonal directly above it, with the remaining cells being empty (0).",
            "The two cells must always have different colors, and their positions can change within their respective diagonals."
        ]
        # 2) The transformation reasoning chain:
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and completely filling the main diagonal and the diagonal above it with their respective cell colors."
        ]
        # 3) Call the parent constructor:
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> (dict, TrainTestData):
        """
        Creates the dictionary of task variables and the train/test grids.
        We randomly choose a grid size once and use it for all examples.
        We generate 3 or 4 train pairs and 1 test pair.
        """
        # Randomly choose the grid size (between 5 and 30 as per instructions)
        grid_size = random.randint(5, 30)
        taskvars = {"grid_size": grid_size}

        # Randomly choose how many train examples (3 or 4)
        nr_train_examples = random.choice([3, 4])
        nr_test_examples = 1

        # Use the built-in helper to create the grids:
        # It calls create_input() and transform_input() for each example.
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
        - Size {vars['grid_size']} x {vars['grid_size']}
        - Exactly two colored cells: one on the main diagonal, one on the diagonal above it
        - Different colors
        - Random positions along those diagonals (ensuring valid indices)
        """
        grid_size = taskvars["grid_size"]
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Randomly pick different colors for the main diagonal and the diagonal above
        color_main = random.randint(1, 9)
        color_above = random.randint(1, 9)
        while color_above == color_main:
            color_above = random.randint(1, 9)

        # Random position for the main diagonal cell
        main_r = random.randint(0, grid_size - 1)
        grid[main_r, main_r] = color_main

        # Random position for the diagonal above it, which is col = row+1
        # We must ensure that row+1 is within bounds => row in [0..grid_size-2]
        above_r = random.randint(0, grid_size - 2)
        grid[above_r, above_r + 1] = color_above

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        - Copy the input
        - Determine the color on the main diagonal (exactly one non-zero cell in that diagonal)
        - Determine the color on the diagonal above it (exactly one non-zero cell in that diagonal)
        - Fill the entire main diagonal with its detected color
        - Fill the entire diagonal above it with its detected color
        """
        grid_size = taskvars["grid_size"]
        output_grid = np.copy(grid)

        # Find the color on the main diagonal (there should be exactly one non-zero)
        main_diag_color = None
        for i in range(grid_size):
            if grid[i, i] != 0:
                main_diag_color = grid[i, i]
                break

        # Find the color on the diagonal above (again, exactly one non-zero)
        above_diag_color = None
        for i in range(grid_size - 1):
            if grid[i, i + 1] != 0:
                above_diag_color = grid[i, i + 1]
                break

        # Fill main diagonal with main_diag_color
        if main_diag_color is not None:
            for i in range(grid_size):
                output_grid[i, i] = main_diag_color

        # Fill diagonal above with above_diag_color
        if above_diag_color is not None:
            for i in range(grid_size - 1):
                output_grid[i, i + 1] = above_diag_color

        return output_grid



