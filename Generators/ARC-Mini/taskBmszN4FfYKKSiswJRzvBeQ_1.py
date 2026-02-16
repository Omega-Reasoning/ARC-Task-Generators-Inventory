import numpy as np
import random

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

# (Optional) If you need certain functions from Framework.input_library or transformation_library,
# you can import them here. Below, we only rely on direct NumPy calls for simplicity.

class TaskBmszN4FfYKKSiswJRzvBeQ_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They are completely filled with multi-colored (1-9) cells."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "Output grids are of size {vars['grid_size']}x1.",
            "They are constructed by copying the main diagonal (top-left to bottom-right) cells and pasting it into the first column of the same row."
        ]

        # 3) Call parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an NxN grid, where N = taskvars['grid_size']. All cells
        are randomly chosen from 1..9, ensuring that no two consecutive
        diagonal cells share the same color, and that the entire grid
        has at least two distinct colors.
        """
        N = taskvars['grid_size']

        while True:
            # Random NxN grid from {1..9}
            grid = np.random.randint(1, 10, size=(N, N))

            # Check diagonal constraint
            no_two_consecutive_diagonal_same = all(
                grid[i, i] != grid[i + 1, i + 1] 
                for i in range(N - 1)
            )
            if not no_two_consecutive_diagonal_same:
                continue

            # Ensure multi-colored: at least two distinct values
            unique_vals = np.unique(grid)
            if len(unique_vals) < 2:
                continue

            return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Given an NxN input grid, produce an Nx1 output grid where
        output[i,0] = grid[i,i].
        """
        N = taskvars['grid_size']
        out = np.zeros((N, 1), dtype=int)

        for i in range(N):
            out[i, 0] = grid[i, i]

        return out

    def create_grids(self):
        """
        1) Randomly choose grid_size in [5..30].
        2) Generate 3–6 train examples and 1 test example using
           create_grids_default().
        """
        grid_size = random.randint(5, 30)
        num_train = random.randint(3, 6)
        num_test = 1

        taskvars = {'grid_size': grid_size}

        # Create standard train and test sets
        train_test_data = self.create_grids_default(num_train, num_test, taskvars)

        return taskvars, train_test_data

