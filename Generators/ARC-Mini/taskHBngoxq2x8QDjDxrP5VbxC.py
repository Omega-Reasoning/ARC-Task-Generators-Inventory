from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optionally use these libraries for input creation (but NOT in transform_input)
from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects

class TaskHBngoxq2x8QDjDxrP5VbxCGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Store input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They are completely filled with {color('cell_color1')} and {color('cell_color2')} cells.",
            "Some of the {color('cell_color1')} cells should be diagonally connected to at least one {color('cell_color2')} cell."
        ]
        # 2. Store transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and changing all {color('cell_color1')} cells to {color('cell_color3')} that are diagonally connected to at least one {color('cell_color2')} cell."
        ]
        # 3. Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Randomize distinct colors in [1..9].
        color_choices = list(range(1, 10))
        random.shuffle(color_choices)
        cell_color1, cell_color2, cell_color3 = color_choices[:3]

        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2,
            "cell_color3": cell_color3,
        }

        # Randomly pick the number of training examples between 3 and 6, plus 1 test.
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1

        # Use the default "create_grids_default" method to produce train/test data
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid fully filled with cell_color1 or cell_color2,
        ensuring there's at least one cell_color1 diagonally adjacent to cell_color2
        and at least one cell_color1 not diagonally adjacent to any cell_color2.
        """

        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]

        def generate_grid():
            # Random grid size between 3x3 and 15x15
            rows = random.randint(3, 30)
            cols = random.randint(3, 30)
            grid = np.zeros((rows, cols), dtype=int)

            # Probability of using cell_color1 vs cell_color2
            p = random.uniform(0.3, 0.7)  # enforce mixture of colors
            for r in range(rows):
                for c in range(cols):
                    if random.random() < p:
                        grid[r, c] = cell_color1
                    else:
                        grid[r, c] = cell_color2
            return grid

        def is_valid_grid(grid):
            # Check how many cell_color1 cells are diagonally adjacent to cell_color2
            rows, cols = grid.shape
            diagonal_adj_count = 0
            total_color1_count = 0

            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] == cell_color1:
                        total_color1_count += 1
                        # Check the four diagonal neighbors
                        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < rows and 0 <= cc < cols:
                                if grid[rr, cc] == cell_color2:
                                    diagonal_adj_count += 1
                                    break  # Found at least one diagonal neighbor

            # We need:
            # 1) Some color1 cells with diagonal adjacency => diagonal_adj_count >= 1
            # 2) Some color1 cells with no diagonal adjacency => diagonal_adj_count < total_color1_count
            # Also ensure at least one cell_color2 is present
            has_color2 = np.any(grid == cell_color2)
            return (diagonal_adj_count >= 1 
                    and diagonal_adj_count < total_color1_count 
                    and has_color2 
                    and total_color1_count > 0)

        # Use the retry mechanism to generate a valid grid within the constraints
        valid_grid = retry(generate_grid, is_valid_grid, max_attempts=100)
        return valid_grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        For every cell_color1 cell that is diagonally connected to 
        at least one cell_color2 cell, change it to cell_color3.
        """
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]
        cell_color3 = taskvars["cell_color3"]

        rows, cols = grid.shape
        output_grid = np.copy(grid)

        # For each cell_color1 cell, check diagonals for cell_color2
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == cell_color1:
                    # Check diagonals
                    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            if grid[rr, cc] == cell_color2:
                                output_grid[r, c] = cell_color3
                                break
        return output_grid


