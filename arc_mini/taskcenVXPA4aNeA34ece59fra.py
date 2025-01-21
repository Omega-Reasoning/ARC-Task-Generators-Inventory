# diagonal_duplicate_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optionally use functions from the transformation library (though this task doesn't strictly need them)
# from transformation_library import ...

# We do use input_library for generating the input grid logic if we want, but here we'll implement it manually
# from input_library import ...

class TaskcenVXPA4aNeA34ece59fraGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Define the input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size ({vars['grid_size']}x{vars['grid_size']}).",
            "They contain a single-colored (1-9) main diagonal (from top-left to bottom-right), along with other colored (1-9) and empty cells."
        ]

        # 2) Define the transformation reasoning chain
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid and duplicate each main diagonal cell by placing a copy directly to its right, even if the cell to the right is already filled."
        ]

        # 3) Pass these to the superclass along with an empty taskvars_definitions (as required)
        taskvars_definitions = {}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        """
        Create a square grid of size grid_size x grid_size.
        Fill the main diagonal with a single color (diagonal_color).
        Then place at least four non-diagonal colored cells (distinct from diagonal_color),
        each separated by at least one empty cell, and ensure that exactly one of them
        is positioned immediately to the right of some diagonal cell.
        """
        grid_size = taskvars["grid_size"]
        diagonal_color = taskvars["diagonal_color"]

        # Create an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Fill the main diagonal
        for i in range(grid_size):
            grid[i, i] = diagonal_color

        # Possible colors for non-diagonal cells
        non_diagonal_colors = [c for c in range(1, 10) if c != diagonal_color]

        # We must ensure that at least one cell is placed to the right of a diagonal cell
        # Choose a diagonal index i (except the last one so we have i+1 in range)
        i_forced = random.randint(0, grid_size - 2)
        forced_color = random.choice(non_diagonal_colors)
        grid[i_forced, i_forced + 1] = forced_color

        # Keep track of placed colored cells (row,col)
        placed_cells = [(i_forced, i_forced + 1)]

        # We need at least 3 more non-diagonal colored cells (total >= 4).
        # Optionally place more for variety.
        nr_extra = random.randint(3, 5)  # So total is between 4 and 6
        attempts = 0

        while len(placed_cells) < (1 + nr_extra):
            attempts += 1
            if attempts > 500:  # Safety to avoid infinite loops in pathological cases
                break
            r = random.randint(0, grid_size - 1)
            c = random.randint(0, grid_size - 1)

            # Must not be on the diagonal
            if r == c:
                continue

            # Must not already be filled
            if grid[r, c] != 0:
                continue

            # Must be at least 1 cell away from previously placed colored cells
            # (the task says "separated by ... empty cells", interpreted as not 4-neighbor adjacent)
            # i.e. no previously placed cell can be within a 1-step "Manhattan" adjacency
            def too_close(r0, c0, r1, c1):
                return abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1

            if any(too_close(r, c, rr, cc) for (rr, cc) in placed_cells):
                continue

            # Place the cell
            new_color = random.choice(non_diagonal_colors)
            grid[r, c] = new_color
            placed_cells.append((r, c))

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        According to the transformation reasoning chain:
        "copy the input grid and duplicate each main diagonal cell by placing a copy 
         directly to its right, even if that cell is already filled."
        """
        grid_size = taskvars["grid_size"]
        out_grid = grid.copy()
        
        for i in range(grid_size - 1):
            out_grid[i, i + 1] = out_grid[i, i]
        
        # If the diagonal ends at the bottom-right corner, there's no 'right' cell for the last diagonal
        return out_grid

    def create_grids(self):
        """
        Randomly choose task-level variables, create 3-6 training examples plus 1 test example.
        Return them in the required format.
        """
        # Choose random puzzle parameters
        grid_size = random.randint(5, 30)
        diagonal_color = random.randint(1, 9)

        taskvars = {
            "grid_size": grid_size,
            "diagonal_color": diagonal_color
        }

        # Number of training pairs
        nr_train_examples = random.randint(3, 6)
        # We'll generate exactly 1 test pair
        nr_test_examples = 1

        # Leverage ARCTaskGenerator.create_grids_default for building train/test data
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data


