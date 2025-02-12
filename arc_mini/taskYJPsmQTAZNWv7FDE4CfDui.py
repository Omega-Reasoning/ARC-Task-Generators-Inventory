from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskYJPsmQTAZNWv7FDE4CfDuiGenerator (ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain with corrected key usage (using quotes)
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain three multi-colored (1-9) cells, each completely separated by empty (0) cells."
        ]
        # 2) Transformation reasoning chain remains the same
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling all empty (0) cells that are adjacent (up,down,left,right) to each colored cell with the respective cell color."
        ]
        # 3) Superclass init
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        """
        Creates an input grid of size (rows x cols) with exactly three distinct colored cells 
        (from 1..9) placed such that each is 8-way separated from the others.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)

        # Choose 3 distinct colors from 1..9
        colors = random.sample(range(1, 10), 3)

        placed_positions = []
        for color in colors:
            while True:
                r = random.randint(0, rows - 1)
                c = random.randint(0, cols - 1)
                # Ensure the new cell is 8-way separated from already placed colored cells
                if all(abs(r - pr) > 1 or abs(c - pc) > 1 for (pr, pc) in placed_positions):
                    grid[r, c] = color
                    placed_positions.append((r, c))
                    break

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Copies the input grid and fills all empty (0) cells that are adjacent
        (up, down, left, right) to each colored cell with that cell's color.
        """
        rows, cols = grid.shape
        out_grid = grid.copy()

        # For each colored cell, fill its 4 neighbors if they are empty
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    color = grid[r, c]
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and out_grid[nr, nc] == 0:
                            out_grid[nr, nc] = color
        return out_grid

    def create_grids(self):
        """
        Creates 3-4 training examples and 1 test example.
        Returns a tuple: (taskvars, train_test_data).
        """
        # Pick random grid dimensions between 6 and 15 (can be up to 30 if desired)
        rows = random.randint(6, 15)
        cols = random.randint(6, 15)
        taskvars = {
            'rows': rows,
            'cols': cols
        }

        # Randomly choose to create 3 or 4 training examples, and exactly 1 test example
        nr_train = random.choice([3, 4])
        nr_test = 1

        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        return taskvars, train_test_data

