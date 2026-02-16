from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, List


class Taskktvuj6nA3Mo2N6bzjURjbFGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains exactly one colored cell, while the remaining cells are empty.",
            "The color and position of the colored cell vary across examples."
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by creating a diamond-shaped object using the same color as the single colored cell in the input.",
            "The diamond is formed by placing the colored cell at the middle position of the first row, last row, first column, and last column (i.e., at index {(vars['grid_size'] - 1)//2}).",
            "Then fill all cells that satisfy |r - mid| + |c - mid| ≤ mid, which produces a solid diamond, where r and c are row and column indices respectively.",
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # ----------------------------
    # INPUT CREATION
    # ----------------------------
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid with exactly one non-zero colored cell.
        The color is chosen to be unique across all examples (train + test).
        """
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Prefer index provided by framework; otherwise, use a pointer we maintain.
        if 'example_index' in gridvars:
            idx = gridvars['example_index']
        else:
            # Fallback if the framework doesn't pass example_index
            idx = taskvars.setdefault('_color_ptr', 0)
            taskvars['_color_ptr'] = idx + 1

        # Use available colors directly without storing as task variable
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        color = available_colors[idx % len(available_colors)]

        row = random.randint(0, grid_size - 1)
        col = random.randint(0, grid_size - 1)
        grid[row, col] = color
        return grid

    # ----------------------------
    # TRANSFORMATION
    # ----------------------------
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        output_grid = np.zeros((grid_size, grid_size), dtype=int)

        # Find the colored cell in the input
        rows, cols = np.where(grid != 0)
        if rows.size == 0:
            return output_grid  # safety, though generator always places one cell

        color = grid[rows[0], cols[0]]
        mid = (grid_size - 1) // 2

        # Fill solid diamond: |r - mid| + |c - mid| <= mid
        for r in range(grid_size):
            for c in range(grid_size):
                if abs(r - mid) + abs(c - mid) <= mid:
                    output_grid[r, c] = color

        return output_grid

    # ----------------------------
    # TASK CREATION
    # ----------------------------
    def create_grids(self):
        """
        - Picks an odd grid size in [5, 30].
        - Creates 3–5 training examples and 1 test example.
        - Ensures every example uses a UNIQUE color for its single input cell.
        """
        valid_sizes = [s for s in range(5, 31) if s % 2 == 1]
        grid_size = random.choice(valid_sizes)

        num_train = random.randint(3, 5)
        num_test = 1
        total_examples = num_train + num_test

        # We have colors 1..9 available (ARC colors excluding 0 background)
        available_colors = 9  # Number of available colors
        if total_examples > available_colors:
            raise ValueError(
                f"Not enough unique colors for {total_examples} examples; "
                f"only {available_colors} available."
            )

        taskvars = {
            'grid_size': grid_size,
        }

        # create_grids_default is assumed to pass gridvars with 'example_index' in order.
        train_test_data = self.create_grids_default(num_train, num_test, taskvars)
        return taskvars, train_test_data



