# my_row_major_color_filling_task.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
class TaskXzUjuEvzsofWzjW3Jy2vY2Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialize the input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each grid is filled with multi-colored cells, with one color being more significant than the others.",
            "There are no empty (0) cells."
        ]

        # 2) Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and analyzing each row to determine the most frequent cell color.",
            "Each row is then filled entirely with its most frequent color."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates train and test grids for our ARC task.

        1) Randomly choose an even grid_size between 6 and 30 (inclusive).
        2) Generate 3-6 training examples and 1 test example.
        3) Return the dictionary of task variables and the train/test data.
        """

        # 1. Pick an even grid size between 6 and 30
        possible_sizes = [s for s in range(6, 31) if s % 2 == 0]
        grid_size = random.choice(possible_sizes)

        # Create the dictionary of task variables
        # (we only need to store grid_size here)
        taskvars = {
            'grid_size': grid_size
        }

        # 2. Generate 3-6 train examples, 1 test example
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1

        # If you do not need special logic across different train grids
        # beyond 'grid_size', you can use the create_grids_default helper:
        train_test_data = self.create_grids_default(
            nr_train_examples,
            nr_test_examples,
            taskvars
        )

        # 3. Return the task variables and the train/test grids
        return taskvars, train_test_data

    def create_input(self,
                     taskvars,
                     gridvars) -> np.ndarray:
        """
        Creates an input grid according to the input reasoning chain.

        Constraints:
        * Grid is of size grid_size x grid_size
        * No cell is 0 (empty)
        * Each row has at least 3 distinct colors
        * In each row, there is a dominant color appearing at least 1 + (grid_size // 2) times

        We implement these constraints row by row.
        """
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)

        min_dominant_count = 1 + (grid_size // 2)
        # Colors can be 1..9
        available_colors = list(range(1, 10))

        for r in range(grid_size):
            # Pick a random dominant color
            dominant_color = random.choice(available_colors)

            # Fill row with that dominant color at least min_dominant_count times
            row_colors = [dominant_color] * min_dominant_count

            # Remaining cells in the row
            remaining_cells = grid_size - min_dominant_count

            # We need at least 2 other distinct colors in the row
            # (to ensure at least 3 distinct colors including dominant_color)
            other_colors = random.sample(available_colors, k=2)
            # If the chosen 2 colors include the dominant_color, pick new ones
            while dominant_color in other_colors:
                other_colors = random.sample(available_colors, k=2)

            # Fill the remainder of the row with a random mix of those 2 other colors
            # so that each row ends up with at least 3 distinct colors
            row_colors.extend(random.choices(other_colors, k=remaining_cells))

            # Shuffle the row so the dominant color isn't just in the front
            random.shuffle(row_colors)
            grid[r, :] = row_colors

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transforms the input grid according to the transformation reasoning chain:
          1) Copy the input grid.
          2) For each row, find its most frequent color.
          3) Fill the entire row with that color.
        """
        grid_size = grid.shape[0]
        output_grid = grid.copy()

        for r in range(grid_size):
            row = output_grid[r, :]
            # Determine the most frequent color in this row
            # (any tie is fine; use max on counts)
            unique, counts = np.unique(row, return_counts=True)
            dominant_color = unique[np.argmax(counts)]
            # Fill the row with the dominant color
            output_grid[r, :] = dominant_color

        return output_grid

