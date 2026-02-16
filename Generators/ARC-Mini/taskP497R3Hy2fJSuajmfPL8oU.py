import numpy as np
import random

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry

class TaskP497R3Hy2fJSuajmfPL8oUGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each input grid contains exactly two differently colored rectangles; the first occupies the first x columns and the second occupies columns x+1 to {vars['cols']}.",
            "No cells are empty.",
            "The sizes and colors of both rectangles vary across examples."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the two differently colored rectangles; the first occupies the first x columns and the second occupies columns x+1 to {vars['cols']}.",
            "The two rectangles are then swapped so that the second becomes the first and the first becomes the second."
        ]

        # 3) Call the superclass initializer
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create a grid that:
          - Has dimensions rows x cols from taskvars.
          - Contains exactly two rectangles: first occupies columns 0 to x-1, 
            second occupies columns x to cols-1.
          - Both rectangles have different colors and span all rows.
          - No empty cells.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']

        # Create a blank grid
        grid = np.zeros((rows, cols), dtype=int)

        # Choose the split point x (where first rectangle ends)
        # Ensure both rectangles have at least 1 column
        x = random.randint(1, cols - 1)

        # Choose two different colors
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)

        # Fill first rectangle: columns 0 to x-1
        grid[:, :x] = color1

        # Fill second rectangle: columns x to cols-1
        grid[:, x:] = color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input by swapping the two rectangles.
        """
        rows, cols = grid.shape
        output_grid = np.zeros((rows, cols), dtype=int)

        # Find the split point by looking for the color change
        split_point = 1
        first_color = grid[0, 0]
        for col in range(1, cols):
            if grid[0, col] != first_color:
                split_point = col
                break

        # Extract the two rectangles
        first_rect = grid[:, :split_point].copy()
        second_rect = grid[:, split_point:].copy()

        # Swap them: put second rectangle first, then first rectangle
        second_width = second_rect.shape[1]
        first_width = first_rect.shape[1]

        # Place second rectangle at the beginning
        output_grid[:, :second_width] = second_rect

        # Place first rectangle after the second
        output_grid[:, second_width:] = first_rect

        return output_grid

    def create_grids(self):
        """
        Creates multiple train input-output pairs and 1 test pair.
        Ensures variety in grid sizes and rectangle configurations.
        """
        # Choose random grid dimensions within constraints
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        taskvars = {'rows': rows, 'cols': cols}

        # Choose number of training examples (3-5 as specified)
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1

        # Use the default convenience method
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data

