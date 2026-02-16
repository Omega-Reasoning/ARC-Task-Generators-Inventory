import numpy as np
import random

# Importing the necessary utility libraries
from Framework.input_library import create_object, Contiguity
from Framework.transformation_library import find_connected_objects

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

class TaskGqqVrV4CnAUNRgZGQZCUQKGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain vertical strips of {color('object_color')} color, where each strip is a vertically connected group of cells with lengths varying between 2 and {vars['rows']}.",
            "These strips are positioned in the first and last columns, starting from the first row and extending downward."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids have the same number of rows as the input grids, but they always have more columns than the input grids.",
            "The output grid is created by widening the {color('object_color')} vertical strips so that each strip becomes a square, with its width equal to its height.",
            "The newly constructed squares maintain the original position of the strips by having the same number of empty (0) columns between them and preserving the same number of empty cells below as before."
        ]
        
        # 3) Initialize the parent class with the reasoning chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Creates an input grid according to the reasoning chain.
        Places two vertical strips of color in the first and last columns, ensuring they have different lengths.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        color = taskvars['object_color']

        # Ensure the strips have different lengths
        while True:
            L1 = random.randint(2, rows)
            L2 = random.randint(2, rows)
            if L1 != L2 and (L1 > 1 or L2 > 1):
                break

        # Create the grid
        grid = np.zeros((rows, cols), dtype=int)

        # Place the first vertical strip (L1 length) in the first column
        for r in range(L1):
            grid[r, 0] = color

        # Place the second vertical strip (L2 length) in the last column
        for r in range(L2):
            grid[r, cols - 1] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transforms the input grid into the output grid by widening the vertical strips into squares.
        """
        rows, cols = grid.shape
        color = taskvars['object_color']

        # Find the lengths of the vertical strips
        L1 = sum(1 for r in range(rows) if grid[r, 0] == color)
        L2 = sum(1 for r in range(rows) if grid[r, cols - 1] == color)

        # Calculate new columns: L1 + original gap + L2
        new_cols = L1 + (cols - 2) + L2
        output = np.zeros((rows, new_cols), dtype=int)

        # Place the left square (size L1 x L1)
        for r in range(L1):
            for c in range(L1):
                output[r, c] = color

        # Preserve the empty space in between

        # Place the right square (size L2 x L2)
        right_start_col = L1 + (cols - 2)
        for r in range(L2):
            for c in range(L2):
                output[r, right_start_col + c] = color

        return output

    def create_grids(self) -> tuple:
        """
        Creates multiple train/test grids with different configurations.
        Ensures high diversity in training examples.
        """
        # Randomize color
        object_color = random.randint(1, 9)
        
        # Randomize rows and columns within valid limits
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        # Task variables to be used in template instantiation
        taskvars = {
            'object_color': object_color,
            'rows': rows,
            'cols': cols
        }

        # Number of training examples (2 or 3) and 1 test example
        nr_train_examples = random.randint(2, 3)
        nr_test_examples = 1

        # Generate train/test grids using the default helper
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data


