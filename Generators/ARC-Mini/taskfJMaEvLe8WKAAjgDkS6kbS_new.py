# example_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optional but encouraged: we can use these libraries to help generate inputs.
from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects

class TaskfJMaEvLe8WKAAjgDkS6kbS_newGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input reasoning chain:
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They only contain three {color('cell_color')} cells and empty cells.",
            "The position of the {color('cell_color')} cells varies across examples."
        ]

        # 2. Transformation reasoning chain:
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid and create a rectangular block of {color('rectangle_color')} color.",
            "The block is placed one row below the row of the highest {color('cell_color')} cell.",
            "The width of the {color('rectangle_color')} rectangular block extends from the leftmost to the rightmost {color('cell_color')} cell, while its length is always two.",
            "The rectangular block may overlap {color('cell_color')} cells if necessary."
        ]

        # 3. Call superclass init:
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
        * Random size between 5 and 30
        * Exactly three cells of color taskvars['cell_color']
        * They should not all be in the same row or same column
        * The highest cell (smallest row index) is placed so that we have at least 2 rows available below it
        """
        cell_color = taskvars['cell_color']

        # Randomly choose grid dimensions
        height = random.randint(5, 30)
        width = random.randint(5, 30)

        # We'll keep trying until we find a valid placement
        def generate_grid():
            grid = np.zeros((height, width), dtype=int)
            # Place three distinct positions
            positions = set()
            while len(positions) < 3:
                r = random.randint(0, height - 1)
                c = random.randint(0, width - 1)
                positions.add((r, c))
            positions = list(positions)

            # Check not all in same row or same column
            rows = [p[0] for p in positions]
            cols = [p[1] for p in positions]

            # Also check that the highest row (min row) is <= height-3 (so there's space for the 2-row block)
            if len(set(rows)) == 1:  # all in the same row
                return None
            if len(set(cols)) == 1:  # all in the same column
                return None
            if min(rows) > height - 3:
                return None

            # If valid, fill with cell_color
            for (r, c) in positions:
                grid[r, c] = cell_color
            return grid

        # Use retry to keep generating until we get a valid grid
        grid = retry(
            generator=generate_grid,
            predicate=lambda g: g is not None
        )
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        1. Output grid is same size.
        2. Create a rectangle of color taskvars['rectangle_color'].
        3. Place its top row one below the highest cell_color cell in the input grid.
        4. Rectangle width = from leftmost to rightmost cell_color cell, height = 2.
        """
        cell_color = taskvars['cell_color']
        rectangle_color = taskvars['rectangle_color']

        output = np.copy(grid)

        # Find all (row, col) positions of the cell_color
        cell_positions = [(r, c) 
                          for r in range(grid.shape[0]) 
                          for c in range(grid.shape[1]) 
                          if grid[r, c] == cell_color]
        
        # Identify highest row, leftmost column, rightmost column of these cells
        highest_row = min(r for r, c in cell_positions)
        leftmost_col = min(c for r, c in cell_positions)
        rightmost_col = max(c for r, c in cell_positions)

        # The rectangle is placed starting at highest_row+1 for 2 rows
        # so rows = [highest_row+1, highest_row+2]
        for row in range(highest_row + 1, highest_row + 3):
            for col in range(leftmost_col, rightmost_col + 1):
                output[row, col] = rectangle_color

        return output

    def create_grids(self):
        """
        Creates:
         - Random distinct cell_color and rectangle_color in [1..9].
         - A random number of train examples [3..6].
         - Exactly 1 test example.
         - Uses self.create_grids_default(...) to produce the final dictionary.
        """
        # Randomly choose two distinct colors
        cell_color = random.randint(1, 9)
        rectangle_color = random.randint(1, 9)
        while rectangle_color == cell_color:
            rectangle_color = random.randint(1, 9)

        # Put them in a dictionary for variable substitution
        taskvars = {
            'cell_color': cell_color,
            'rectangle_color': rectangle_color
        }

        nr_train = random.randint(3, 6)
        nr_test = 1

        # Use the default helper to create train/test
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        return taskvars, train_test_data



