from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task79Wi23KvwkNQf83DTdS5xgGenerator(ARCTaskGenerator):

    def __init__(self):
        # 1) Input reasoning chain (verbatim from the instructions)
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They only contain two horizontally connected {color('cell_color1')} and {color('cell_color2')} cells, with the remaining cells being empty (0).",
            "In the connected pair, the {color('cell_color1')} cell is on the left and the {color('cell_color2')} cell is on the right.",
            "The position of these connected cells can vary across examples."
        ]
        
        # 2) Transformation reasoning chain (verbatim from the instructions)
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and checking for an empty cell to the right of the {color('cell_color2')} cell.",
            "If the {color('cell_color2')} cell has an empty cell to its right, the empty cell is filled with {color('cell_color1')} cell; otherwise, it remains unchanged."



        ]
        
        # 3) Call superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid of random size between 5x5 and 30x30 that contains exactly
        two horizontally connected cells: cell_color1 (left) and cell_color2 (right).
        If gridvars['no_right_space'] is True, place these two cells at the rightmost column
        so that the cell_color2 has no empty cell to its right.
        Otherwise, place them such that cell_color2 has at least one empty cell to the right.
        """
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']
        no_right_space = gridvars['no_right_space']  # bool

        # Randomly choose grid dimensions
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        grid = np.zeros((rows, cols), dtype=int)

        # We need to place two horizontally connected colored cells.
        # If no_right_space is True, we'll place them in the last two columns.
        # If no_right_space is False, we'll ensure there's at least one empty column to the right of cell_color2.
        if no_right_space:
            # Place them so that cell_color2 is in the last column
            # Randomly choose which row
            r = random.randint(0, rows - 1)
            # The pair will occupy columns (cols-2) and (cols-1).
            # Left cell is cell_color1, right cell is cell_color2
            grid[r, cols - 2] = cell_color1
            grid[r, cols - 1] = cell_color2
        else:
            # We want at least one empty cell to the right
            # Let's pick a random row
            r = random.randint(0, rows - 1)
            # We want to place them in columns c and c+1, ensuring c+1 < cols-1
            # so that c+2 <= cols-1 -> c <= cols-3
            c = random.randint(0, cols - 3)
            grid[r, c] = cell_color1
            grid[r, c + 1] = cell_color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation chain:
         * If the cell_color2 cell has an empty cell to its right (within bounds),
           fill that cell with cell_color1.
         * Otherwise, leave the grid unchanged.
        """
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']

        # Make a copy
        out_grid = grid.copy()

        # Find the single cell_color2 cell
        loc = np.where(out_grid == cell_color2)
        if len(loc[0]) == 0:
            # Unexpected but let's be defensive. Do nothing if we can't find cell_color2.
            return out_grid

        # We expect exactly one location for cell_color2; but let's handle if there's more than one by taking the first
        r = loc[0][0]
        c = loc[1][0]

        # Check if there's an empty cell to the right
        if c + 1 < out_grid.shape[1]:
            if out_grid[r, c + 1] == 0:
                # Fill it with cell_color1
                out_grid[r, c + 1] = cell_color1

        return out_grid

    def create_grids(self) -> tuple[dict, TrainTestData]:
        """
        We create:
          - A random choice of distinct cell_color1 and cell_color2 in [1..9].
          - 4 training grids: 2 examples with no_right_space=True, 2 examples with no_right_space=False.
          - 2 test grids: 1 example with no_right_space=True, 1 with no_right_space=False.
        Return (taskvars, train_test_data).
        """
        # 1) Pick distinct cell colors
        cell_color1 = random.randint(1, 9)
        cell_color2 = random.randint(1, 9)
        while cell_color2 == cell_color1:
            cell_color2 = random.randint(1, 9)

        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2
        }

        # 2) Create training data
        train_pairs = []
        # We'll create 2 examples for no_right_space = True
        for _ in range(2):
            input_grid = self.create_input(taskvars, {"no_right_space": True})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        # We'll create 2 examples for no_right_space = False
        for _ in range(2):
            input_grid = self.create_input(taskvars, {"no_right_space": False})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # 3) Create test data:
        # one with no_right_space=True, one with no_right_space=False
        test_pairs = []
        input_grid = self.create_input(taskvars, {"no_right_space": True})
        output_grid = self.transform_input(input_grid, taskvars)
        test_pairs.append(GridPair(input=input_grid, output=output_grid))

        input_grid = self.create_input(taskvars, {"no_right_space": False})
        output_grid = self.transform_input(input_grid, taskvars)
        test_pairs.append(GridPair(input=input_grid, output=output_grid))

        train_test_data = TrainTestData(train=train_pairs, test=test_pairs)
        return taskvars, train_test_data



