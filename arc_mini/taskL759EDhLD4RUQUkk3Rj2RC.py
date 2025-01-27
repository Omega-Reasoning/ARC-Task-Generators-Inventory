# my_arc_task_generator.py
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskL759EDhLD4RUQUkk3Rj2RCGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (exactly as given)
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain only several {color('cell_color')} cells along with empty cells.",
            "The {color('cell_color')} cells are arranged so that there are at least two empty cells directly below each {color('cell_color')} cell."
        ]
        # 2) Transformation reasoning chain (exactly as given)
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and expanding each {color('cell_color')} cell vertically downward by one cell, maintaining the same color."
        ]
        # 3) Call super-constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        1. Randomly pick the color for 'cell_color' (from 1..9).
        2. Randomly pick how many train examples (3..6).
        3. Generate each train example input with create_input() and transform it.
        4. Do the same for a single test example.
        5. Return the dictionary of task variables and the set of train/test data.
        """
        taskvars = {}

        # Randomly choose cell_color from 1..9
        taskvars["cell_color"] = random.randint(1, 9)

        # Decide how many training examples (3..6)
        nr_train = random.randint(3, 6)
        # We'll produce exactly 1 test example
        nr_test = 1

        # Collect train grids
        train_data = []
        for _ in range(nr_train):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            train_data.append(GridPair(input=inp, output=out))

        # Collect test grids
        test_data = []
        for _ in range(nr_test):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            test_data.append(GridPair(input=inp, output=out))

        return taskvars, TrainTestData(train=train_data, test=test_data)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Creates an input grid with the following constraints:
          * Grid size between 5x5 and 30x30 (for variety, we limit ourselves to ~5..10 in each dimension).
          * Only 'cell_color' (non-zero) and 0 (empty).
          * At least 3 cells of color 'cell_color'.
          * Each placed color cell has exactly 2 empty rows below it.
          * No color cells can be in the last two rows.
          * In each column, color cells are separated by at least 2 empty rows
            (i.e., if a color cell is at row r, another color cell in the same column
             can only be placed at row >= r+3).
        """
        cell_color = taskvars["cell_color"]

        # Pick random dimensions:
        height = random.randint(5, 10)
        width = random.randint(5, 10)
        grid = np.zeros((height, width), dtype=int)

        # We want at least 3 colored cells
        # Maximum feasible is (height-2)*width if every possible row was used
        max_places = min((height - 2) * width, 10)  # also cap at 10 for variety
        nr_cells_to_place = random.randint(3, max_places)

        placed_cells = 0

        # We'll try up to 'nr_cells_to_place' times to place color cells.
        # Constraint summary for a candidate (r, c):
        #   1) r <= height-3 (so we have r+1, r+2 within the grid)
        #   2) grid[r,c] == 0 and grid[r+1,c] == 0 and grid[r+2,c] == 0
        #   3) no existing color cell in the same column c at row r' where |r - r'| < 3
        #      (so there's a gap of at least 2 empty rows).
        for _ in range(nr_cells_to_place):
            for __ in range(100):  # up to 100 tries to place a single color cell
                r = random.randint(0, height - 3)  # can't be in the last two rows
                c = random.randint(0, width - 1)

                # Check next two rows are empty
                if grid[r, c] != 0 or grid[r+1, c] != 0 or grid[r+2, c] != 0:
                    continue

                # Check vertical spacing in the same column c
                # We want no color cells in rows [r-2, r+2]
                # i.e. for any row r' in that col where grid[r', c]==cell_color,
                # we require |r - r'| >= 3.
                too_close = False
                for rr in range(max(0, r-2), min(height, r+3)):
                    if grid[rr, c] == cell_color:
                        too_close = True
                        break
                if too_close:
                    continue

                # If all checks pass, place the cell
                grid[r, c] = cell_color
                placed_cells += 1
                break

        # Enforce at least 3 color cells. If not enough were placed, attempt forced placement
        needed = 3 - placed_cells
        while needed > 0:
            attempts_left = 100
            placed_successfully = False
            while attempts_left > 0:
                attempts_left -= 1
                r = random.randint(0, height - 3)
                c = random.randint(0, width - 1)

                if (grid[r, c] != 0 or 
                    grid[r+1, c] != 0 or 
                    grid[r+2, c] != 0):
                    continue

                # Check vertical spacing in col c
                too_close = False
                for rr in range(max(0, r-2), min(height, r+3)):
                    if grid[rr, c] == cell_color:
                        too_close = True
                        break
                if too_close:
                    continue

                # Place
                grid[r, c] = cell_color
                placed_cells += 1
                placed_successfully = True
                break

            if not placed_successfully:
                # Can't place more color cells without violating constraints
                break
            needed -= 1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        The transformation:
          * Copy the input grid.
          * For each cell that has color 'cell_color', set the cell immediately below
            it to the same color (i.e., expand downward by one row).
        """
        cell_color = taskvars["cell_color"]
        output_grid = np.copy(grid)

        # For each position that is cell_color, color the cell below
        height, width = grid.shape
        for r in range(height - 1):  # avoid bottom row
            for c in range(width):
                if grid[r, c] == cell_color:
                    output_grid[r+1, c] = cell_color

        return output_grid


