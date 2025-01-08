# Filename: arc_agi_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

# We only import from input_library.py in create_input()
# and from transformation_library.py if needed in transform_input() (allowed).
# But we must not use input_library.py inside transform_input().
# Here we do not strictly need transformation_library for these particular transformations.
# If you wish to expand or detect objects, you could import relevant functions:
# from transformation_library import find_connected_objects, GridObject, GridObjects
# from input_library import create_object, retry, ...

class TaskCzSeQxLDe3AmrK2nfCZtcKGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input (observation) reasoning chain
        observation_chain = [
            "Input grids are of size n x n, where n is an odd number.",
            "Each input grid has one {color('fill_color1')} cell in the center of the grid and one {color('fill_color2')} cell directly above the {color('fill_color1')} cell.",
            "The remaining cells are empty (0)."
        ]
        
        # 2. Transformation reasoning chain
        reasoning_chain = [
            "To create the output grid, copy the input grid.",
            "Fill the empty (0) cells in the middle of the first and last columns and rows, as well as all empty cells above the {color('fill_color2')} cell, with {color('fill_color2')} color.",
            "Fill the remaining empty (0) cells in the middle row and column with {color('fill_color1')} color."
        ]
        
        # 3. Call super().__init__
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        We create 3 train grids and 1 test grid, each with a distinct odd size n between 5 and 19,
        and a single pair of fill_color1 and fill_color2 (from 1..9, different from each other).
        """
        # Possible odd sizes between 4..20 (but 4 is not odd, so we start from 5)
        possible_sizes = [5, 7, 9, 11, 13, 15, 17, 19]
        
        # Pick 4 distinct odd sizes for the 3 train pairs + 1 test pair
        random.shuffle(possible_sizes)
        sizes = possible_sizes[:4]
        
        # Pick two different colors between 1 and 9
        fill_color1 = random.randint(1, 9)
        fill_color2 = random.randint(1, 9)
        while fill_color2 == fill_color1:
            fill_color2 = random.randint(1, 9)
        
        # We'll store them as task variables:
        taskvars = {
            "fill_color1": fill_color1,
            "fill_color2": fill_color2,
        }
        
        # Now create 3 training examples and 1 test example
        train_examples = []
        for i in range(3):
            gridvars = {"n": sizes[i]}
            inp = self.create_input(taskvars, gridvars)
            outp = self.transform_input(inp, taskvars)
            train_examples.append(GridPair(input=inp, output=outp))
        
        test_examples = []
        gridvars_test = {"n": sizes[3]}
        inp_test = self.create_input(taskvars, gridvars_test)
        outp_test = self.transform_input(inp_test, taskvars)
        test_examples.append(GridPair(input=inp_test, output=outp_test))
        
        train_test_data = TrainTestData(train=train_examples, test=test_examples)
        return taskvars, train_test_data
    
    def create_input(self, taskvars, gridvars):
        """
        Create an n x n grid (n odd, between 5..19).
        Place:
          - fill_color1 at the center (row=n//2, col=n//2),
          - fill_color2 directly above it (row=n//2 -1, col=n//2),
        The rest is 0.
        """
        n = gridvars["n"]
        fill_color1 = taskvars["fill_color1"]
        fill_color2 = taskvars["fill_color2"]
        
        # Create empty grid
        grid = np.zeros((n, n), dtype=int)
        
        # Place fill_color1 in the center
        center = n // 2
        grid[center, center] = fill_color1
        
        # Place fill_color2 just above the center (if it exists, but n//2 -1 >= 0 for n>=3)
        if center - 1 >= 0:
            grid[center - 1, center] = fill_color2
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        According to the transformation chain:
          1) Copy the grid.
          2) Fill empty (0) cells in:
             - the middle of the first row & last row & first column & last column
             - all empty cells above the fill_color2 cell
             with fill_color2
          3) Fill the remaining empty (0) cells in the middle row and column with fill_color1.
        """
        fill_color1 = taskvars["fill_color1"]
        fill_color2 = taskvars["fill_color2"]
        
        out_grid = np.copy(grid)
        n = out_grid.shape[0]
        center = n // 2
        
        # 1) Identify the (row,col) of the fill_color2 cell in input
        #    By our creation rule, it's at (center-1, center) if that index is valid
        #    We'll double-check that cell in the input is indeed fill_color2
        #    If it's out of bounds, we skip (for smaller n, but it won't happen for n >= 3).
        r2, c2 = center - 1, center
        
        # 2) Fill empty cells in the middle of the first row/last row/first col/last col with fill_color2
        #    "in the middle" means col = center for row=0 and row=n-1
        #    and row = center for col=0 and col=n-1
        # First row middle cell
        if out_grid[0, center] == 0:
            out_grid[0, center] = fill_color2
        # Last row middle cell
        if out_grid[n-1, center] == 0:
            out_grid[n-1, center] = fill_color2
        # Middle row first column
        if out_grid[center, 0] == 0:
            out_grid[center, 0] = fill_color2
        # Middle row last column
        if out_grid[center, n-1] == 0:
            out_grid[center, n-1] = fill_color2
        
        #    ... also fill all empty cells above the fill_color2 cell's column with fill_color2
        if 0 <= r2 < n and 0 <= c2 < n:
            # fill all 0 cells in the same column c2 for rows < r2
            for rr in range(r2):
                if out_grid[rr, c2] == 0:
                    out_grid[rr, c2] = fill_color2
        
        # 3) Fill the remaining empty (0) cells in the middle row and middle column with fill_color1
        # Middle row
        for col in range(n):
            if out_grid[center, col] == 0:
                out_grid[center, col] = fill_color1
        # Middle column
        for row in range(n):
            if out_grid[row, center] == 0:
                out_grid[row, center] = fill_color1
        
        return out_grid


