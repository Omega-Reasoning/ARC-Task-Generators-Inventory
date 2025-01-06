# my_arc_generator.py
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring   # optionally use for more interesting randomization
import numpy as np
import random

class Tasktaskf6gsVEQtdb9ujWw6JS4ud2Generator(ARCTaskGenerator):
    def __init__(self):
        """
        Our generator. We store the input and transformation reasoning chains (templates) 
        and then call the parent constructor with those chains.
        """
        # 1) Copy the input reasoning chain from the problem statement
        input_reasoning_chain = [
            "Input grids are of size {vars['n']}x{vars['m']}.",
            "Input grids contain cells of different colors from (0-9)."
        ]

        # 2) Copy the transformation reasoning chain from the problem statement
        transformation_reasoning_chain = [
            # Note: The problem statement has a likely typo for the second dimension 
            # but the specification text clarifies it's (2*m - 1). We'll correct it here:
            "To construct the output grid; add an empty (0) column between each column of the input grid "
            "and add an empty (0) row between each row of the input grid.",
            "Keep the color of the original cells preserved."
        ]
        
        # 3) Call the parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars,
                     gridvars) -> np.ndarray:
        """
        Create an input grid of size n x m with random cell values in [0..9].
        """
        n = taskvars['n']
        m = taskvars['m']
        
        # Here we simply do a uniform random assignment of colors in [0..9].
        # If desired, you can call a utility from input_library to create more interesting patterns.
        # e.g. use random_cell_coloring on an initially empty grid
        grid = np.random.randint(low=0, high=10, size=(n, m))
        
        return grid

    def transform_input(self,
                        grid: np.ndarray,
                        taskvars) -> np.ndarray:
        """
        Transform the input grid according to the reasoning chain:
         1) The output size is (2n-1) x (2m-1).
         2) Insert an empty (0) row between each row of the input grid,
            and an empty (0) column between each column of the input grid.
         3) Copy the original cell colors into their positions at (2*r, 2*c).
        """
        n, m = grid.shape
        out_n = 2 * n - 1
        out_m = 2 * m - 1
        
        # Initialize output with zeros
        output_grid = np.zeros((out_n, out_m), dtype=int)

        # Place the original cells at (2*r, 2*c)
        for r in range(n):
            for c in range(m):
                output_grid[2*r, 2*c] = grid[r, c]
        
        return output_grid

    def create_grids(self):
        """
        Create 3-6 train grids and 1 test grid, each of size n x m (with n, m randomly in [8..15]), 
        then store them in a train/test dictionary. We also return the task variables 
        to instantiate the reasoning chain templates.
        """
        # Pick random n and m ensuring output < 30x30 => (2n-1) and (2m-1) <= 30
        n = random.randint(4, 10)
        m = random.randint(4, 10)
        
        taskvars = {
            'n': n,
            'm': m
        }

        # We choose a random number of train examples from 3 to 6
        nr_train = random.randint(3, 6)
        nr_test = 1  # We'll just create 1 test example

        # Using the built-in helper that just calls create_input() and transform_input()
        data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, data


