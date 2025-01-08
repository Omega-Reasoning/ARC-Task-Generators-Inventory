import numpy as np
import random
from typing import Dict, Any, Tuple
# Required imports from your instructions
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
# Optional, but recommended libraries
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects

class TasktiUDYJ9f4sTjKd7DSWV2qgPGenerator(ARCTaskGenerator):
    def __init__(self):
        # -----------------------
        # 1) Input Reasoning Chain (verbatim from your specification)
        # -----------------------
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each input grid contains a single C-shaped object made of 4-way connected cells of the same color (1-9).",
            "The C-shaped object can be described as an object of the form [[c,c],[c,0],[c,c]] for a color c.",
            "All other cells are empty (0)."
        ]

        # -----------------------
        # 2) Transformation Reasoning Chain (verbatim from your specification)
        # -----------------------
        transformation_reasoning_chain = [
            "To construct the output grid, start with a zero-filled grid.",
            "For each cell at position (i, j) in the input grid, calculate its new position (j, n-1-i) for a 90° clockwise rotation.",
            "Copy the cell from the input grid to its calculated new position in the output grid."

        ]

        # -----------------------
        # 3) Call super().__init__()
        # -----------------------
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Create the dictionary of task variables (vars) and the train/test data.
        We pick a random even grid size between 5 and 10,
        then produce 3 training pairs and 1 test pair by default.
        """
        # Pick an even grid size in the range [5..10]
        # The requirement is that grid_size is even and in [5..10].
        # Hence valid even choices are 6, 8, 10.
        grid_size = random.choice([6, 8, 10])
        taskvars = {"grid_size": grid_size}

        # Create 3 train pairs and 1 test pair using the default convenience method
        # which calls create_input() / transform_input() for each pair.
        train_test_data = self.create_grids_default(nr_train_examples=3,
                                                    nr_test_examples=1,
                                                    taskvars=taskvars)
        return taskvars, train_test_data

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an NxN grid with a single C-shaped object of random color
        placed at a random position such that it fits inside the grid.
        """
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Pick a random color for the C-shape (1..9)
        c = random.randint(1, 9)

        # The C shape is 3 rows by 2 columns
        shape_height, shape_width = 3, 2
        
        # Compute a random top-left position (row_off, col_off) where it can fit
        # so that  0 <= row_off <= grid_size - 3, 0 <= col_off <= grid_size - 2
        row_off = random.randint(0, grid_size - shape_height)
        col_off = random.randint(0, grid_size - shape_width)

        # The pattern of the C-shape:
        #   (r_off+0, c_off+0) = c, (r_off+0, c_off+1) = c
        #   (r_off+1, c_off+0) = c
        #   (r_off+2, c_off+0) = c, (r_off+2, c_off+1) = c
        grid[row_off + 0, col_off + 0] = c
        grid[row_off + 0, col_off + 1] = c
        grid[row_off + 1, col_off + 0] = c
        grid[row_off + 2, col_off + 0] = c
        grid[row_off + 2, col_off + 1] = c

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Rotate the grid 90° clockwise. Per the transformation reasoning chain:
        (i, j) -> (j, n-1-i).
        """
        # One approach: use np.rot90 with k=-1 (since the default is counterclockwise).
        # Alternatively, do the manual indexing. We choose np.rot90 for brevity.
        return np.rot90(grid, k=-1)

