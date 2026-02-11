import numpy as np
import random
from typing import Dict, Any
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
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid contains a single C-shaped object made of 4-way connected cells of the same color (1-9).",
            "The C-shaped object can be described as an object of the form [[c,c],[c,0],[c,c]] for a color c.",
            "All other cells are empty (0)."
        ]

        # -----------------------
        # 2) Transformation Reasoning Chain (verbatim from your specification)
        # -----------------------
        transformation_reasoning_chain = [
            "Output grids are of size {vars['cols']}x{vars['rows']}.",
            "The output grids are constructed by rotating the input grid 90° clockwise. This is done by calculating a new position for each cell: for a cell at position (i, j) in the input grid, its new position in the output grid is (j, {vars['rows'] - 1} - i).",
            "Copy the cell from the input grid to its calculated new position in the output grid."
        ]

        # -----------------------
        # 3) Call super().__init__()
        # -----------------------
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Create the dictionary of task variables (vars) and the train/test data.
        We pick a random grid size between 6 and 30 for both rows and cols,
        ensuring that rows != cols.
        """
        size_range = list(range(6, 31))
        rows = random.choice(size_range)
        cols = random.choice([x for x in size_range if x != rows])  # ensure cols != rows

        taskvars = {"rows": rows, "cols": cols}

        train_test_data = self.create_grids_default(nr_train_examples=3,
                                                    nr_test_examples=1,
                                                    taskvars=taskvars)
        return taskvars, train_test_data

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)

        # Pick a random color for the C-shape (1..9)
        c = random.randint(1, 9)

        # The C shape is 3 rows by 2 columns
        shape_height, shape_width = 3, 2

        # Compute a valid top-left placement offset
        row_off = random.randint(0, rows - shape_height)
        col_off = random.randint(0, cols - shape_width)

        # Place the C-shaped pattern
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
        return np.rot90(grid, k=-1)
