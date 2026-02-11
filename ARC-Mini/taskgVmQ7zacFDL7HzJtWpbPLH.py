# arc_task_generator_custom.py

import random
import numpy as np
from typing import Dict, Any, Tuple, List

# --- IMPORTS FROM THE INSTRUCTIONS ---
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects  # Potentially unused, just an example
from input_library import retry  # Potentially unused
# We do not use input_library.py in transform_input(), but we may use it in create_input().
# The instructions specifically said we can optionally use them.

class TaskgVmQ7zacFDL7HzJtWpbPLHGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (exact strings from prompt)
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain completely filled rows, with each row having a consistent color, and the color varies across rows."
        ]

        # 2) Transformation reasoning chain (exact strings from prompt)
        transformation_reasoning_chain = [
            "Output grids are of size {vars['rows']}x{vars['rows']}.",
            "Each row from the input is transformed into a single-colored cell in the output, forming a diagonal line from top-left to bottom-right.",
            "The order of colors is preserved from the input to the output."
        ]

        # 3) Call superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, 
                     taskvars: Dict[str, Any], 
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Creates an input grid of size rows x cols, where each row is filled with a distinct color.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']

        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Randomly assign a color to each row.
        # We ensure at least some color variation by not repeating the exact color from the previous row.
        prev_color = 0
        for r in range(rows):
            while True:
                color = random.randint(1, 9)  # pick any color 1..9
                if color != prev_color:
                    break
            grid[r, :] = color
            prev_color = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transforms the input grid into a rows x rows output grid,
        copying each row's color into a diagonal position.
        """
        rows = taskvars['rows']
        # The output grid is rows x rows
        output = np.zeros((rows, rows), dtype=int)
        
        # For each row, pick that row's color (take the first column cell, since all columns are the same)
        # and put it on the diagonal in the output.
        for i in range(rows):
            row_color = grid[i, 0]
            output[i, i] = row_color

        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Creates a set of train grids (3-6 examples) plus 1 test grid with random rows/cols.
        """
        # Randomly choose the size of the grids for this entire Task
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        taskvars = {
            'rows': rows,
            'cols': cols
        }

        # Decide how many train examples to produce
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1  # typically 1 test example

        # Use ARCTaskGenerator's helper to produce train/test pairs
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data


