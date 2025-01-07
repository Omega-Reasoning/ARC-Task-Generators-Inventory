# my_arc_generator.py

# 1) Required imports
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
# We may optionally use functions from input_library, but only in create_input():
from input_library import retry  # as an example, not strictly needed here

# We may also optionally use transformation_library in either create_input or transform_input:
from transformation_library import find_connected_objects, GridObject, GridObjects


class TaskULXmq8k24GG7tNvmdgfKFvGenerator(ARCTaskGenerator):
    def __init__(self):
        """
        ARC Task Generator subclass that creates puzzles according to:
          Input Reasoning Chain:
            1) "Input grids can have different sizes."
            2) "Each input grid contains several {vars['number_of_cells']} {color('cell_color')} cells in the first row."
            3) "The remaining cells in the input grid are empty (0)."

          Transformation Reasoning Chain:
            1) "To construct the output grids, copy the input grid and find all connected cells in the first row."
            2) "Fill the entire columns of these connected cells with the same color, while keeping the single cells as it is."

        General Instructions / Invariants:
            * The grid size is between 8 and 20 rows/columns
            * cell_color is an integer from 1 to 9
            * number_of_cells is an integer between 3 and 8
            * Each input grid should have at least one group of adjacent cell_color cells in the first row.
        """
        # 1) Input reasoning chain (list of strings, uses f-string placeholders for task variables)
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains {vars['number_of_cells']} {color('cell_color')} cells in the first row.",
            "The remaining cells in the input grid are empty (0)."
        ]

        # 2) Transformation reasoning chain (list of strings, uses f-string placeholders for task variables)
        transformation_reasoning_chain = [
            "To construct the output grids, copy the input grid and find all connected {color('cell_color')} cells in the first row.",
            "Fill the entire columns of these connected {color('cell_color')} cells with the same color, while keeping the single {color('cell_color')} cells unchanged."
        ]

        # 3) Superclass init call
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
          1. Randomly choose grid dimensions (height, width) between 8..20.
          2. Fill the first row with exactly taskvars['number_of_cells'] cells of color taskvars['cell_color'].
             Among those cells, ensure at least one pair is adjacent.
          3. The rest of the grid is empty (0).

        We must satisfy the constraint that there's at least one group of >=2 contiguous colored cells in the first row.
        """
        cell_color = taskvars['cell_color']
        number_of_cells = taskvars['number_of_cells']
        
        # Randomly pick grid dimensions
        height = random.randint(8, 20)
        width = random.randint(8, 20)
        
        # Create the grid filled with zeros
        grid = np.zeros((height, width), dtype=int)
        
        # We must place 'number_of_cells' cells of color 'cell_color' in the first row,
        # ensuring at least 2 contiguous cells. We'll keep retrying until successful.
        def place_cells():
            # Start fresh in the first row
            row = np.zeros(width, dtype=int)
            # Choose positions for 'number_of_cells' colored cells out of 'width'
            positions = random.sample(range(width), number_of_cells)
            for pos in positions:
                row[pos] = cell_color
            
            # Check if there's at least one adjacency
            # i.e. row[i] == row[i+1] == cell_color for some i
            adjacency_found = any(row[i] == cell_color and row[i+1] == cell_color
                                  for i in range(width - 1))
            return row, adjacency_found
        
        # We'll try up to 50 times just in case
        for _ in range(50):
            row, adjacency_found = place_cells()
            if adjacency_found:
                grid[0] = row
                break
        else:
            # If the loop completes, raise an error (very unlikely with random sampling).
            raise ValueError("Could not place at least one contiguous pair in the first row.")
        
        return grid

    def fill_columns_if_needed(self, out_grid, start, end, color):
        length = end - start
        if length > 1:
            for col_index in range(start, end):
                out_grid[:, col_index] = color

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        out_grid = grid.copy()
        first_row = out_grid[0]
        start_idx = None
        current_color = 0
        
        for i in range(len(first_row)):
            if first_row[i] != 0:
                if first_row[i] == current_color:
                    continue
                else:
                    if current_color != 0 and start_idx is not None:
                        self.fill_columns_if_needed(out_grid, start_idx, i, current_color)
                    start_idx = i
                    current_color = first_row[i]
            else:
                if current_color != 0 and start_idx is not None:
                    self.fill_columns_if_needed(out_grid, start_idx, i, current_color)
                start_idx = None
                current_color = 0
        
        if current_color != 0 and start_idx is not None:
            self.fill_columns_if_needed(out_grid, start_idx, len(first_row), current_color)
        
        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        1. Randomly set the task variables: 'cell_color' (1..9) and 'number_of_cells' (3..8).
        2. Generate 3-6 training pairs and 1 test pair using the create_input() and transform_input() methods.
        3. Return the task variables plus all train/test data.
        """
        taskvars = {}
        taskvars['cell_color'] = random.randint(1, 9)
        taskvars['number_of_cells'] = random.randint(3, 8)

        # We'll choose how many train examples randomly between 3..6
        nr_train = random.randint(3, 6)
        nr_test = 1

        # We can use create_grids_default(...) which auto-generates these pairs
        train_test_data = self.create_grids_default(
            nr_train_examples=nr_train,
            nr_test_examples=nr_test,
            taskvars=taskvars
        )
        
        return taskvars, train_test_data


