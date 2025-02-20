#!/usr/bin/env python3
"""
Example ARC-AGI Task Generator

This generator creates grids according to the specified input reasoning chain:
1. Input grids are square of size {vars['grid_size']}x{vars['grid_size']}.
2. The last row is completely filled with {color('cell_color1')}.
3. The second last row contains at least two {color('cell_color1')} cells, each separated by 1 or 2 empty cells.
4. For each {color('cell_color1')} cell in the second last row, the cell directly above it in the third last row is {color('cell_color2')}.
5. All other cells remain empty (0).

The transformation reasoning chain is:
* Copy the input grid, remove all {color('cell_color2')} cells from the third last row, and shift them down to the last row in the same column, overwriting the cell_color1 there.

We ensure:
- grid_size is between 5 and 30.
- cell_color1 != cell_color2.
- We produce 3-4 training examples and 1 test example, each with a varied arrangement
  of cell_color1 cells in the second last row, so that a test taker can deduce the pattern.
"""

import numpy as np
import random

# Optional: we could use some of these utilities, but here we do it directly
# from input_library import create_object, retry, Contiguity
# from transformation_library import find_connected_objects, GridObject, GridObjects

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

class Task3618c87eGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a completely filled last row with {color('cell_color1')} cells, several {color('cell_color1')} cells in the second last row, and several {color('cell_color2')} cells in the third last row, placed directly above the {color('cell_color1')} cells in the second last row.",
            "Each {color('cell_color1')} cell in the second last row must be separated by at least one or two empty (0) cells.",
            "All other cells remain empty (0)."
        ]
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid, removing all {color('cell_color2')} cells from the third last row, and shifting them to the last row in the same column."
        ]
        # 3) Call superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Creates an input grid of size grid_size x grid_size where:
        - The last row is completely filled with cell_color1.
        - The second last row has at least two cell_color1 cells, each separated by 1 or 2 zeros.
        - For each cell_color1 cell in the second last row, there is a cell_color2 cell
          directly above it (i.e. in the third last row).
        - All other cells remain zero (empty).
        
        Returns
        -------
        A numpy.ndarray representing the generated input grid.
        """
        grid_size = taskvars['grid_size']
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # 1) Completely fill the last row with color1
        grid[grid_size - 1, :] = color1
        
        # 2) Generate random arrangement of color1 in the second last row,
        #    ensuring at least 2 cells, separated by 1 or 2 empty cells.
        arrangement = self._generate_arrangement(grid_size)
        
        # Place these color1 cells in the second last row
        for col in arrangement:
            grid[grid_size - 2, col] = color1
        
        # 3) Place color2 cells directly above each color1 cell in third last row
        #    i.e. row index = grid_size - 3, same column.
        third_last_row_idx = grid_size - 3
        for col in arrangement:
            # Make sure the grid has at least 3 rows, which it does by our constraints (5-30)
            if third_last_row_idx >= 0:
                grid[third_last_row_idx, col] = color2
        
        # Return the grid and also store arrangement in gridvars if needed
        # (Some usage patterns might want the arrangement for checking variety.)
        gridvars['arrangement'] = arrangement
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transforms the input grid to the output grid by:
        - Copying the input grid.
        - Removing all cell_color2 cells from the third last row.
        - Shifting them down to the last row (same column), overwriting whatever was there.
        """
        new_grid = np.copy(grid)
        grid_size = taskvars['grid_size']
        color2 = taskvars['cell_color2']
        
        third_last_row = grid_size - 3  # 0-based index
        if third_last_row < 0:
            # Should not happen due to grid_size >= 5, but just in case.
            return new_grid
        
        # Move color2 cells from third last row down to last row in the same column
        for col in range(grid_size):
            if new_grid[third_last_row, col] == color2:
                # Remove from the third last row
                new_grid[third_last_row, col] = 0
                # Place in the last row (overwrite)
                new_grid[grid_size - 1, col] = color2
        
        return new_grid

    def create_grids(self):
        """
        Creates 3-4 train examples and 1 test example, ensuring variety in the arrangement
        of cell_color1 in the second last row. Returns (taskvars, TrainTestData).
        """
        # Randomly choose grid size
        grid_size = random.randint(5, 30)
        
        # Random distinct colors
        cell_color1 = random.randint(1, 9)
        cell_color2 = random.randint(1, 9)
        while cell_color2 == cell_color1:
            cell_color2 = random.randint(1, 9)
        
        # Store in taskvars
        taskvars = {
            'grid_size': grid_size,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2
        }
        
        # Number of training examples: randomly choose 3 or 4
        nr_train = random.randint(3, 4)
        
        # We'll collect unique arrangements (to ensure variety).
        arrangements_set = set()
        train_pairs = []
        
        # Generate train examples
        while len(train_pairs) < nr_train:
            gridvars = {}
            inp = self.create_input(taskvars, gridvars)
            arrangement = tuple(gridvars['arrangement'])
            if arrangement not in arrangements_set:
                arrangements_set.add(arrangement)
                outp = self.transform_input(inp, taskvars)
                train_pairs.append({
                    'input': inp,
                    'output': outp
                })
        
        # Generate test example with a different arrangement
        test_pair = None
        while test_pair is None:
            gridvars = {}
            inp = self.create_input(taskvars, gridvars)
            arrangement = tuple(gridvars['arrangement'])
            if arrangement not in arrangements_set:
                # found a new arrangement
                outp = self.transform_input(inp, taskvars)
                test_pair = {
                    'input': inp,
                    'output': outp
                }
        
        train_test_data = {
            'train': train_pairs,
            'test': [test_pair]
        }
        
        return taskvars, train_test_data

    def _generate_arrangement(self, grid_size: int):
        """
        Generates a random arrangement of columns to place cell_color1
        in the second last row such that:
        - At least 2 positions
        - Each pair of consecutive positions differs by 2 or 3
          (i.e. there's 1 or 2 empty cells between them).
        """
        # Keep trying until we get at least two positions
        for _ in range(100):  # avoid infinite loop
            cols = []
            col = random.randint(0, 1)  # Start randomly in the first 2 columns
            while True:
                cols.append(col)
                # Step by either 2 or 3
                step = random.choice([2, 3])
                col = col + step
                if col >= grid_size:
                    break
            if len(cols) >= 2:
                return cols
        
        # Fallback: if for some reason we can't get 2 (very unlikely), pick two columns by force
        return [0, 2]

# --------------- TEST CODE ---------------
if __name__ == "__main__":
    # Instantiate the generator
    generator = CellColorShiftTask()
    # Create one set of train/test grids
    taskvars, train_test_data = generator.create_grids()
    
    # Visualize them
    print("Task Variables:", taskvars)
    generator.visualize_train_test_data(train_test_data)
