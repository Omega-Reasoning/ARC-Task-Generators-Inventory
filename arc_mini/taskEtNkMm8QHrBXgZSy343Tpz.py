# generator_separated_fill.py
#
# Example ARC Task Generator that places exactly three colored cells separated by empty cells
# in the input grid, then transforms the input by flood-filling empty 8-way connected regions 
# of each special-colored cell. 

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskEtNkMm8QHrBXgZSy343TpzGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialising the input reasoning chain
        self.input_reasoning_chain = [
            "All input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid contains exactly three {color('cell_color')} cells, positioned in such a way that they are completely separated from one another by empty (0) cells."
        ]
        # 2) Initialising the transformation reasoning chain
        self.transformation_reasoning_chain = [
            "The output grid is created by copying the input grid, and for each {color('cell_color')} cell, all adjacent 8-way connected empty (0) cells are filled with {color('fill_color')} color, forming a {color('fill_color')} frame around each {color('cell_color')} cell.",
            "The frames can overlap each other in cases where two or more {color('cell_color')} cells are close to each other."
        ]
        # 3) Call the parent constructor
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Sets up task variables and creates the train/test grids.
        We use 3-5 training examples and 1 test example.
        """
        # Randomly choose grid size
        rows = random.randint(6, 15)
        cols = random.randint(6, 15)

        # Randomly choose colors (different from each other)
        cell_color = random.randint(1, 9)
        fill_color = random.randint(1, 9)
        while fill_color == cell_color:
            fill_color = random.randint(1, 9)

        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color': cell_color,
            'fill_color': fill_color
        }

        # Randomly choose how many train examples (3-5) and 1 test example
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1

        # We can use the helper from the parent if the logic for input is the same for all examples
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid of size rows x cols with exactly three cells of 'cell_color',
        ensuring:
         - No 'cell_color' cell is on the edge.
         - The three 'cell_color' cells are not adjacent (not even diagonally).
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        cell_color = taskvars['cell_color']

        grid = np.zeros((rows, cols), dtype=int)

        # We'll place exactly 3 cell_color cells inside the grid, not touching edges,
        # and not diagonally/cardinally adjacent.
        positions = []
        attempts = 0
        max_attempts = 1000
        
        while len(positions) < 3 and attempts < max_attempts:
            attempts += 1
            # pick a random position away from edges
            r = random.randint(1, rows - 2)
            c = random.randint(1, cols - 2)
            
            # check adjacency to previously placed cell_color cells
            # we want to ensure no two cell_color cells are adjacent or diagonally touching
            is_valid = True
            for (pr, pc) in positions:
                if abs(pr - r) <= 1 and abs(pc - c) <= 1:
                    is_valid = False
                    break
            
            if is_valid:
                positions.append((r, c))

        if len(positions) < 3:
            # if we failed to place them, fallback to a simpler pattern in the center
            # (this is unlikely, but a safeguard)
            positions = [
                (1, 1),
                (rows // 2, cols // 2),
                (rows - 2, cols - 2)
            ]

        # Place the cells
        for (r, c) in positions:
            grid[r, c] = cell_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Copy the input grid and for each cell_color cell, only fill the **immediately**
        connected 8-way empty (0) cells with fill_color.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        cell_color = taskvars['cell_color']
        fill_color = taskvars['fill_color']

        out_grid = grid.copy()

        # Identify all positions that have 'cell_color'
        cell_positions = [(r, c) for r in range(rows) for c in range(cols) if out_grid[r, c] == cell_color]

        # Define 8-way directions
        directions = [(-1,-1), (-1,0), (-1,1),
                    (0,-1),          (0,1),
                    (1,-1), (1,0),  (1,1)]

        # For each cell_color cell, fill **only** its adjacent empty cells (0)
        for (sr, sc) in cell_positions:
            for dr, dc in directions:
                nr, nc = sr + dr, sc + dc
                # Ensure within bounds and check if it's an empty (0) cell
                if 0 <= nr < rows and 0 <= nc < cols and out_grid[nr, nc] == 0:
                    out_grid[nr, nc] = fill_color

        return out_grid



