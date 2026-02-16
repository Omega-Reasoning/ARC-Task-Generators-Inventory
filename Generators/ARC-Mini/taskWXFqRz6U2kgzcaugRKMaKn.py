from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple


class TaskWXFqRz6U2kgzcaugRKMaKnGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid contains one or two multi-colored (1-9) cells in each row, except in the last column, while all other cells remain empty (0).",
            "If there are two colored cells in a single row, they must have different colors and be separated by empty (0) cells."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and extending each colored cell to the right until it reaches another colored cell or the edge of the grid."
        ]

        # 3) Call the superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, 
                     taskvars: dict, 
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid of size rows x cols with the constraints:
        - Each row has either 1 or 2 colored cells (no colored cell in last column).
        - If 2 cells in a row, they have different colors and are separated by at least one empty cell.
        - Across the entire grid, ensure at least one row has exactly 2 colored cells
          and at least one row has exactly 1 colored cell.
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]

        grid = np.zeros((rows, cols), dtype=int)

        # Decide for each row whether it has 1 or 2 colored cells.
        # We must ensure at least one row of each type (single or double).
        # Start by randomly choosing all row types, then fix if needed.
        row_types = [random.choice(["S", "D"]) for _ in range(rows)]
        
        # Ensure at least one single (S) and one double (D)
        if "S" not in row_types:
            row_types[0] = "S"
        if "D" not in row_types:
            row_types[-1] = "D"

        # Now fill the grid row by row
        for r in range(rows):
            if row_types[r] == "S":
                # Single colored cell
                c = random.randint(0, cols - 2)  # up to cols-2 to avoid last column
                color = random.randint(1, 9)    # random color
                grid[r, c] = color

            else:
                # Double colored cells
                # We need two distinct columns, both < cols-1, with a gap between them
                valid_placements = []
                # Collect all valid pairs (c1, c2) where c1 < c2 < cols-1 and at least 1 gap
                for c1 in range(cols - 2):
                    for c2 in range(c1 + 2, cols - 1):
                        valid_placements.append((c1, c2))
                if not valid_placements:
                    # Fallback: if no valid placements, degrade to single
                    c = random.randint(0, cols - 2)
                    color = random.randint(1, 9)
                    grid[r, c] = color
                else:
                    c1, c2 = random.choice(valid_placements)
                    color1 = random.randint(1, 9)
                    # Ensure second color is different
                    color2 = random.choice([c for c in range(1, 10) if c != color1])
                    grid[r, c1] = color1
                    grid[r, c2] = color2
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Implement the transformation:
        - Copy the input grid.
        - For each colored cell in each row, extend it to the right until
          reaching another colored cell or the edge of the grid.
        """
        rows, cols = grid.shape
        out_grid = grid.copy()

        for r in range(rows):
            # We'll scan from left to right tracking colored cells.
            c = 0
            while c < cols:
                if out_grid[r, c] != 0:
                    # Found a colored cell, say color col_val
                    col_val = out_grid[r, c]
                    # Look ahead for next colored cell in the row or boundary
                    next_color_col = None
                    for c2 in range(c + 1, cols):
                        if out_grid[r, c2] != 0:
                            next_color_col = c2
                            break

                    if next_color_col is not None:
                        # Fill [c, next_color_col) with col_val
                        for fill_col in range(c, next_color_col):
                            out_grid[r, fill_col] = col_val
                        # Move c to next_color_col for next iteration
                        c = next_color_col
                    else:
                        # No next colored cell, fill to the edge
                        for fill_col in range(c, cols):
                            out_grid[r, fill_col] = col_val
                        c = cols  # done for this row
                else:
                    c += 1

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Create 3-4 train examples and 2 test examples, each consistent with the
        input reasoning chain and transformation logic.
        - We randomly choose the grid size once per entire task (vars['rows'], vars['cols']).
        - Then we generate multiple examples with the same size but different random placements.
        """
        # Randomly choose row/col size for the entire puzzle
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        # We'll store these in taskvars so they can be substituted in the chains
        taskvars = {
            "rows": rows,
            "cols": cols
        }

        # Randomly decide how many train grids (3 or 4) and we fix 2 test grids
        nr_train_examples = random.choice([3, 4])
        nr_test_examples = 2

        # We can use the built-in helper to create train and test data, 
        # using the same create_input/transform_input for each example.
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data

