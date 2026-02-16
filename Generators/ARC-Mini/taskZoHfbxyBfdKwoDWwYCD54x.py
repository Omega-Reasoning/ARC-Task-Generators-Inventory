from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
import numpy as np
import random
import math
from typing import Dict, Any, Tuple

class TasktZoHfbxyBfdKwoDWwYCD54xGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain from the specification
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['columns']}.",
            "Each input grid contains several {color('cell_color')} cells with the remaining cells being empty (0)."
        ]
        # 2) Transformation reasoning chain from the specification
        transformation_reasoning_chain = [
            "Output grids are of size {vars['rows2']}x{vars['columns2']}.",
            "To construct the output grid,transform each cell in the input grid into a 2x2 block in the output grid."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        """
        We pick random rows, columns, and cell_color according to the constraints:
            - rows, columns between 4 and 9
            - rows2 = 2 * rows, columns2 = 2 * columns
            - cell_color from 1..9
        Then we create 4 training pairs and 1 test pair, ensuring each input grid
        meets the constraints (no 4-way connectivity among colored cells, correct fraction).
        """
        taskvars = {}

        # Randomly pick rows, columns in [4..9]
        rows = random.randint(4, 30)
        columns = random.randint(4,30)

        # The color used for the puzzle
        cell_color = random.randint(1, 9)  # from 1..9

        # Derive rows2 and columns2
        rows2 = 2 * rows
        columns2 = 2 * columns

        # Store them in taskvars
        taskvars["rows"] = rows
        taskvars["columns"] = columns
        taskvars["cell_color"] = cell_color
        taskvars["rows2"] = rows2
        taskvars["columns2"] = columns2
        
        # We generate 4 training grids and 1 test grid.
        train_data = []
        # To help ensure distinct grids, keep a set of "signatures" (positions of colored cells)
        seen_signatures = set()

        # Create 4 training examples
        for _ in range(4):
            input_grid = self.create_input(taskvars, {}, seen_signatures)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))
        
        # Create 1 test example
        test_data = []
        input_grid_test = self.create_input(taskvars, {}, seen_signatures)
        output_grid_test = self.transform_input(input_grid_test, taskvars)
        test_data.append(GridPair(input=input_grid_test, output=output_grid_test))

        return taskvars, TrainTestData(train=train_data, test=test_data)

    def create_input(self, taskvars, gridvars, seen_signatures=None):
        """
        Creates an input grid with size rows x columns, containing colored cells
        (with color cell_color) that satisfy:
            * Non-4-way-connected (i.e., no two colored cells share an edge)
            * The fraction of colored cells is between m/4 and m/3, where m = rows*columns
            * (Optionally) Different from previously generated grids (checked via signature).
        """
        rows = taskvars["rows"]
        columns = taskvars["columns"]
        color_val = taskvars["cell_color"]
        total_cells = rows * columns
        
        min_count = math.ceil(total_cells / 4.0)
        max_count = math.floor(total_cells / 3.0)
        
        # Just in case, clamp min_count and max_count
        if min_count < 1:
            min_count = 1
        if max_count < min_count:
            max_count = min_count
        
        # We'll attempt to place a random count of colored cells in [min_count..max_count].
        # Then ensure no two colored cells are 4-adjacent.
        def generate_one_grid():
            grid = np.zeros((rows, columns), dtype=int)
            
            # Decide how many colored cells
            nr_colored = random.randint(min_count, max_count)

            # Positions tried so far
            used_positions = set()
            
            for _ in range(nr_colored):
                # Attempt to find a spot that doesn't share an edge with any existing colored cell
                attempts = 0
                while True:
                    attempts += 1
                    if attempts > 5000:
                        # If we can't place them, just fail and retry the entire grid
                        return None
                    r = random.randrange(rows)
                    c = random.randrange(columns)
                    if (r, c) in used_positions:
                        continue
                    # Check adjacency with previously placed colored cells
                    neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                    # If any neighbor is in used_positions, skip
                    if any(nb in used_positions for nb in neighbors):
                        continue
                    # Place color
                    grid[r, c] = color_val
                    used_positions.add((r, c))
                    break
            # Return the completed grid
            return grid

        # We'll keep trying to generate a valid grid until we find one not used before
        while True:
            candidate = generate_one_grid()
            if candidate is None:
                # Could not place colored cells without adjacency, try again
                continue
            
            # Make a quick signature: store the list of positions of colored cells sorted
            colored_positions = np.argwhere(candidate == color_val)
            signature = tuple(map(tuple, colored_positions))  # convert to tuple of (r,c) 
            if seen_signatures is not None:
                if signature in seen_signatures:
                    # We want a different example; keep trying
                    continue
                seen_signatures.add(signature)
            
            return candidate

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        The transformation:
            "Transform each cell in the input grid into a 2x2 block in the output grid."
        If input shape is (rows, columns), output shape must be (2*rows, 2*columns).
        For cell (r, c) of color X in the input, the corresponding 2x2 block in the 
        output (covering rows [2r..2r+1], cols [2c..2c+1]) is set to X.
        """
        rows = grid.shape[0]
        columns = grid.shape[1]
        out_rows = 2 * rows
        out_cols = 2 * columns
        
        output_grid = np.zeros((out_rows, out_cols), dtype=int)
        
        for r in range(rows):
            for c in range(columns):
                val = grid[r, c]
                # Map (r, c) -> block in output_grid
                rr = 2 * r
                cc = 2 * c
                output_grid[rr,   cc]   = val
                output_grid[rr,   cc+1] = val
                output_grid[rr+1, cc]   = val
                output_grid[rr+1, cc+1] = val
        
        return output_grid


