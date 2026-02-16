# my_arc_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskcwsbonBBqy3ZMUioENEbxgGenerator(ARCTaskGenerator):
    def __init__(self):
        """
        1) Initialize the input reasoning chain (list of strings).
        2) Initialize the transformation reasoning chain (list of strings).
        3) Call super().__init__(...).
        """
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains one {color('cell_color')} cell, positioned differently on the border of the grid.",
            "The remaining cells are empty (0)."
        ]
        
        reasoning_chain = [
            "To construct the output grid, copy the input grid.",
            "Fill in the entire row or column containing that colored cell according to the following rules.",
            "If the {color('cell_color')} cell is in the first column, last column, or one of the four corners of the grid, fill the entire row containing that cell with the same color.",
            "If the {color('cell_color')} cell is in the top or bottom row (but not a corner), fill the entire column containing that cell with the same color."
        ]
        
        super().__init__(observation_chain, reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        We create 6 train examples and 1 test example.
        Each input grid must have exactly one colored cell on the border.
        We ensure the following coverage:
          - One training example has the cell on the last row (except corners).
          - One training example has the cell on the last column (except corners).
          - One training example has the cell on the first column (except corners).
          - One training example has the cell on the first row (except corners).
          - Two training examples have the cell in a corner (two different corners).
          - The test example has the cell on a border in a position different from all training ones.
        We also randomize grid sizes between 7x7 and 12x12.
        """

        # Step 1: Pick a color for cell_color (1..9)
        cell_color = random.randint(1, 9)

        # We define 6 specific border placements for training, plus 1 for test.
        # This approach ensures we meet the constraints about coverage.
        
        # 1) last row (except corners)
        # 2) last column (except corners)
        # 3) first column (except corners)
        # 4) first row (except corners)
        # 5) corner #1
        # 6) corner #2
        # Then test: a different border position from the above.

        # We'll define these placements as lambdas returning (row, col) 
        # relative to chosen grid size.

        # The corner definitions can be top-left, top-right, bottom-left, bottom-right
        # We'll pick two distinct corners for training.

        def pos_last_row(h, w):
            # random column not corner
            c = random.randint(1, w-2)
            return (h-1, c)
        
        def pos_last_col(h, w):
            # random row not corner
            r = random.randint(1, h-2)
            return (r, w-1)
        
        def pos_first_col(h, w):
            # random row not corner
            r = random.randint(1, h-2)
            return (r, 0)
        
        def pos_first_row(h, w):
            # random col not corner
            c = random.randint(1, w-2)
            return (0, c)
        
        # For corners, pick which corners to use:
        # let's do top-left and bottom-right for variety in the training set.
        def pos_corner_topleft(h, w):
            return (0, 0)

        def pos_corner_bottomright(h, w):
            return (h-1, w-1)

        # We'll store these in a list for the 6 training examples.
        training_position_functions = [
            pos_last_row,
            pos_last_col,
            pos_first_col,
            pos_first_row,
            pos_corner_topleft,
            pos_corner_bottomright
        ]
        
        # For the test example, we want a border position different from all the above.
        # One easy way: pick a random corner or border cell not used above.
        # We'll do a simple approach: pick among the corners or edges that we haven't used 
        #   OR a different corner we haven't used, or a side we haven't used exactly in that position.
        # For demonstration, let's just pick "top-right corner" (0, w-1) 
        # provided none of the training positions used top-right corner. 
        # If top-right corner was used, we'll pick "bottom-left corner" (h-1, 0), etc.
        # If both corners are used, we'll pick a random edge location not used. 
        # For simplicity, let's define a small helper.

        possible_test_positions = [
            lambda h, w: (0, w-1),      # top-right corner
            lambda h, w: (h-1, 0)       # bottom-left corner
        ]
        
        # We'll finalize the test position after we've generated the train positions, 
        # ensuring no duplication.

        # We'll generate train data
        train = []
        used_positions = set()  # set of (row, col, h, w) for uniqueness
        for func in training_position_functions:
            # Randomly pick a grid size
            h = random.randint(7, 12)
            w = random.randint(7, 12)
            row, col = func(h, w)
            used_positions.add((row, col, h, w))
            
            input_grid = self.create_input(
                taskvars={"cell_color": cell_color},
                gridvars={
                    "height": h,
                    "width": w,
                    "cell_row": row,
                    "cell_col": col
                }
            )
            output_grid = self.transform_input(input_grid, {"cell_color": cell_color})
            train.append({
                "input": input_grid,
                "output": output_grid
            })
        
        # Now pick a test example. We try the possible_test_positions in order, 
        #   or revert to a random border position not used in training.
        test_example = None
        for candidate in possible_test_positions:
            # Try a few random attempts at size to see if we can get a unique position:
            for _ in range(10):
                h = random.randint(7, 30)
                w = random.randint(7, 30)
                row, col = candidate(h, w)
                if (row, col, h, w) not in used_positions:
                    # we found a distinct combination
                    test_input_grid = self.create_input(
                        taskvars={"cell_color": cell_color},
                        gridvars={
                            "height": h,
                            "width": w,
                            "cell_row": row,
                            "cell_col": col
                        }
                    )
                    test_output_grid = self.transform_input(
                        test_input_grid, 
                        {"cell_color": cell_color}
                    )
                    test_example = {
                        "input": test_input_grid,
                        "output": test_output_grid
                    }
                    break
            if test_example is not None:
                break
        
        # If we still have no test_example, fallback to a random border position 
        # that wasn't used in training. This is a safety net if the corners were all used.
        if test_example is None:
            for _ in range(100):
                h = random.randint(7, 12)
                w = random.randint(7, 12)
                # place cell in a random border position
                # top row or bottom row or left col or right col
                if random.random() < 0.5:
                    # choose top or bottom row
                    row = 0 if random.random() < 0.5 else (h - 1)
                    col = random.randint(0, w-1)
                else:
                    # choose left or right column
                    col = 0 if random.random() < 0.5 else (w - 1)
                    row = random.randint(0, h-1)
                if (row, col, h, w) not in used_positions:
                    test_input_grid = self.create_input(
                        taskvars={"cell_color": cell_color},
                        gridvars={
                            "height": h,
                            "width": w,
                            "cell_row": row,
                            "cell_col": col
                        }
                    )
                    test_output_grid = self.transform_input(
                        test_input_grid, 
                        {"cell_color": cell_color}
                    )
                    test_example = {
                        "input": test_input_grid,
                        "output": test_output_grid
                    }
                    break
        
        # Construct final TrainTestData
        train_test_data = {
            "train": train,
            "test": [test_example] if test_example else []
        }

        # The dictionary of variables used in the input/transformation reasoning chain
        taskvars = {
            "cell_color": cell_color
        }
        
        return taskvars, train_test_data

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
          1) The grid size is between 7 and 12 in both dimensions.
          2) Exactly one cell of color cell_color is placed on the border.
          3) All other cells are 0.
        We assume gridvars has keys 'height', 'width', 'cell_row', 'cell_col'.
        """
        cell_color = taskvars["cell_color"]
        h = gridvars["height"]
        w = gridvars["width"]
        row = gridvars["cell_row"]
        col = gridvars["cell_col"]
        
        # Create empty grid
        grid = np.zeros((h, w), dtype=int)
        
        # Place the single colored cell
        grid[row, col] = cell_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:

        """
        Implement the transformation reasoning chain:
          1) Copy input grid
          2) If the single {color('cell_color')} cell is in first/last column or corner -> fill entire row.
             If it is in top/bottom row (but not corner) -> fill entire column.
        """
        cell_color = taskvars["cell_color"]
        
        # Copy grid
        out_grid = grid.copy()

        # Locate the single non-zero cell (the puzzle states exactly one)
        positions = np.argwhere(out_grid == cell_color)
        if len(positions) == 0:
            # no colored cell found (shouldn't happen if constraints are followed)
            return out_grid
        
        r, c = positions[0]
        h, w = out_grid.shape
        
        # Check corner condition or first/last column
        # corners: (0,0), (0, w-1), (h-1, 0), (h-1, w-1)
        is_corner = ((r == 0 and c == 0) or 
                     (r == 0 and c == w-1) or
                     (r == h-1 and c == 0) or
                     (r == h-1 and c == w-1))
        
        if is_corner or (c == 0) or (c == w - 1):
            # fill the entire row
            out_grid[r, :] = cell_color
        else:
            # otherwise, it must be on the top or bottom row (but not corner) => fill the column
            out_grid[:, c] = cell_color
        
        return out_grid


