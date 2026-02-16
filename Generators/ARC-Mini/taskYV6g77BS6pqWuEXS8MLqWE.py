from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optionally, we can import from our libraries if desired:
# from Framework.transformation_library import find_connected_objects, ...
# from Framework.input_library import ...

class TaskYV6g77BS6pqWuEXS8MLqWEGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain (exactly copied from the prompt):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a completely filled {color('object_color')} middle row, along with several multi-colored (1-9) cells in the top and bottom halves of the grid.",
            "The multi-colored cells are positioned so that those in the top half are mirrored in the bottom half, using the middle row as the line of reflection, but with different colors."
        ]
        
        # 2) The transformation reasoning chain (exactly copied from the prompt):
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and changing the multi-colored cells to {color('cell_color1')} if they are in the top half, and to {color('cell_color2')} if they are in the bottom half."
        ]
        
        # 3) Call the parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates the task variables and the training/testing grids.
        Returns:
            - taskvars: Dictionary with keys = variables for the template
            - train_test_data: Dictionary with 'train' and 'test' lists of GridPair
        """

        # 1. Randomly pick rows (odd, between 3 and 30) and cols (between 7 and 30)
        #    Also pick distinct colors for object_color, cell_color1, cell_color2.
        #    Each must be in [1..9], all different.
        possible_rows = [r for r in range(3, 31) if r % 2 == 1]  # odd numbers from 3 to 29 inclusive
        rows = random.choice(possible_rows)
        cols = random.randint(7, 30)

        # Distinct colors from 1..9
        color_choices = random.sample(range(1, 10), 3)
        object_color = color_choices[0]
        cell_color1 = color_choices[1]
        cell_color2 = color_choices[2]

        taskvars = {
            "rows": rows,
            "cols": cols,
            "object_color": object_color,
            "cell_color1": cell_color1,
            "cell_color2": cell_color2
        }

        # 2. Randomly decide how many training examples (3 or 4), always 1 test example
        nr_train_examples = random.choice([3, 4])
        nr_test_examples = 1

        # 3. Use the built-in helper to create training and test pairs
        #    (no special cross-example constraints needed for this puzzle).
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid according to the described input reasoning chain:
        1. Grid of size rows x cols, all zeros initially.
        2. Fill the middle row completely with object_color.
        3. Place several multi-colored cells in the top half, reflect them in the bottom half 
           with different colors (none of them can be object_color).
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        object_color = taskvars["object_color"]

        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Middle row index
        mid_row = rows // 2

        # 1) Fill the middle row with object_color
        grid[mid_row, :] = object_color

        # 2) Place random multi-colored cells in the top half. We reflect them in the bottom half
        #    with different colors. 
        #    Constraint: top cell color != object_color; bottom cell color != object_color and != top color.
        top_half = range(0, mid_row)  # row indices for the top half
        bottom_half = range(mid_row + 1, rows)  # row indices for the bottom half

        # Decide how many random cells to place in the top half (at least 1)
        # Try for up to roughly 20% of the top half area, but minimum 1:
        top_half_area = mid_row * cols
        num_cells_top = random.randint(3, max(3, top_half_area // 5))

        # Pick random distinct positions in the top half
        chosen_positions = set()
        attempts = 0
        while len(chosen_positions) < num_cells_top and attempts < 5 * num_cells_top:
            r = random.choice(list(top_half))
            c = random.randint(0, cols - 1)
            chosen_positions.add((r, c))
            attempts += 1

        # Place and reflect
        for (r, c) in chosen_positions:
            # top color must not be object_color
            top_color_candidates = [clr for clr in range(1, 10) if clr != object_color]
            top_color = random.choice(top_color_candidates)

            # reflected cell row in the bottom half
            reflected_r = rows - 1 - r
            # bottom color must differ from object_color AND from top_color
            bottom_color_candidates = [clr for clr in range(1, 10)
                                       if clr != object_color and clr != top_color]
            # if we want strictly different from top_color, pick from the above
            if not bottom_color_candidates:
                # fallback if there's no candidate (shouldn't happen with 1..9 range)
                bottom_color_candidates = [clr for clr in range(1, 10) if clr != object_color]
            bottom_color = random.choice(bottom_color_candidates)

            # Place them in the grid
            grid[r, c] = top_color
            grid[reflected_r, c] = bottom_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid to the output grid according to:
        "The output grid is created by copying the input grid and changing the multi-colored cells to
         cell_color1 if they are in the top half, and to cell_color2 if they are in the bottom half."
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        object_color = taskvars["object_color"]
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]

        out_grid = grid.copy()
        mid_row = rows // 2

        # For the top half (0..mid_row-1), any cell not 0 or object_color becomes cell_color1
        for r in range(mid_row):
            for c in range(cols):
                if out_grid[r, c] != 0 and out_grid[r, c] != object_color:
                    out_grid[r, c] = cell_color1

        # For the bottom half (mid_row+1..end), any cell not 0 or object_color becomes cell_color2
        for r in range(mid_row + 1, rows):
            for c in range(cols):
                if out_grid[r, c] != 0 and out_grid[r, c] != object_color:
                    out_grid[r, c] = cell_color2

        return out_grid


