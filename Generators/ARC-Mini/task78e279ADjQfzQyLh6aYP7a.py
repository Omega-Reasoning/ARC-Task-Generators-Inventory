from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optionally, you can use functions from the input_library for the create_input() method
# but not in transform_input(). (They are not used here, but you can import them if needed.)
# from Framework.input_library import create_object, retry, random_cell_coloring, ...

# Optionally, you can use the transformation_library for both create_input() and transform_input() methods
# from Framework.transformation_library import find_connected_objects, GridObject, GridObjects, ...

class Task78e279ADjQfzQyLh6aYP7aGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (list of strings copied from the input)
        observation_chain = [
            "Input grids can have different sizes.",
            "The input grids contain two {color('cell_color1')} cells, separated by empty (0) cells in the top row and two {color('cell_color2')} cells, separated by empty (0) cells in the bottom row.",
            "The remaining cells are empty (0)."
        ]
        
        # 2) Transformation reasoning chain (list of strings copied from the input)
        reasoning_chain = [
            "The output grid is constructed by copying the input grid, and filling in the empty (0) cells between the two colored cells in both the top and bottom rows based on the following condition.",
            "If the two colored cells are separated by one empty (0) cell, fill the empty (0) cell with the same color.",
            "If the two colored cells are separated by two or more empty (0) cells, leave it unchanged."
        ]
        
        # 3) Initialise the parent class with these chains
        super().__init__(observation_chain, reasoning_chain)
    
    def create_grids(self):
        """
        Create 3 train examples and 1 test example. 
        We pick random sizes, random positions for the colored cells (but separated),
        and random distinct colors for cell_color1 and cell_color2 from 1..9.
        Return:
            (task_variables, TrainTestData)
        """
        # 1) Choose random colors
        cell_color1 = random.randint(1, 9)
        cell_color2 = random.randint(1, 9)
        while cell_color2 == cell_color1:
            cell_color2 = random.randint(1, 9)

        # Store them in taskvars
        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2
        }
        
        # We want to create 3 training grids and 1 test grid with different separations.
        # According to constraints:
        #  - One train example has separation=1 for both top and bottom.
        #  - One train example has separation=3 for both top and bottom.
        #  - The third train example can have another separation (e.g., 2).
        #  - The test example should differ in positions from the training examples (and can
        #    have any separation that does not conflict with the variety constraints).

        # Let’s define the separations we want for top & bottom for each training example:
        train_separations = [1, 3, 2]  # the required ones
        test_separation = 2  # or 4, or something else

        # 2) Create the training pairs
        train_pairs = []
        used_positions = []  # Keep track of (top_left, top_right, bottom_left, bottom_right) for variety
        
        for sep in train_separations:
            input_grid = self.create_input(
                taskvars=taskvars,
                gridvars={"separation": sep, "used_positions": used_positions}
            )
            output_grid = self.transform_input(input_grid.copy(), taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # 3) Create the test pair
        #    We want to ensure we choose different positions than the training examples.
        test_input = self.create_input(
            taskvars=taskvars,
            gridvars={"separation": test_separation, "used_positions": used_positions, "is_test": True}
        )
        test_output = self.transform_input(test_input.copy(), taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        # 4) Prepare train/test data
        train_test_data = TrainTestData(train=train_pairs, test=test_pairs)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid according to the input reasoning chain.
        We:
          - Pick a random size [10..20] for rows and columns
          - Place 2 cell_color1 cells on top row separated by 'separation' empty cells
          - Place 2 cell_color2 cells on bottom row separated by 'separation' empty cells
          - Return the resulting grid
        """
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]

        # Constraints / instructions from gridvars
        separation = gridvars.get("separation", 2)
        used_positions = gridvars.get("used_positions", [])
        is_test = gridvars.get("is_test", False)

        # 1) Choose random grid size in [10..20]
        rows = random.randint(10,30)
        cols = random.randint(10, 30)

        # 2) Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # 3) We must place two cell_color1 in the top row, two cell_color2 in the bottom row
        #    ensuring they are separated by 'separation' empty cells.
        #    Also ensure we do not reuse the exact same positions that were used before
        #    (this helps guarantee variety).
        
        # We'll repeatedly attempt random placements until we find one that is not yet used.
        # We'll record the positions for top-left, top-right, bottom-left, bottom-right.
        # The top row is row=0, bottom row is row=rows-1.

        max_attempts = 50
        attempt = 0
        chosen_positions = None
        
        while attempt < max_attempts:
            attempt += 1
            # For top row: pick left col < right col, with right col = left col + separation + 1
            # But we allow the user to place these pairs anywhere so that:
            #   0 <= left_col < right_col < cols
            #   right_col - left_col - 1 = separation
            # => right_col = left_col + separation + 1
            left_col_top = random.randint(0, cols - separation - 2)
            right_col_top = left_col_top + separation + 1

            left_col_bot = random.randint(0, cols - separation - 2)
            right_col_bot = left_col_bot + separation + 1
            
            top_left_pos = (0, left_col_top)
            top_right_pos = (0, right_col_top)
            bottom_left_pos = (rows - 1, left_col_bot)
            bottom_right_pos = (rows - 1, right_col_bot)

            # Check if these positions are used before:
            #   We'll store as a tuple for easy checking. 
            #   We do not require that test is drastically different from *every* training example 
            #   but the instructions do say "position of these cells ... should be different
            #    in test example from what it was in train examples." 
            #   So let's do a straightforward approach: we won't let them be exactly the same
            #   4 positions as any previously used set.
            pos_tuple = (top_left_pos, top_right_pos, bottom_left_pos, bottom_right_pos)
            if pos_tuple not in used_positions:
                # This arrangement is new
                chosen_positions = pos_tuple
                break

        if not chosen_positions:
            # Fallback: just pick the first we found
            chosen_positions = (0, 0), (0, separation+1), (rows-1, 0), (rows-1, separation+1)

        # Mark these positions used so we don't use them again
        used_positions.append(chosen_positions)

        # Place the colored cells
        (top_left_pos, top_right_pos, bottom_left_pos, bottom_right_pos) = chosen_positions
        grid[top_left_pos] = cell_color1
        grid[top_right_pos] = cell_color1
        grid[bottom_left_pos] = cell_color2
        grid[bottom_right_pos] = cell_color2

        return grid

    def transform_input(self, grid, taskvars):
        """
        According to the transformation reasoning chain:
          1. Copy input to output (grid is already a copy in practice).
          2. In top row, if the two cells have separation==1 empty cell between them, fill that cell with color.
             Otherwise leave as is.
          3. Same for bottom row with cell_color2.
        """
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]
        
        # grid is already a copy of the input (in typical usage).
        # We'll scan row=0 for the two cell_color1, row=rows-1 for the two cell_color2.

        rows, cols = grid.shape
        
        # --- top row
        # find columns where cell_color1 is placed
        top_cells = [c for c in range(cols) if grid[0, c] == cell_color1]
        if len(top_cells) == 2:
            left_c, right_c = min(top_cells), max(top_cells)
            gap_size = right_c - left_c - 1
            if gap_size == 1:
                # fill the single gap with color1
                fill_col = left_c + 1
                grid[0, fill_col] = cell_color1
        
        # --- bottom row
        # find columns where cell_color2 is placed
        bottom_cells = [c for c in range(cols) if grid[rows-1, c] == cell_color2]
        if len(bottom_cells) == 2:
            left_c, right_c = min(bottom_cells), max(bottom_cells)
            gap_size = right_c - left_c - 1
            if gap_size == 1:
                fill_col = left_c + 1
                grid[rows-1, fill_col] = cell_color2
        
        return grid


