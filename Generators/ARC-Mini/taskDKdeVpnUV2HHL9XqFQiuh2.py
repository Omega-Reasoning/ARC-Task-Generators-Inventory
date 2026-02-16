from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# Optional but encouraged:
# from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
# from Framework.input_library import retry, create_object, Contiguity

class TaskDKdeVpnUV2HHL9XqFQiuh2Generator(ARCTaskGenerator):

    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "All input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a diagonal line made of same-colored (1-9) cells running either from top-left to bottom-right or from top-right to bottom-left, with the remaining cells being empty (0).",
            "The diagonal line is not necessarily the main diagonal; it can be any diagonal starting from the first row and extending to the last row."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and completely filling each column that contains a colored cell with the same color."
        ]

        # 3) Super call
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create a single input grid according to:
          1) The grid size rows x cols
          2) A single color for the diagonal
          3) A diagonal direction (left->right or right->left)
          4) A valid start column ensuring the diagonal extends from first to last row
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        color = gridvars['color']
        direction = gridvars['direction']  # 'lr' or 'rl'
        start_col = gridvars['start_col']  # chosen so the diagonal fits

        grid = np.zeros((rows, cols), dtype=int)

        if direction == 'lr':
            # Place diagonal from (0, start_col) down-right
            # row = i, col = start_col + i
            for i in range(rows):
                grid[i, start_col + i] = color
        else:
            # direction == 'rl'
            # Place diagonal from (0, start_col) down-left
            # row = i, col = start_col - i
            for i in range(rows):
                grid[i, start_col - i] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid by filling any column that has a non-zero color
        with that color entirely.
        """
        out_grid = grid.copy()
        rows, cols = out_grid.shape

        for c in range(cols):
            # Check which colors appear in this column (excluding 0)
            unique_colors = np.unique(out_grid[:, c])
            unique_colors = unique_colors[unique_colors != 0]

            # In this particular puzzle setup, we expect at most one color per column
            if len(unique_colors) == 1:
                out_grid[:, c] = unique_colors[0]

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        1) Randomly choose a single rows x cols for all grids (with cols >= rows).
        2) Decide how many training examples (3 or 4) and fix 2 test examples.
        3) For each example, pick a unique color and a diagonal direction,
           ensuring at least one left->right and one right->left in both train and test.
        4) Generate input grids and transform them into output grids.
        5) Return (taskvars, train_test_data).
        """
        # 1) Randomly pick rows, cols with 3 <= rows <= 30 and cols >= rows
        rows = random.randint(3, 10)
        cols = random.randint(rows, 10)  # ensure cols >= rows for valid diagonals

        # 2) Decide how many training examples (3 or 4), then fix test examples = 2
        nr_train_examples = random.choice([3, 4])
        nr_test_examples = 2
        total_examples = nr_train_examples + nr_test_examples

        # 3) Directions needed: ensure at least 1 'lr' (left->right) and 1 'rl' (right->left)
        #    for both train and test sets.
        # We'll create a list of directions for the train set ensuring coverage,
        # then do the same for test.

        # For training, we must place at least one 'lr' and one 'rl'.
        # For test, also at least one 'lr' and one 'rl'.
        # We can distribute them in a simple pattern or randomize while ensuring coverage.

        # Start by creating the needed directions for train
        train_directions = []
        # Guarantee one 'lr' and one 'rl'
        train_directions.append('lr')
        train_directions.append('rl')
        # Fill the rest (if needed) randomly
        while len(train_directions) < nr_train_examples:
            train_directions.append(random.choice(['lr', 'rl']))
        random.shuffle(train_directions)

        # Do the same for test
        test_directions = ['lr', 'rl']
        # If we have exactly 2 test examples, that covers both directions.
        # If you later changed the number of test examples, you'd need to fill similarly.

        # Now pick distinct colors for each example
        # We have at most 6 examples total. We can safely pick from 9 possible colors (1..9).
        distinct_colors = random.sample(range(1, 10), total_examples)
        train_colors = distinct_colors[:nr_train_examples]
        test_colors = distinct_colors[nr_train_examples:]

        # We'll create gridvars for each example: color, direction, start_col
        def make_gridvars(color, direction):
            # For direction 'lr', we need start_col s.t. s + rows-1 < cols -> s <= cols - rows
            if direction == 'lr':
                max_start = cols - rows
                start = random.randint(0, max_start)
            else:
                # direction == 'rl'
                # need start_col >= (rows - 1)
                # so start in [rows-1, cols-1]
                start = random.randint(rows - 1, cols - 1)

            return dict(color=color, direction=direction, start_col=start)

        train_pairs = []
        for c, d in zip(train_colors, train_directions):
            gv = make_gridvars(c, d)
            inp = self.create_input({'rows': rows, 'cols': cols}, gv)
            out = self.transform_input(inp, {})
            train_pairs.append(GridPair(input=inp, output=out))

        # Make test grids
        test_pairs = []
        for c, d in zip(test_colors, test_directions):
            gv = make_gridvars(c, d)
            inp = self.create_input({'rows': rows, 'cols': cols}, gv)
            out = self.transform_input(inp, {})
            test_pairs.append(GridPair(input=inp, output=out))

        # Prepare the final dictionary of variables used in the template.
        # Because the template references {vars['rows']} and {vars['cols']}, we store them:
        taskvars = {
            'rows': rows,
            'cols': cols
        }

        # Combine train/test data
        train_test_data = TrainTestData(
            train=train_pairs,
            test=test_pairs
        )

        return taskvars, train_test_data


