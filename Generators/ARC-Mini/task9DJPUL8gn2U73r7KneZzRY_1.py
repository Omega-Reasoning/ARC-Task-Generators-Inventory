# diagonal_filling_task.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
class Task9DJPUL8gn2U73r7KneZzRY_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "All input grids are squares.",
            "Let n be the height and width of an input grid, where n is an odd number.",
            "Each input grid contains diagonally connected cells in one of the two possible directions: top-left to bottom-right or top-right to bottom-left, with the remaining cells being empty.",
            "All diagonal cells are of {color('diagonal_color')} color, with a single {color('middlecell_color')} cell positioned exactly in the middle of the diagonal."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling either the middle column or middle row, depending on the diagonal direction.",
            "If the diagonal direction is top-left to bottom-right, the entire middle column is filled; otherwise, the middle row is filled.",
            "All filled cells are of {color('middlecell_color')} color."
        ]

        # 3) Superclass initialisation
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain given the taskvars and gridvars.

        The grid is n x n (n odd). We fill either the main diagonal (TL-BR) or 
        the anti-diagonal (TR-BL). All diagonal cells are diagonal_color except 
        the very middle cell which is middlecell_color.
        """
        n = gridvars['n']  # size of the square grid
        direction = gridvars['direction']
        diagonal_color = taskvars['diagonal_color']
        middlecell_color = taskvars['middlecell_color']

        grid = np.zeros((n, n), dtype=int)  # start empty

        if direction == 'TLBR':
            # top-left to bottom-right diagonal
            for i in range(n):
                grid[i, i] = diagonal_color
            grid[n // 2, n // 2] = middlecell_color
        else:
            # top-right to bottom-left diagonal
            for i in range(n):
                grid[i, n - 1 - i] = diagonal_color
            grid[n // 2, n - 1 - (n // 2)] = middlecell_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain.

        We detect the diagonal direction by checking if grid[0, 0] == diagonal_color.
        If so, we fill the middle column. Otherwise, fill the middle row.
        """
        n = grid.shape[0]
        diagonal_color = taskvars['diagonal_color']
        middlecell_color = taskvars['middlecell_color']

        # Copy the grid to avoid mutating the input in place
        output_grid = grid.copy()

        # Detect direction:
        # If the top-left corner is diagonal_color, the diagonal is TL->BR,
        # otherwise it must be TR->BL.
        if output_grid[0, 0] == diagonal_color:
            # Fill the middle column
            for row in range(n):
                output_grid[row, n // 2] = middlecell_color
        else:
            # Fill the middle row
            for col in range(n):
                output_grid[n // 2, col] = middlecell_color

        return output_grid

    def create_grids(self) -> tuple:
        """
        Initialise task variables (diagonal_color, middlecell_color) used in templates 
        and then create train/test data grids.

        We produce 3-6 train examples and exactly 2 test examples, ensuring
        at least one example from each diagonal direction in both train and test sets.
        """

        # Pick two different colors in [1..9]
        # (We do not allow 0 because 0 is 'empty')
        diagonal_color = random.randint(1, 9)
        while True:
            middlecell_color = random.randint(1, 9)
            if middlecell_color != diagonal_color:
                break

        taskvars = {
            "diagonal_color": diagonal_color,
            "middlecell_color": middlecell_color
        }

        # We want 3-6 training examples
        nr_train = random.randint(3, 6)

        train_pairs = []
        directions_used_train = set()

        # Helper to create a single (input, output) pair
        def make_pair(direction: str):
            # Random odd n in [5..30], though we can clamp to a smaller range for practicality
            # to ensure variety but also not too big
            n = random.choice([x for x in range(5, 31) if x % 2 == 1])
            inp = self.create_input(taskvars, {"n": n, "direction": direction})
            out = self.transform_input(inp, taskvars)
            return {"input": inp, "output": out}

        # We must ensure at least one top-left->bottom-right ("TLBR")
        # and at least one top-right->bottom-left ("TRBL") in training.
        # We'll first create guaranteed pairs:
        forced_directions = ["TLBR", "TRBL"]

        for forced_dir in forced_directions:
            pair = make_pair(forced_dir)
            train_pairs.append(pair)
            directions_used_train.add(forced_dir)

        # We have already 2 training pairs; we add the rest up to nr_train randomly
        while len(train_pairs) < nr_train:
            direction = random.choice(["TLBR", "TRBL"])
            pair = make_pair(direction)
            train_pairs.append(pair)
            directions_used_train.add(direction)

        # Similarly, create 2 test examples, ensuring one of each direction
        test_pairs = []

        # We want exactly one "TLBR" and one "TRBL" for the test set
        for direction in ["TLBR", "TRBL"]:
            pair = make_pair(direction)
            test_pairs.append(pair)

        train_test_data = {
            "train": train_pairs,
            "test": test_pairs
        }

        return taskvars, train_test_data


