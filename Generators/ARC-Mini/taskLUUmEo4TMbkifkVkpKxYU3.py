from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
class TaskLUUmEo4TMbkifkVkpKxYU3Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size 1x{vars['cols']}.",
            "They contain exactly two colored (1-9) rectangular objects, with no empty (0) cells.",
            "Both rectangular objects have a length of one, but they have different widths and colors (1-9)."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids have the same number of columns as the input grid, but the number of rows is equal to the width of the widest rectangle in the input grid.",
            "The output grids are created by copying the single input row and extending each rectangular object downward, while maintaining its original color.",
            "Each object extends downward until the number of rows it occupies equals its width, forming square-shaped objects, while the remaining cells remain empty (0)."
        ]

        # 3) Call the parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid of size (1 x cols). The row is partitioned into
        two rectangles of distinct widths (w1, w2) and distinct colors. 
        The entire row has no empty cells (0).
        """
        cols = taskvars['cols']  # total number of columns for this puzzle
        # We need two distinct widths w1, w2 such that w1 + w2 = cols and w1 != w2.
        # Because we want a valid puzzle, keep picking until w1 != w2:
        while True:
            w1 = random.randint(2, cols - 2)
            w2 = cols - w1
            if w1 != w2:
                break

        # Pick two different colors in [1..9]
        colorA = random.randint(1, 9)
        while True:
            colorB = random.randint(1, 9)
            if colorB != colorA:
                break

        # Construct the single-row grid
        # First w1 columns = colorA, next w2 columns = colorB
        grid = np.zeros((1, cols), dtype=int)
        grid[0, :w1] = colorA
        grid[0, w1:(w1 + w2)] = colorB

        # Store the chosen widths and colors for reference in gridvars (not strictly necessary,
        # but sometimes can be useful if we want to ensure variety across all examples)
        gridvars['w1'] = w1
        gridvars['w2'] = w2
        gridvars['colorA'] = colorA
        gridvars['colorB'] = colorB

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the single-row input grid into the multi-row output grid
        according to the transformation reasoning chain:
        - The output row count = width of the widest rectangle in the input.
        - Each rectangle extends downward by its width, forming a square in that region.
        - All other cells remain 0 (empty).
        """
        # The grid has shape (1, cols). We know it's composed of exactly two
        # contiguous color rectangles: the first w1 columns are colorA, next w2 columns are colorB.
        # Let's recover w1, w2, colorA, colorB by scanning the row, or from the grid directly.

        row = grid[0]  # single row
        cols = row.shape[0]

        # Identify the boundary between the first rectangle and the second
        # We expect exactly two color segments, so we can find the first column
        # where the color changes from row[0].
        colorA = row[0]
        # find boundary
        boundary_idx = 1
        while boundary_idx < cols and row[boundary_idx] == colorA:
            boundary_idx += 1
        colorB = row[boundary_idx]

        w1 = boundary_idx  # number of columns for object A
        w2 = cols - boundary_idx  # number of columns for object B

        # The output has row count = max(w1, w2) and the same number of columns = cols
        max_w = max(w1, w2)
        out_grid = np.zeros((max_w, cols), dtype=int)

        # For each row i in [0..max_w-1]:
        # - columns [0..w1-1] get colorA if i < w1
        # - columns [w1..(w1 + w2 -1)] get colorB if i < w2
        for i in range(max_w):
            # Fill colorA region if within vertical extension
            if i < w1:
                out_grid[i, :w1] = colorA
            # Fill colorB region if within vertical extension
            if i < w2:
                out_grid[i, w1:(w1 + w2)] = colorB

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Create multiple (3-6) training examples and 1 test example,
        each with a random 'cols' satisfying 5 <= cols <= 30.
        
        We'll pick a single 'cols' for the entire puzzle so that
        the input reasoning chain text {vars['cols']} is consistent,
        but for each training/test example, we generate different
        w1, w2, colorA, colorB for variety.
        """
        # 1) Choose a single 'cols' for the entire puzzle (5..30).
        #    We store it in taskvars so that {vars['cols']} is available for the chain text.
        #    We also ensure there's no chance of generating a puzzle with w1 = w2 = cols/2 
        #    by re-sampling if that occurs (but we actually handle that in create_input()).
        cols = random.randint(5, 30)

        taskvars = {'cols': cols}

        # 2) Decide how many training examples (3-6) and create them
        nr_train = random.randint(3, 6)
        train_pairs = []
        for _ in range(nr_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # 3) Create one test example
        test_pairs = []
        input_grid_test = self.create_input(taskvars, {})
        # The correct output for test is computed as well, but typically
        # ARC tasks show the test output as an empty or withheld solution.
        # We'll provide it for completeness in the data structure.
        output_grid_test = self.transform_input(input_grid_test, taskvars)
        test_pairs.append(GridPair(input=input_grid_test, output=output_grid_test))

        # 4) Return the dictionary of task variables and the train/test data
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)



