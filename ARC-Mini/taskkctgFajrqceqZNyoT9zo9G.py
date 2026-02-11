from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
# Optionally use these libraries as per your instructions:
# from input_library import retry, create_object, Contiguity
# from transformation_library import find_connected_objects, GridObject, GridObjects

class TaskkctgFajrqceqZNyoT9zo9GGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (copied exactly as given)
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain a completely filled main diagonal (top-left to bottom-right) with either {color('object_color1')} or {color('object_color2')} cells, with the remaining cells being empty (0)."
        ]
        # 2) Transformation reasoning chain (copied exactly as given)
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and modifying the diagonal direction and color based on the input color and grid size.",
            "If the grid size is even, the diagonal direction remains the same; otherwise, the diagonal is flipped to the inverse diagonal (top-right to bottom-left).",
            "The diagonal color is determined based on the input grid color: {color('object_color1')} changes to {color('object_color3')}, and {color('object_color2')} changes to {color('object_color4')}."
        ]
        # 3) super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Create 3-4 train grids and 2 test grids fulfilling the given constraints:
         - Distinct colors object_color1..4
         - One train + test: even size, diagonal color=object_color1
         - One train + test: odd size, diagonal color=object_color2
         - One train: odd size, diagonal color=object_color1
        """
        # Randomly pick 4 distinct colors from 1..9
        distinct_colors = random.sample(range(1, 10), 4)
        taskvars = {
            "object_color1": distinct_colors[0],
            "object_color2": distinct_colors[1],
            "object_color3": distinct_colors[2],
            "object_color4": distinct_colors[3]
        }

        # Build our train & test data
        # We will create exactly 3 train examples and 2 test examples per the constraints.

        train_data = []
        test_data = []

        # 1) Train: even size, color1
        train_data.append(self._build_example(
            size=self._random_even_size(),
            diagonal_color=taskvars["object_color1"],
            taskvars=taskvars
        ))

        # 2) Train: odd size, color2
        train_data.append(self._build_example(
            size=self._random_odd_size(),
            diagonal_color=taskvars["object_color2"],
            taskvars=taskvars
        ))

        # 3) Train: odd size, color1
        train_data.append(self._build_example(
            size=self._random_odd_size(),
            diagonal_color=taskvars["object_color1"],
            taskvars=taskvars
        ))

        # Test: even size, color1
        test_data.append(self._build_example(
            size=self._random_even_size(),
            diagonal_color=taskvars["object_color1"],
            taskvars=taskvars
        ))
        # Test: odd size, color2
        test_data.append(self._build_example(
            size=self._random_odd_size(),
            diagonal_color=taskvars["object_color2"],
            taskvars=taskvars
        ))

        train_test_data = {
            "train": train_data,
            "test": test_data
        }

        return taskvars, train_test_data

    def _build_example(self, size: int, diagonal_color: int, taskvars) -> GridPair:
        """
        Utility to build one grid pair (input, output) with the given size and diagonal color,
        then apply the transformation.
        """
        # We pass these as gridvars for create_input
        gridvars = {
            "size": size,
            "diagonal_color": diagonal_color
        }
        inp = self.create_input(taskvars, gridvars)
        out = self.transform_input(inp, taskvars)
        return {
            "input": inp,
            "output": out
        }

    def _random_even_size(self) -> int:
        """
        Returns a random even integer between 5 and 30 (inclusive).
        The smallest even >=5 is 6, so choose from [6,8,10,...,30].
        """
        candidates = [x for x in range(6, 31) if x % 2 == 0]
        return random.choice(candidates)

    def _random_odd_size(self) -> int:
        """
        Returns a random odd integer between 5 and 30 (inclusive).
        So choose from [5,7,9,...,29].
        """
        candidates = [x for x in range(5, 31) if x % 2 == 1]
        return random.choice(candidates)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Generate a square input grid of size gridvars['size'] with a main
        diagonal filled by gridvars['diagonal_color'] and zeros elsewhere.
        """
        size = gridvars['size']
        diag_col = gridvars['diagonal_color']

        grid = np.zeros((size, size), dtype=int)
        # Fill the main diagonal with diag_col
        for i in range(size):
            grid[i, i] = diag_col
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the grid according to:
         - If size is even, keep the same main diagonal.
         - If size is odd, flip to the inverse diagonal.
         - Replace diagonal color from color1->color3 or color2->color4.
        """
        # For clarity, read out the color variables
        c1 = taskvars["object_color1"]
        c2 = taskvars["object_color2"]
        c3 = taskvars["object_color3"]
        c4 = taskvars["object_color4"]

        size = grid.shape[0]
        out = np.copy(grid)

        if size % 2 == 0:
            # Even size: same main diagonal
            for i in range(size):
                if out[i, i] == c1:
                    out[i, i] = c3
                elif out[i, i] == c2:
                    out[i, i] = c4
        else:
            # Odd size: flip to the inverse diagonal
            # First, clear out the main diagonal
            for i in range(size):
                out[i, i] = 0
            # Then fill inverse diagonal with updated color
            for i in range(size):
                # If original grid[i, i] had c1 or c2, we
                # copy that new color to the inverse diagonal
                original_color = grid[i, i]
                if original_color == c1:
                    out[i, size - 1 - i] = c3
                elif original_color == c2:
                    out[i, size - 1 - i] = c4

        return out


