from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity
from transformation_library import GridObject
import numpy as np
import random
from typing import Dict, Any, Tuple

class TaskcLpAqHMFgGScEdjba3Qv84newGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "The input grids have several same-colored (1-9) cells in each row, with the remaining cells being empty (0)."
        ]
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid and for each row, fill all the empty (0) cells between colored cells with the same color.",
            "All the other empty (0) cells remain the same."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = random.randint(5, 15)
        cols = random.randint(5, 15)
        grid = np.zeros((rows, cols), dtype=int)
        
        for r in range(rows):
            color = random.randint(1, 9)
            num_cells = random.randint(2, cols // 2)
            positions = sorted(random.sample(range(cols), num_cells))
            grid[r, positions] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()
        rows, cols = grid.shape

        for r in range(rows):
            non_zero_indices = np.where(grid[r] != 0)[0]
            if len(non_zero_indices) > 1:
                for i in range(len(non_zero_indices) - 1):
                    start, end = non_zero_indices[i], non_zero_indices[i + 1]
                    output_grid[r, start:end + 1] = grid[r, start]

        return output_grid

    def create_grids(self) -> tuple:
        taskvars = {}
        num_train_examples = random.randint(2, 3)
        num_test_examples = 1

        train_test_data = self.create_grids_default(num_train_examples, num_test_examples, taskvars)

        return taskvars, train_test_data


