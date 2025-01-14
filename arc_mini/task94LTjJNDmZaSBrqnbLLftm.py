from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import GridObject
import numpy as np
import random

class Task94LTjJNDmZaSBrqnbLLftmGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size 2x{vars['col']}.",
            "They contain a completely filled first row, with multi-colored (1-9) cells.",
            "The second row is completely empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and expanding each colored cell in the first row (except the cell at position (0,0)) diagonally to the bottom-left in the second row, maintaining the color of the original cell.",
            "The last cell in the second row is always {color('cell_color')}.",
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        col = taskvars['col']
        grid = np.zeros((2, col), dtype=int)

        # Fill the first row with random colors
        grid[0, :] = [random.randint(1, 9) for _ in range(col)]
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        col = taskvars['col']
        cell_color = taskvars['cell_color']

        output_grid = grid.copy()

        # Expand colored cells diagonally to the bottom-left in the second row
        for c in range(1, col):
            output_grid[1, c - 1] = grid[0, c]

        # Set the last cell in the second row
        output_grid[1, -1] = cell_color

        return output_grid

    def create_grids(self) -> tuple:
        # Randomly initialize task variables
        col = random.randint(3, 9)
        cell_color = random.randint(1, 9)

        taskvars = {
            'col': col,
            'cell_color': cell_color
        }

        train_test_data = self.create_grids_default(3, 1, taskvars)

        return taskvars, train_test_data


