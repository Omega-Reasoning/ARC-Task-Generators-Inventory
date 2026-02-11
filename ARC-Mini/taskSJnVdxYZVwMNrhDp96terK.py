from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskSJnVdxYZVwMNrhDp96terKGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input matrix contains a {color('cell_color1')} cell, with the remaining cells being empty (0)."
        ]
        reasoning_chain = [
            "To construct the output matrix, copy the input matrix and color the entire row and column of the {color('cell_color1')} cell with {color('cell_color1')} color."
        ]
        super().__init__(observation_chain, reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        cell_color = taskvars['cell_color1']
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        grid = np.zeros((rows, cols), dtype=int)
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        grid[r, c] = cell_color
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        cell_color = taskvars['cell_color1']
        new_grid = grid.copy()
        coords = np.where(grid == cell_color)
        if len(coords[0]) == 0:
            return new_grid
        r, c = coords[0][0], coords[1][0]
        new_grid[r, :] = cell_color
        new_grid[:, c] = cell_color
        return new_grid

    def create_grids(self) -> (dict, TrainTestData):
        cell_color1 = random.randint(1, 9)
        taskvars = {'cell_color1': cell_color1}
        nr_train = random.randint(3, 6)
        nr_test = 1
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        return taskvars, train_test_data
