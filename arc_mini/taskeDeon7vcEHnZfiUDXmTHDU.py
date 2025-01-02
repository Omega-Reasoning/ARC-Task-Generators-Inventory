import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, random_cell_coloring
from transformation_library import find_connected_objects

class TasktaskeDeon7vcEHnZfiUDXmTHDUGenerator(ARCTaskGenerator):
    def __init__(self):
        input_chain = [
            "Input grids can have different sizes.",
            "Each input matrix contains a completely filled row and column with {color('object_color')} color."
        ]
        transformation_chain = [
            "To construct the output matrix, fill only the cell at the intersection of the filled row and column with {color('object_color')} color."
        ]
        super().__init__(input_chain, transformation_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        grid = np.zeros((rows, cols), dtype=int)
        row_filled = random.randint(0, rows - 1)
        col_filled = random.randint(0, cols - 1)
        color_val = taskvars["object_color"]
        grid[row_filled, :] = color_val
        grid[:, col_filled] = color_val
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        color_val = taskvars["object_color"]
        rows, cols = grid.shape
        row_filled = [r for r in range(rows) if np.all(grid[r, :] == color_val)][0]
        col_filled = [c for c in range(cols) if np.all(grid[:, c] == color_val)][0]
        out_grid = np.zeros_like(grid)
        out_grid[row_filled, col_filled] = color_val
        return out_grid

    def create_grids(self):
        taskvars = {"object_color": random.randint(1, 9)}
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data
