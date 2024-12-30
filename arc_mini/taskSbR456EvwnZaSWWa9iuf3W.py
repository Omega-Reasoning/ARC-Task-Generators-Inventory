import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry

class TasktaskSbR456EvwnZaSWWa9iuf3WGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input matrix contains {color('cell_color1')} and {color('cell_color2')} cells, with the rest of the cells being empty (0)."
        ]
        reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and extending each {color('cell_color1')} cell vertically downwards, filling empty (0) cells with {color('cell_color1')} color, until a {color('cell_color2')} cell or the bottom edge of the matrix is reached."
        ]
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        cell_color1 = random.randint(1, 9)
        cell_color2 = random.randint(1, 9)
        while cell_color2 == cell_color1:
            cell_color2 = random.randint(1, 9)

        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2
        }

        nr_train = random.randint(3, 6)
        nr_test = 1
        data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, data

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]

        def generator():
            height = random.randint(5, 10)
            width = random.randint(5, 10)
            grid = np.zeros((height, width), dtype=int)

            num_color1 = random.randint(1, max(1, (height*width)//4))
            num_color2 = random.randint(1, max(1, (height*width)//4))

            for _ in range(num_color2):
                r = random.randint(0, height - 1)
                c = random.randint(0, width - 1)
                grid[r, c] = cell_color2

            for _ in range(num_color1):
                r = random.randint(1, height - 2)
                c = random.randint(1, width - 2)
                grid[r, c] = cell_color1

            return grid

        def predicate(g: np.ndarray):
            H, W = g.shape
            has_color1_interior = any(
                g[r, c] == cell_color1
                for r in range(1, H - 1)
                for c in range(1, W - 1)
            )
            has_color2 = (cell_color2 in g)
            return has_color1_interior and has_color2

        valid_grid = retry(generator, predicate, max_attempts=100)
        return valid_grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        color1 = taskvars["cell_color1"]
        color2 = taskvars["cell_color2"]
        out_grid = np.copy(grid)
        rows, cols = out_grid.shape

        for r in range(rows):
            for c in range(cols):
                if out_grid[r, c] == color1:
                    rr = r + 1
                    while rr < rows:
                        if out_grid[rr, c] == color2:
                            break
                        elif out_grid[rr, c] == 0:
                            out_grid[rr, c] = color1
                            rr += 1
                        else:
                            rr += 1

        return out_grid
