import random
import numpy as np
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

class Task9DJPUL8gn2U73r7KneZzRYGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids are of size nxn, where n is an odd number.",
            "Each input grid contains diagonally connected cells, in either of the two possible diagonal directions; top-left to bottom-right or top-right to bottom-left.",
            "The cells are of {color('cell_color1')} color, with one {color('cell_color2')} cell positioned in the exact middle of the diagonal.",
            "The remaining cells in the input grid are empty (0)."
        ]
        reasoning_chain = [
            "To construct the output grid, copy the input grid and completely fill in the column containing {color('cell_color2')} cell, with {color('cell_color2')} color."
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
        
        def generate_example(diagonal_type=None):
            while True:
                n = random.randint(8, 30)
                if n % 2 == 1:
                    break
            gridvars = {"n": n, "diagonal_type": diagonal_type}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            return {
                "input": input_grid,
                "output": output_grid
            }
        
        examples = []
        examples.append(generate_example("main"))
        examples.append(generate_example("anti"))
        
        for _ in range(nr_train - 2):
            diagonal_type = random.choice(["main", "anti"])
            examples.append(generate_example(diagonal_type))
        
        test_example = [generate_example(None)]
        
        train_test_data = {
            "train": examples,
            "test": test_example
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]
        n = gridvars["n"]
        diagonal_type = gridvars.get("diagonal_type")
        if diagonal_type is None:
            diagonal_type = random.choice(["main", "anti"])
        
        grid = np.zeros((n, n), dtype=int)
        
        if diagonal_type == "main":
            for i in range(n):
                grid[i, i] = cell_color1
            mid = n // 2
            grid[mid, mid] = cell_color2
        else:
            for i in range(n):
                grid[i, n - 1 - i] = cell_color1
            mid = n // 2
            grid[mid, n - 1 - mid] = cell_color2
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        cell_color2 = taskvars["cell_color2"]
        output_grid = np.copy(grid)
        positions = np.argwhere(output_grid == cell_color2)
        if len(positions) == 0:
            return output_grid
        
        row_with_color2, col_with_color2 = positions[0]
        output_grid[:, col_with_color2] = cell_color2
        
        return output_grid

