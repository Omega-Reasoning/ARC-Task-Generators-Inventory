from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object
from Framework.transformation_library import GridObject
import numpy as np
import random

class Taskj6sRkV2XCqM85vAbXp93jFGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size nxn.",
            "In each input grid, there are four same-colored (1-9) connected cells on the main diagonal (top-left to bottom-right). The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid and extend the diagonal cells with the same color in both directions as needed to reach the corners of the grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        n = taskvars['n']
        color = taskvars['color']
        grid = np.zeros((n, n), dtype=int)
        
        # Place 4 diagonal cells in main diagonal direction
        start_idx = random.randint(0, n - 4)
        
        for i in range(4):
            grid[start_idx + i, start_idx + i] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        n = grid.shape[0]
        color = taskvars['color']
        
        for i in range(n):
            if grid[i, i] == color:
                for j in range(i + 1, n):
                    grid[j, j] = color
                for j in range(i - 1, -1, -1):
                    grid[j, j] = color
                break
        
        return grid

    def create_grids(self) -> tuple:
        n = random.randint(8, 30)
        color = random.randint(1, 9)
        taskvars = {'n': n, 'color': color}
        
        train_pairs = [
            GridPair(
                input=(input_grid := self.create_input(taskvars, {})),
                output=self.transform_input(input_grid.copy(), taskvars)
            ) for _ in range(random.randint(2, 3))
        ]
        
        test_pair = GridPair(
            input=(test_input_grid := self.create_input(taskvars, {})),
            output=self.transform_input(test_input_grid.copy(), taskvars)
        )
        
        train_test_data = TrainTestData(train=train_pairs, test=[test_pair])
        return taskvars, train_test_data

