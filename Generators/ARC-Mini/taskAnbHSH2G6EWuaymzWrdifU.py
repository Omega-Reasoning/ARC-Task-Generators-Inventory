import numpy as np
import random
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry, create_object
from Framework.transformation_library import GridObject

class TaskAnbHSH2G6EWuaymzWrdifUGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a completely filled grid with multi-colored (1-9) cells and no empty (0) cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and only swapping the first row with the last row of the grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate a filled grid with no two adjacent cells having the same color
        colors = list(range(1, 10))
        
        for r in range(rows):
            for c in range(cols):
                available_colors = colors[:]
                if r > 0 and grid[r - 1, c] in available_colors:
                    available_colors.remove(grid[r - 1, c])  # Remove color above
                if c > 0 and grid[r, c - 1] in available_colors:
                    available_colors.remove(grid[r, c - 1])  # Remove color left
                grid[r, c] = random.choice(available_colors)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()
        output_grid[0], output_grid[-1] = output_grid[-1].copy(), output_grid[0].copy()
        return output_grid
    
    def create_grids(self) -> tuple:
        taskvars = {
            'rows': random.randint(5, 30),
            'cols': random.randint(5, 30)
        }
        
        num_train_examples = random.randint(3, 6)
        num_test_examples = 1
        
        train_test_data = self.create_grids_default(num_train_examples, num_test_examples, taskvars)
        return taskvars, train_test_data
    

