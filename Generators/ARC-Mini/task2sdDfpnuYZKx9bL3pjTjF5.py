from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task2sdDfpnuYZKx9bL3pjTjF5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains some multi-colored (1–9) cells, with the remaining cells being empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {(vars['grid_size'] * 2)} × {(vars['grid_size'] * 2)}.",
            "The output grids are constructed by expanding each cell into a 2×2 block of the same color.",
            "Each cell becomes a block, and the position of each block corresponds proportionally to its original position in the input."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Randomly color some cells with various colors
        color_palette = list(range(1, 10))  # Colors 1-9
        density = random.uniform(0.1, 0.4)  # 10-40% of cells colored
        
        random_cell_coloring(grid, color_palette, density=density, background=0)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        input_size = grid.shape[0]
        output_size = input_size * 2
        
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Expand each cell into a 2x2 block
        for r in range(input_size):
            for c in range(input_size):
                color = grid[r, c]
                # Map input cell (r,c) to output 2x2 block at (2*r, 2*c)
                output_grid[2*r:2*r+2, 2*c:2*c+2] = color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        grid_size = random.randint(5, 30)
        taskvars = {'grid_size': grid_size}
        
        # Generate 3-5 training examples and 1 test example
        num_train = random.randint(3, 5)
        train_test_data = self.create_grids_default(num_train, 1, taskvars)
        
        return taskvars, train_test_data

