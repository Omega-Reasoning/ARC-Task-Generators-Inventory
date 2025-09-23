from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from typing import Dict, Any, Tuple
import numpy as np
import random

class Task6HgqM5w6nkwmrRdQBqWYMoGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains exactly one {color('block_color')} 3×3 block; all remaining cells are empty (0).",
            "The position of the block varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by transforming the 3×3 block from the input into a plus shape.",
            "The plus shape is obtained by removing the four corner cells of the 3×3 block.",
            "The remaining cells (center, top, bottom, left, right) form the plus shape."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        block_color = taskvars['block_color']
        
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Choose random position for 3x3 block (ensuring it fits in grid)
        max_row = grid_size - 3
        max_col = grid_size - 3
        start_row = random.randint(0, max_row)
        start_col = random.randint(0, max_col)
        
        # Place 3x3 block
        grid[start_row:start_row+3, start_col:start_col+3] = block_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        
        # Find the 3x3 block by looking for non-zero values
        rows, cols = np.nonzero(grid)
        if len(rows) > 0:
            # Find bounding box of the block
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Remove corner cells to create plus shape
            # Top-left corner
            output_grid[min_row, min_col] = 0
            # Top-right corner  
            output_grid[min_row, max_col] = 0
            # Bottom-left corner
            output_grid[max_row, min_col] = 0
            # Bottom-right corner
            output_grid[max_row, max_col] = 0
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Randomly choose task variables
        grid_size = random.randint(5, 15)
        block_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Any color except 0 (background)
        
        taskvars = {
            'grid_size': grid_size,
            'block_color': block_color
        }
        
        # Generate 3-5 training examples and 1 test example
        num_train = random.randint(3, 5)
        train_test_data = self.create_grids_default(num_train, 1, taskvars)
        
        return taskvars, train_test_data

