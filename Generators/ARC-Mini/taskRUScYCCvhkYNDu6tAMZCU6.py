from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskRUScYCCvhkYNDu6tAMZCU6Generator(ARCTaskGenerator):
    """
    Generator that creates grids with a single colored cell and transforms them
    by adding a frame of the same color around all four borders.
    """
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']} x {vars['cols']}.",
            "Each input grid contains exactly one single-colored (1-9) cell."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the single-colored cell.",
            "Using the color of this cell, create a one-cell-wide frame along all four borders of the grid.",
            "If the original single cell is in the interior (not on the border), remove it after creating the frame."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create input grid with exactly one single-colored cell."""
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Start with empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Place exactly one colored cell at random position
        cell_color = random.randint(1, 9)
        row_pos = random.randint(0, rows - 1)
        col_pos = random.randint(0, cols - 1)
        
        grid[row_pos, col_pos] = cell_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by adding frame using the color of the single cell."""
        rows, cols = grid.shape
        output = grid.copy()
        
        # Find the single colored cell
        colored_positions = np.where(grid != 0)
        if len(colored_positions[0]) > 0:
            # Get the position and color of the single cell
            cell_row = colored_positions[0][0]
            cell_col = colored_positions[1][0]
            cell_color = grid[cell_row, cell_col]
            
            # Create frame along all four borders
            # Top and bottom borders
            output[0, :] = cell_color  # Top row
            output[rows-1, :] = cell_color  # Bottom row
            
            # Left and right borders
            output[:, 0] = cell_color  # Left column
            output[:, cols-1] = cell_color  # Right column
            
            # Remove the original single cell if it's in the interior (not on border)
            is_on_border = (cell_row == 0 or cell_row == rows-1 or 
                           cell_col == 0 or cell_col == cols-1)
            if not is_on_border:
                output[cell_row, cell_col] = 0
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create train and test grids with randomized grid sizes."""
        # Generate random grid dimensions within constraints
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        
        taskvars = {
            'rows': rows,
            'cols': cols
        }
        
        # Create 3-5 training examples and 1 test example
        num_train = random.randint(3, 5)
        train_test_data = self.create_grids_default(num_train, 1, taskvars)
        
        return taskvars, train_test_data

