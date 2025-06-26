from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple

class Tasked36ccf7(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each grid contains a random number of colored cells, all uniformly colored with a single randomly chosen color, while all other cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by rotating the input grid 90 degrees counterclockwise."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables - these will be consistent across all examples in this task
        n = random.randint(5, 30)  # Grid size between 5 and 30
        
        taskvars = {
            'n': n
        }
        
        # Create 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        train_test_data = self.create_grids_default(num_train, 1, taskvars)
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        
        # Create empty grid
        grid = np.zeros((n, n), dtype=int)
        
        # Choose a random color for this specific grid (1-9, excluding 0 which is background)
        color = random.randint(1, 9)
        
        # Calculate density to ensure number of colored cells is less than total cells
        # We'll use a random density between 0.1 and 0.9 to ensure variety while staying under 100%
        max_density = 0.9  # Keep below 1.0 to satisfy constraint
        min_density = 0.1  # Ensure at least some cells are colored
        density = random.uniform(min_density, max_density)
        
        # Apply random coloring with the chosen color
        random_cell_coloring(grid, color, density=density, background=0, overwrite=False)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Rotate 90 degrees counterclockwise
        # np.rot90 with k=1 rotates counterclockwise by 90 degrees
        rotated_grid = np.rot90(grid, k=1)
        return rotated_grid

