from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskQfFKb3t2kUaEQQBKerMhjtGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid is completely filled with multi-colored (1-9) cells.",
            "The arrangement of these multi-colored cells varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and emptying all colored cells that lie below the main diagonal (from top-left to bottom-right).",
            "All cells on or above the main diagonal remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid completely filled with multi-colored cells (1-9)."""
        size = taskvars['grid_size']
        
        # Create a grid filled with random colors from 1-9 (no empty cells)
        grid = np.random.randint(1, 10, size=(size, size))
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by emptying cells below the main diagonal."""
        output_grid = grid.copy()
        
        # Empty all cells below the main diagonal (set to 0)
        for i in range(output_grid.shape[0]):
            for j in range(output_grid.shape[1]):
                if i > j:  # Below main diagonal
                    output_grid[i, j] = 0
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and generate train/test grids."""
        # Randomly choose grid size between 5 and 30
        grid_size = random.randint(5, 30)
        
        # Randomly choose number of training examples (3-5)
        num_train = random.randint(3, 5)
        
        taskvars = {
            'grid_size': grid_size
        }
        
        # Generate training examples
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

