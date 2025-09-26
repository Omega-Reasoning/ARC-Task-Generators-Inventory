# arc_task_generator_subclass.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

# Import transformation functions if needed
# from transformation_library import find_connected_objects, GridObject, GridObjects

class TaskQ7PVog2EvMiogz6MUD9g8Generator(ARCTaskGenerator):
    def __init__(self):
        # Initialize the input reasoning chain
        input_reasoning_chain = [
            "All input grids are of size nxn, where n is an odd number.",
            "In each input matrix, either the middle column or the middle row is completely filled with {color('object_color')} color.",
            "The remaining cells are empty (0)."
        ]
        
        # Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "To construct the output grid, initialize an empty (0) grid, of the same size as the input grid.",
            "If the middle row is filled in the input grid, color the main diagonal (top-left to bottom-right) in the output grid with the same color.",
            "If the middle column is filled in the input grid, color the inverse diagonal (top-right to bottom-left) in the output grid with the same color.",
            "All other cells of the output grid are set to empty (0)."
        ]
        
        # Call the superclass initializer
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain given the task and grid variables.
        """
        size = gridvars['size']
        object_color = taskvars['object_color']
        fill_type = gridvars['fill']  # Either 'row' or 'column'
        
        # Initialize empty grid
        grid = np.zeros((size, size), dtype=int)
        
        if fill_type == 'row':
            middle = size // 2
            grid[middle, :] = object_color
        else:
            middle = size // 2
            grid[:, middle] = object_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain, producing an output grid.
        """
        size = grid.shape[0]
        object_color = taskvars['object_color']
        output_grid = np.zeros_like(grid)
        middle = size // 2
        
        # Check if the middle row is filled
        if np.all(grid[middle, :] == object_color):
            # Color the main diagonal
            np.fill_diagonal(output_grid, object_color)
        elif np.all(grid[:, middle] == object_color):
            # Color the inverse diagonal
            for i in range(size):
                output_grid[i, size - 1 - i] = object_color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Initialize task variables and create train and test data grids.
        """
        # Define task variables
        taskvars = {
            'object_color': random.randint(1, 9)  # Color from 1 to 9
        }
        
        train_test_data = {
            'train': [],
            'test': []
        }
        
        # Randomly decide the number of training examples (3-6)
        nr_train = random.randint(2, 3)
        
        # Create training examples
        for _ in range(nr_train):
            gridvars = self._generate_gridvars()
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            # Ensure the output grid is different from the input grid
            if not np.array_equal(input_grid, output_grid):
                train_test_data['train'].append({
                    'input': input_grid,
                    'output': output_grid
                })
            else:
                # Retry if transformation leads to identical grid
                input_grid = self.create_input(taskvars, gridvars)
                output_grid = self.transform_input(input_grid, taskvars)
                train_test_data['train'].append({
                    'input': input_grid,
                    'output': output_grid
                })
        
        # Create one test example
        gridvars = self._generate_gridvars()
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        train_test_data['test'].append({
            'input': test_input,
            'output': test_output
        })
        
        return taskvars, train_test_data
    
    def _generate_gridvars(self) -> Dict[str, Any]:
        """
        Generate grid-specific variables such as size and fill type.
        Ensures that size is an odd number between 5 and 20.
        Fill type is randomly chosen between 'row' and 'column'.
        """
        size = random.choice([n for n in range(5, 31) if n % 2 == 1])
        fill = random.choice(['row', 'column'])
        return {
            'size': size,
            'fill': fill
        }
    
    @staticmethod
    def visualize_train_test_data(train_test_data: TrainTestData):
        """
        Visualize the train and test grids using the superclass method.
        """
        super(ARCTaskGenerator, MiddleToDiagonalTaskGenerator).visualize_train_test_data(train_test_data)




