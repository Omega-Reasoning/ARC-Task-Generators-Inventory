from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object
from Framework.transformation_library import GridObject

class Taskc2EqtyLfzvRZwKATbRsvpyGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They only contain four {color('cell_color')} cells, arranged in a way that they define the four corners of a rectangle.",
            "The remaining cells are empty."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling in the empty (0) cells, which are part of the rectangle, with {color('cell_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        grid = np.zeros((height, width), dtype=int)
        
        # Choose rectangle dimensions
        min_row = random.randint(0, height - 3)
        max_row = random.randint(min_row + 2, height - 1)
        min_col = random.randint(0, width - 3)
        max_col = random.randint(min_col + 2, width - 1)
        
        # Assign corner points
        cell_color = taskvars['cell_color']
        grid[min_row, min_col] = cell_color
        grid[min_row, max_col] = cell_color
        grid[max_row, min_col] = cell_color
        grid[max_row, max_col] = cell_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()
        cell_color = taskvars['cell_color']
        
        # Get corner points
        corner_positions = np.argwhere(output_grid == cell_color)
        min_row, min_col = corner_positions.min(axis=0)
        max_row, max_col = corner_positions.max(axis=0)
        
        # Fill the rectangle with the given color
        output_grid[min_row:max_row+1, min_col:max_col+1] = cell_color
        
        return output_grid
    
    def create_grids(self) -> tuple:
        taskvars = {'cell_color': random.randint(1, 9)}  # Random color assignment
        train_test_data = {
            'train': [],
            'test': []
        }
        
        num_train_examples = random.randint(3, 6)
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_test_data['train'].append({'input': input_grid, 'output': output_grid})
        
        # Generate a single test example
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['test'].append({'input': input_grid, 'output': output_grid})
        
        return taskvars, train_test_data


