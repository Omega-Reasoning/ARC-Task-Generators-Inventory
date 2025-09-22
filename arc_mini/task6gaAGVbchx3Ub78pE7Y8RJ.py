from arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task6gaAGVbchx3Ub78pE7Y8RJGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains exactly two vertical strips of different sizes, placed in two different columns.",
            "One strip is of {color('strip1')} color and the other is of {color('strip2')} color.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by changing the positions of the two vertical strips.",
            "In this process, each vertical strip is transformed into a horizontal strip: the strip originally placed in the i-th column is shifted to the i-th row.",
            "Each strip is then moved horizontally to the left so that it is aligned to the left boundary of the grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Get strip parameters from gridvars
        strip1_col = gridvars['strip1_col']
        strip2_col = gridvars['strip2_col']
        strip1_start = gridvars['strip1_start']
        strip1_end = gridvars['strip1_end']
        strip2_start = gridvars['strip2_start']
        strip2_end = gridvars['strip2_end']
        
        # Place first vertical strip
        for row in range(strip1_start, strip1_end + 1):
            grid[row, strip1_col] = taskvars['strip1']
            
        # Place second vertical strip
        for row in range(strip2_start, strip2_end + 1):
            grid[row, strip2_col] = taskvars['strip2']
            
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        output_grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Find vertical strips and convert to horizontal
        for col in range(grid_size):
            # Check if this column contains a strip
            strip_rows = []
            strip_color = 0
            
            for row in range(grid_size):
                if grid[row, col] != 0:
                    strip_rows.append(row)
                    strip_color = grid[row, col]
            
            if strip_rows:
                # Create horizontal strip in the same row as the column index
                # aligned to the left boundary
                strip_length = len(strip_rows)
                for i in range(strip_length):
                    if i < grid_size:  # Ensure we don't go out of bounds
                        output_grid[col, i] = strip_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(5, 15),
            'strip1': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'strip2': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        # Ensure strips have different colors
        while taskvars['strip2'] == taskvars['strip1']:
            taskvars['strip2'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # Generate training examples
        train_examples = []
        num_train = random.randint(3, 5)
        
        for _ in range(num_train):
            # Generate gridvars for this example
            grid_size = taskvars['grid_size']
            
            # Choose two different columns for the strips
            strip1_col = random.randint(0, grid_size - 1)
            strip2_col = random.randint(0, grid_size - 1)
            while strip2_col == strip1_col:
                strip2_col = random.randint(0, grid_size - 1)
            
            # Generate strip lengths and positions (ensure at least 2 cells each)
            # Strip 1
            strip1_length = random.randint(2, grid_size)
            strip1_start = random.randint(0, grid_size - strip1_length)
            strip1_end = strip1_start + strip1_length - 1
            
            # Strip 2
            strip2_length = random.randint(2, grid_size)
            strip2_start = random.randint(0, grid_size - strip2_length)
            strip2_end = strip2_start + strip2_length - 1
            
            gridvars = {
                'strip1_col': strip1_col,
                'strip2_col': strip2_col,
                'strip1_start': strip1_start,
                'strip1_end': strip1_end,
                'strip2_start': strip2_start,
                'strip2_end': strip2_end
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        grid_size = taskvars['grid_size']
        
        # Choose two different columns for the strips
        strip1_col = random.randint(0, grid_size - 1)
        strip2_col = random.randint(0, grid_size - 1)
        while strip2_col == strip1_col:
            strip2_col = random.randint(0, grid_size - 1)
        
        # Generate strip lengths and positions (ensure at least 2 cells each)
        strip1_length = random.randint(2, grid_size)
        strip1_start = random.randint(0, grid_size - strip1_length)
        strip1_end = strip1_start + strip1_length - 1
        
        strip2_length = random.randint(2, grid_size)
        strip2_start = random.randint(0, grid_size - strip2_length)
        strip2_end = strip2_start + strip2_length - 1
        
        test_gridvars = {
            'strip1_col': strip1_col,
            'strip2_col': strip2_col,
            'strip1_start': strip1_start,
            'strip1_end': strip1_end,
            'strip2_start': strip2_start,
            'strip2_end': strip2_end
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
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


