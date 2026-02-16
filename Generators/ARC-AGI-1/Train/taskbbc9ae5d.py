import numpy as np
import random
from typing import Dict, Any, Tuple, List
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

class Taskbbc9ae5dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size 1xn, where n is an even integer in the range [6, 30].",
            "A contiguous segment of cells, starting at the leftmost position, is assigned with a random color.",
            "The segment length is chosen randomly and satisfies 1 ≤ length ≤ n - n//{vars['m']} + 1"
        ]
        
        transformation_reasoning_chain = [
            "The output grid is first constructed as an empty grid with the same number of columns as the input grid, and with a number of rows equal to n//{vars['m']}.",
            "The first row of the output grid is filled with an exact copy of the input grid.",
            "For each subsequent row, the previous row is duplicated, and one additional cell to the right of the existing colored segment is filled with the same color.",
            "This process continues until the right boundary of the output grid is reached."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, 
                     taskvars: Dict[str, Any], 
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """Create a 1xn input grid with a colored segment starting from the left."""
        n = gridvars['n']
        color = gridvars['color']
        m = taskvars['m']
        
        # Create 1xn grid filled with background (0)
        grid = np.zeros((1, n), dtype=int)
        
        # Determine segment length: 1 ≤ length ≤ n - n//m + 1
        max_length = n - n // m + 1
        if max_length < 1:
            max_length = 1
        segment_length = random.randint(1, max_length)
        
        # Fill the leftmost segment_length cells with the random color
        grid[0, :segment_length] = color
        
        return grid

    def transform_input(self, 
                       grid: np.ndarray, 
                       taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input grid according to the progressive expansion rule."""
        n = grid.shape[1]
        m = taskvars['m']
        
        # Find the color used in the input (first non-zero value)
        color = None
        for val in grid[0, :]:
            if val != 0:
                color = val
                break
        
        if color is None:
            # Should not happen, but handle edge case
            return grid
        
        # Output grid dimensions: (n // m) rows x n columns
        output_height = n // m
        output = np.zeros((output_height, n), dtype=int)
        
        # First row is an exact copy of the input
        output[0, :] = grid[0, :]
        
        # For each subsequent row, duplicate previous row and extend by one cell
        for row_idx in range(1, output_height):
            output[row_idx, :] = output[row_idx - 1, :]
            
            # Find the rightmost colored cell in the previous row
            colored_cells = np.where(output[row_idx, :] == color)[0]
            if len(colored_cells) > 0:
                rightmost_colored = colored_cells[-1]
                # Extend by one cell to the right if within bounds
                if rightmost_colored + 1 < n:
                    output[row_idx, rightmost_colored + 1] = color
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and train/test grids."""
        # Initialize task variables
        taskvars = {
            'm': random.choice([2, 3, 5, 6])  # Divisor for grid width
        }
        
        # Create 3-6 training examples
        num_train = random.randint(3, 6)
        train_pairs = []
        
        for _ in range(num_train):
            # Generate n such that n % m == 0 and n is in [6, 30]
            m = taskvars['m']
            # Find valid values of n
            valid_n = [n for n in range(6, 31) if n % m == 0]
            n = random.choice(valid_n)
            
            # Each grid gets a random color
            gridvars = {
                'n': n,
                'color': random.randint(1, 9)
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create 1 test example
        m = taskvars['m']
        valid_n = [n for n in range(6, 31) if n % m == 0]
        n = random.choice(valid_n)
        
        gridvars = {
            'n': n,
            'color': random.randint(1, 9)
        }
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_pairs = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }
        
        return taskvars, train_test_data


