from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random

class Taskbbc9ae5d(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size 1xn, where n is an even integer in the range [6, 30].",
            "A contiguous segment of cells, starting at the leftmost position, is assigned the color {color('color_1')}.",
            "The segment length is chosen randomly and satisfies 1 ≤ length < n//2+1."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is first constructed as an empty grid with the same number of columns as the input grid, and with a number of rows equal to half the number of columns.",
            "The first row of the output grid is filled with an exact copy of the input grid.",
            "For each subsequent row, the previous row is duplicated, and one additional cell to the right of the existing colored segment is filled with the same color.",
            "This process continues until the right boundary of the output grid is reached."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        """Create a 1×n grid with a colored segment starting from the left."""
        n = gridvars['n']
        segment_length = gridvars['segment_length']
        color_1 = taskvars['color_1']
        
        # Create 1×n grid
        grid = np.zeros((1, n), dtype=int)
        
        # Fill the leftmost segment_length cells with color_1
        grid[0, :segment_length] = color_1
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """Transform input according to the expansion rules."""
        input_height, n = grid.shape
        output_height = n // 2
        
        # Create output grid
        output = np.zeros((output_height, n), dtype=int)
        
        # First row is exact copy of input
        output[0, :] = grid[0, :]
        
        # For each subsequent row, extend the colored segment by one cell
        for row in range(1, output_height):
            # Copy previous row
            output[row, :] = output[row-1, :]
            
            # Find the rightmost colored cell in the previous row
            colored_positions = np.where(output[row-1, :] != 0)[0]
            if len(colored_positions) > 0:
                rightmost_colored = colored_positions[-1]
                # If there's space to extend right, add one more colored cell
                if rightmost_colored + 1 < n:
                    color_to_use = output[row-1, rightmost_colored]  # Use same color
                    output[row, rightmost_colored + 1] = color_to_use
        
        return output
    
    def create_grids(self):
        """Create task variables and generate train/test grids."""
        
        # Generate task variables
        taskvars = {
            'color_1': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Any color except 0 (background)
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Generate even n in range [6, 30]
            n = random.choice([i for i in range(6, 31) if i % 2 == 0])
            
            # Segment length satisfies 1 ≤ length < n//2+1
            max_segment_length = n // 2
            segment_length = random.randint(1, max_segment_length)
            
            gridvars = {
                'n': n,
                'segment_length': segment_length
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        n = random.choice([i for i in range(6, 31) if i % 2 == 0])
        max_segment_length = n // 2
        segment_length = random.randint(1, max_segment_length)
        
        gridvars = {
            'n': n,
            'segment_length': segment_length
        }
        
        test_input = self.create_input(taskvars, gridvars)
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


