from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry

class Taskd13f3404Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grid is of size {vars['n']} x {vars['n']}.",
            "{vars['n']} random cells are colored with {vars['n']} different colors. The remaining cells are empty (0).",
            "For every colored cell, all cells located diagonally down and to the right from it must either be empty or lie outside the grid boundaries."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is of size of {2*vars['n']} x {2*vars['n']}",
            "It is constructed by first copying the input grid into the {vars['n']} x {vars['n']} cells at the upper-left corner of the output grid.",
            "For every colored cell in the input grid, all cells located diagonally down and to the right from it—at equal steps—are also colored with the same color, as long as they remain within the boundaries of the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        n = taskvars['n']
        grid_size = n 
        
        def generate_valid_grid():
            # Create an empty grid
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Generate n distinct colors (range 1-9)
            colors = random.sample(range(1, 10), n)
            
            # Place n colored cells in valid positions
            positions = []
            
            for i, color in enumerate(colors):
                # Try to find a valid position for this color
                max_attempts = 100
                for _ in range(max_attempts):
                    # Generate a random position
                    row = random.randint(0, grid_size - 1)
                    col = random.randint(0, grid_size - 1)
                    
                    # Check if this position is valid
                    if grid[row, col] != 0:
                        continue  # Position already occupied
                    
                    # Check if this position conflicts with diagonals from existing colored cells
                    # or if existing colored cells conflict with diagonals from this position
                    valid = True
                    
                    # Check if any existing colored cell has this position in its diagonal path
                    for prev_row, prev_col in positions:
                        # Checking if current position is in diagonal of any previous position
                        if prev_row < row and prev_col < col:  # Only check lower-right diagonals
                            diff_row = row - prev_row
                            diff_col = col - prev_col
                            if diff_row == diff_col:  # If on same diagonal
                                valid = False
                                break
                    
                    if not valid:
                        continue
                        
                    # Check if this position will have any existing colored cell in its diagonal path
                    # We need to check all points diagonally down-right from this position
                    for dr in range(1, grid_size):
                        new_row, new_col = row + dr, col + dr
                        if new_row >= grid_size or new_col >= grid_size:
                            break  # Outside grid bounds
                        
                        if grid[new_row, new_col] != 0:
                            valid = False
                            break
                    
                    if valid:
                        grid[row, col] = color
                        positions.append((row, col))
                        break
                
                # If we couldn't find a valid position after max attempts, return None
                if len(positions) != i + 1:
                    return None
            
            return grid
        
        # Generate grid until a valid one is produced
        return retry(
            generate_valid_grid,
            lambda x: x is not None
        )
    
    def transform_input(self, grid, taskvars):
        n = taskvars['n']
        input_size = n 
        output_size = 2 * n 
        
        # Create a larger output grid
        output = np.zeros((output_size, output_size), dtype=int)
        
        # Copy the input grid to the upper-left corner
        output[:input_size, :input_size] = grid
        
        # Propagate each color diagonally
        for r in range(input_size):
            for c in range(input_size):
                color = grid[r, c]
                if color > 0:  # If this is a colored cell
                    # Propagate diagonally down and right
                    diag_length = min(output_size - r, output_size - c)
                    for step in range(diag_length):
                        output[r + step, c + step] = color
        
        return output
    
    def create_grids(self):
        # Set taskvars
        taskvars = {'n': random.randint(3, 9)}
        
        # Generate 3-6 train examples
        num_train = random.randint(3, 6)
        
        def generate_examples(n):
            return [
                {
                    'input': (input_grid := self.create_input(taskvars, {})),
                    'output': self.transform_input(input_grid, taskvars)
                }
                for _ in range(n)
            ]
        
        return taskvars, {
            'train': generate_examples(num_train),
            'test': generate_examples(1)
        }