from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskd406998bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size 3 x n.",
            "Each grid contains a random number of colored cells of color {color('object_color')},with the constraint that no two colored cells are horizontally or vertically adjacent (4-connectivity).",
            "Each column exactly contains one colored cell.",
            "All the remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All colored cells are identified and they are of color {color('object_color')}.",
            "Starting from the rightmost column and moving leftward in steps of two, each colored cell in the respective column is overwritten with {color('fill_color')} until the left boundary is reached."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        width = gridvars['width']
        object_color = taskvars['object_color']
        
        def generate_valid_grid():
            # Create empty 3×width grid
            grid = np.zeros((3, width), dtype=int)
            
            for col in range(width):
                # Get available positions based on adjacency constraints
                available_rows = []
                
                for row in range(3):
                    valid = True
                    
                    # Only check horizontal adjacency (4-connectivity, not 8-connectivity)
                    if col > 0:
                        # Check if same row as previous column (horizontally adjacent)
                        for prev_row in range(3):
                            if grid[prev_row, col-1] == object_color and prev_row == row:
                                valid = False
                                break
                    
                    if valid:
                        available_rows.append(row)
                
                # If no valid rows available, this configuration is impossible
                if not available_rows:
                    return None
                
                # Randomly choose from available rows
                chosen_row = random.choice(available_rows)
                grid[chosen_row, col] = object_color
            
            return grid
        
        def is_valid_grid(grid):
            if grid is None:
                return False
            
            # Check that no two colored cells are adjacent (4-connectivity only)
            for r in range(3):
                for c in range(width):
                    if grid[r, c] == object_color:
                        # Check only horizontal and vertical neighbors (not diagonal)
                        neighbors = [
                            (r-1, c),  # up
                            (r+1, c),  # down
                            (r, c-1),  # left
                            (r, c+1)   # right
                        ]
                        
                        for nr, nc in neighbors:
                            if 0 <= nr < 3 and 0 <= nc < width:
                                if grid[nr, nc] == object_color:
                                    return False
            return True
        
        return retry(generate_valid_grid, is_valid_grid, max_attempts=100)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        object_color = taskvars['object_color']
        fill_color = taskvars['fill_color']
        
        height, width = grid.shape
        
        # Starting from the rightmost column and moving leftward in steps of two
        for col in range(width - 1, -1, -2):  # Start from rightmost, step by -2
            # Find the colored cell in this column and overwrite it
            for row in range(height):
                if output_grid[row, col] == object_color:
                    output_grid[row, col] = fill_color
                    break
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate different colors
        all_colors = list(range(1, 10))  # Colors 1-9 (excluding 0 which is background)
        random.shuffle(all_colors)
        object_color = all_colors[0]
        fill_color = all_colors[1]  # Ensure different colors
        
        # Random number of training examples
        num_train = random.randint(3, 6)
        
        # Generate task variables
        taskvars = {
            'object_color': object_color,
            'fill_color': fill_color
        }
        
        # Generate training examples with varying widths
        train_examples = []
        for _ in range(num_train):
            width = random.randint(5, 15)  # Can use larger range now
            gridvars = {'width': width}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_width = random.randint(6, 20)
        test_gridvars = {'width': test_width}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
