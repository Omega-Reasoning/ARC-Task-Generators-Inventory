from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taske179c5f4Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x n, where n is an integer number greater than 1.",
            "In each input grid, the bottom-left cell is the only non-empty cell and is colored {color('color_1')}.",
            "All other cells remain empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The bottom-left colored cell is identified and is set as the initial cell.",
            "The color of the initial cell is {color('color_1')}.",
            "Starting from the initial cell, a zigzag pattern is filled upward with {color('color_1')} as follows: cells are colored moving diagonally up and right (row decreases by 1, column increases by 1) until the right edge of the grid is reached; the last cell colored in this step becomes the initial cell for the next step, where cells are colored moving diagonally up and left (row decreases by 1, column decreases by 1) until the left edge of the grid is reached; the last cell colored in this step then becomes the initial cell for the next iteration of coloring diagonally up and right.",
            "These steps are repeated alternately until the zigzag reaches the top row.",
            "After the zigzag pattern is completed, all remaining empty cells are filled with {color('color_2')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']  # Fixed number of rows from taskvars
        cols = gridvars.get('cols', random.randint(2, min(rows-1, 15)))  # Variable columns
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Place colored cell at bottom-left
        grid[rows-1, 0] = taskvars['color_1']
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        rows, cols = grid.shape
        
        # Start from bottom-left corner
        current_row = rows - 1
        current_col = 0
        
        # Track zigzag direction: True = moving right (anti-diagonal), False = moving left (diagonal)
        moving_right = True
        
        while current_row >= 0:
            if moving_right:
                # Step 1: Move diagonally up-right until right edge
                while current_row >= 0 and current_col < cols:
                    output[current_row, current_col] = taskvars['color_1']
                    current_row -= 1
                    current_col += 1
                
                # Continue from the last position (which went out of bounds)
                # Go back to the last valid position
                current_row += 1
                current_col -= 1
                
                # Now move diagonally up-left from this position
                current_row -= 1
                current_col -= 1
                moving_right = False
                    
            else:
                # Step 2: Move diagonally up-left until left edge
                while current_row >= 0 and current_col >= 0:
                    output[current_row, current_col] = taskvars['color_1']
                    current_row -= 1
                    current_col -= 1
                
                # Continue from the last position (which went out of bounds)
                # Go back to the last valid position
                current_row += 1
                current_col += 1
                
                # Now move diagonally up-right from this position
                current_row -= 1
                current_col += 1
                moving_right = True
        
        # Fill remaining empty cells with color_2
        output[output == 0] = taskvars['color_2']
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables - rows is FIXED for all grids
        taskvars = {
            'rows': random.randint(8, 30),  # Fixed number of rows for ALL grids
            'color_1': random.randint(1, 9),
            'color_2': random.randint(1, 9)
        }
        
        # Ensure colors are different
        while taskvars['color_2'] == taskvars['color_1']:
            taskvars['color_2'] = random.randint(1, 9)
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Vary ONLY the number of columns for each example
            gridvars = {
                'cols': random.randint(2, min(taskvars['rows']//2, 15))  # Ensure rows > cols
            }
            
            # Ensure grid doesn't exceed ARC limits
            gridvars['cols'] = min(gridvars['cols'], 30)
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with different number of columns
        test_gridvars = {
            'cols': random.randint(3, min(taskvars['rows']-1, 12))
        }
        
        # Ensure grid doesn't exceed ARC limits
        test_gridvars['cols'] = min(test_gridvars['cols'], 30)
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
