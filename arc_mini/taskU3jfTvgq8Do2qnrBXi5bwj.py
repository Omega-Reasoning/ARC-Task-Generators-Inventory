from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskU3jfTvgq8Do2qnrBXi5bwjGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size 3 × {vars['cols']}.",
            "Each input grid contains exactly 4 colored cells, which appear in pairs.",
            "Two colored cells are placed in the first row and two in the last row.",
            "In both the first and last rows, the cells appear either at the beginning or at the end of the row.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by reducing the grid to 2 rows, with the number of columns halved.",
            "Each pair of adjacent cells in the input is merged into a single cell in the output.", 
            "The new cell keeps the same color as the original pair.",
            "The relative positions of all cells remain consistent with the input."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        cols = taskvars['cols']
        grid = np.zeros((3, cols), dtype=int)
        
        # Get the two different colors for this grid
        color1 = gridvars['color1'] 
        color2 = gridvars['color2']
        
        # Place pairs in first row - either at beginning or end
        first_row_position = gridvars['first_row_position']  # 'start' or 'end'
        if first_row_position == 'start':
            grid[0, 0] = color1
            grid[0, 1] = color1
        else:  # 'end'
            grid[0, cols-2] = color1
            grid[0, cols-1] = color1
            
        # Place pairs in last row - either at beginning or end  
        last_row_position = gridvars['last_row_position']  # 'start' or 'end'
        if last_row_position == 'start':
            grid[2, 0] = color2
            grid[2, 1] = color2
        else:  # 'end'
            grid[2, cols-2] = color2
            grid[2, cols-1] = color2
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = grid.shape
        output_cols = cols // 2
        output = np.zeros((2, output_cols), dtype=int)
        
        # Merge pairs from first row (row 0) into output row 0
        for c in range(0, cols, 2):
            output_col = c // 2
            # Take the color from either cell in the pair (they should be the same)
            if grid[0, c] != 0:
                output[0, output_col] = grid[0, c]
            elif c + 1 < cols and grid[0, c + 1] != 0:
                output[0, output_col] = grid[0, c + 1]
        
        # Merge pairs from last row (row 2) into output row 1  
        for c in range(0, cols, 2):
            output_col = c // 2
            # Take the color from either cell in the pair (they should be the same)
            if grid[2, c] != 0:
                output[1, output_col] = grid[2, c]
            elif c + 1 < cols and grid[2, c + 1] != 0:
                output[1, output_col] = grid[2, c + 1]
                
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random number of columns (even, between 6 and 14)
        cols = random.choice([6, 8, 10, 12, 14])
        
        # Available colors (excluding 0 which is background)
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        taskvars = {
            'cols': cols
        }
        
        # Generate 3-5 training examples
        num_train = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train):
            # Choose two different colors for this example
            colors = random.sample(available_colors, 2)
            
            gridvars = {
                'color1': colors[0],
                'color2': colors[1], 
                'first_row_position': random.choice(['start', 'end']),
                'last_row_position': random.choice(['start', 'end'])
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_colors = random.sample(available_colors, 2)
        test_gridvars = {
            'color1': test_colors[0],
            'color2': test_colors[1],
            'first_row_position': random.choice(['start', 'end']),
            'last_row_position': random.choice(['start', 'end'])
        }
        
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

