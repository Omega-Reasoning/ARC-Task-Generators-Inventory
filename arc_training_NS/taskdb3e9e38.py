from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskdb3e9e38(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "It contains a single vertical bar hanging from the top edge of the grid, composed of cells colored {color('color_1')}.",
            "The height of the vertical bar varies across different input grids.",
            "The height of the vertical bar is at least 1 and at most one less than the total number of rows in the grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The vertical bar of color {color('color_1')} is identified, and its column index i and height j (the number of consecutive colored cells from the top) are recorded.",
            "For each positive offset k, if at least one of the columns i + k or i - k lies within the grid boundaries, the top j - k cells in those columns are colored with {color('color_2')} when k is odd, and with {color('color_1')} when k is even, provided that j - k is greater than zero."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Get grid dimensions from gridvars or use random defaults
        if 'grid_height' in gridvars and 'grid_width' in gridvars:
            height = gridvars['grid_height']
            width = gridvars['grid_width']
        else:
            height = random.randint(5, 15)
            width = random.randint(5, 15)
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Place vertical bar
        if 'bar_column' in gridvars and 'bar_height' in gridvars:
            col = gridvars['bar_column']
            bar_height = gridvars['bar_height']
        else:
            col = random.randint(0, width - 1)
            bar_height = random.randint(1, height - 1)
        
        # Fill the vertical bar from top
        for row in range(bar_height):
            grid[row, col] = taskvars['color_1']
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        height, width = grid.shape
        
        # Find the vertical bar
        bar_column = None
        bar_height = 0
        
        # Look for the column with the vertical bar
        for col in range(width):
            if grid[0, col] == taskvars['color_1']:
                # Count consecutive cells from top
                current_height = 0
                for row in range(height):
                    if grid[row, col] == taskvars['color_1']:
                        current_height += 1
                    else:
                        break
                
                if current_height > bar_height:
                    bar_column = col
                    bar_height = current_height
        
        if bar_column is None:
            return output_grid
        
        # Apply the transformation rule
        for k in range(1, max(width, height)):  # Positive offsets only
            # Check if either i+k or i-k is within bounds
            left_col = bar_column - k
            right_col = bar_column + k
            
            if not (0 <= left_col < width or 0 <= right_col < width):
                continue
            
            # Calculate height for this offset
            fill_height = bar_height - k
            if fill_height <= 0:
                continue
            
            # Determine color based on offset parity
            fill_color = taskvars['color_2'] if k % 2 == 1 else taskvars['color_1']
            
            # Fill left column if in bounds
            if 0 <= left_col < width:
                for row in range(fill_height):
                    output_grid[row, left_col] = fill_color
            
            # Fill right column if in bounds
            if 0 <= right_col < width:
                for row in range(fill_height):
                    output_grid[row, right_col] = fill_color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        taskvars = {
            'color_1': available_colors[0],
            'color_2': available_colors[1]
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Vary grid dimensions and bar properties
            grid_height = random.randint(5, 15)
            grid_width = random.randint(5, 15)
            bar_column = random.randint(0, grid_width - 1)
            bar_height = random.randint(1, grid_height - 1)
            
            gridvars = {
                'grid_height': grid_height,
                'grid_width': grid_width,
                'bar_column': bar_column,
                'bar_height': bar_height
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            # Ensure the transformation actually produces a different result
            if not np.array_equal(input_grid, output_grid):
                train_examples.append({
                    'input': input_grid,
                    'output': output_grid
                })
        
        # Ensure we have at least 3 valid training examples
        while len(train_examples) < 3:
            grid_height = random.randint(6, 12)
            grid_width = random.randint(6, 12)
            bar_column = random.randint(1, grid_width - 2)  # Ensure space for radiation
            bar_height = random.randint(2, grid_height - 1)  # Ensure some radiation possible
            
            gridvars = {
                'grid_height': grid_height,
                'grid_width': grid_width,
                'bar_column': bar_column,
                'bar_height': bar_height
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            if not np.array_equal(input_grid, output_grid):
                train_examples.append({
                    'input': input_grid,
                    'output': output_grid
                })
        
        # Generate test example
        grid_height = random.randint(6, 12)
        grid_width = random.randint(6, 12)
        bar_column = random.randint(1, grid_width - 2)
        bar_height = random.randint(2, grid_height - 1)
        
        test_gridvars = {
            'grid_height': grid_height,
            'grid_width': grid_width,
            'bar_column': bar_column,
            'bar_height': bar_height
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
