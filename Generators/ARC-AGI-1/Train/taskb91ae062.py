from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskb91ae062Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} × {vars['n']}.",
            "A random number of cells are colored.",
            "The coloring uses between 2 and {vars['max_colors']} distinct colors, where the maximum is constrained by floor(30 / {vars['n']}) to ensure the output grid does not exceed 30×30.",
            "All remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "Let m be the number of distinct colors present in the input grid.",
            "The output grid is formed by expanding each input cell into an m × m block: If the input cell is colored, the entire block is filled with that color. If the input cell is empty (0), the entire block remains empty."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        max_colors = taskvars['max_colors']
        
        def generate_grid():
            # Create empty grid
            grid = np.zeros((n, n), dtype=int)
            
            # Choose number of colors to use (2 to max_colors, ensuring more than one)
            num_colors = random.randint(2, max_colors)
            
            # Choose which colors to use (avoiding 0 which is background)
            available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            colors = random.sample(available_colors, num_colors)
            
            # Randomly color cells with density between 0.2 and 0.7
            # Higher minimum density to ensure we get multiple colors
            density = random.uniform(0.2, 0.7)
            random_cell_coloring(grid, colors, density=density, background=0)
            
            return grid
        
        def has_multiple_colors(grid):
            unique_colors = np.unique(grid)
            # Count non-background colors
            non_bg_colors = len(unique_colors) - 1 if 0 in unique_colors else len(unique_colors)
            return non_bg_colors >= 2
        
        # Use retry to ensure we get a grid with multiple colors
        return retry(generate_grid, has_multiple_colors)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        
        # Find number of distinct colors (m)
        unique_colors = np.unique(grid)
        # Remove background color (0) from count
        m = len(unique_colors) - 1 if 0 in unique_colors else len(unique_colors)
        
        # Handle edge case where grid is all empty (shouldn't happen with our constraints)
        if m == 0:
            return np.zeros((n, n), dtype=int)
        
        # Create output grid of size (n*m) × (n*m)
        output = np.zeros((n * m, n * m), dtype=int)
        
        # For each cell in input grid, expand to m×m block in output
        for i in range(n):
            for j in range(n):
                color = grid[i, j]
                # Calculate the m×m block position in output grid
                start_row = i * m
                end_row = start_row + m
                start_col = j * m
                end_col = start_col + m
                
                # Fill the entire m×m block with the color
                output[start_row:end_row, start_col:end_col] = color
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Randomly choose grid size (3 to 10)
        n = random.randint(3, 10)
        
        # Calculate maximum number of colors based on grid size
        # Since output is n*m and we want n*m <= 30, then m <= 30/n
        # We use floor division to get the maximum integer value
        max_colors = min(9, 30 // n)  # Cap at 9 (we have colors 1-9 available)
        
        # Ensure we have at least 2 colors possible
        max_colors = max(2, max_colors)
        
        taskvars = {
            'n': n,
            'max_colors': max_colors
        }
        
        # Generate 3-6 training examples
        num_train = random.randint(3, 6)
        
        train_examples = []
        test_examples = []
        
        # Create training examples
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create one test example
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data