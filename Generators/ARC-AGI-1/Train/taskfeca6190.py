from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from Framework.input_library import random_cell_coloring

class Taskfeca6190Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size 1 x {vars['n']}.",
            "In each input grid, a random number of cells is colored, each with a random color.",
            "An input grid with one colored cell is valid provided that the cell is not in the last position.",
            "The random number of colored cells, their locations, and their colors vary in each input grid to ensure diversity"
        ]
        
        transformation_reasoning_chain = [
            "The number of distinct colors used in the input grid is first determined and denoted as m.",
            "The output grid is of size ({vars['n']}*m) x ({vars['n']}*m).",
            "The output grid is constructed by initializing a blank grid of the specified size.",
            "The input grid is copied into the bottom-left corner of the output grid.",
            "Each colored cell in the output grid is identified.",
            "For every such cell, the entire anti-diagonal passing through it is colored with the same color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        
        # Create a 1×n grid
        grid = np.zeros((1, n), dtype=int)
        
        # Get available colors (1-9, excluding 0 which is background)
        available_colors = list(range(1, 10))
        
        # Determine number of cells to color (at least 1, at most n-1 to avoid last position constraint)
        max_colored = min(n - 1, len(available_colors))
        num_colored = random.randint(1, max_colored)
        
        # Select random positions, ensuring we don't pick the last position if we only have 1 colored cell
        available_positions = list(range(n))
        if num_colored == 1:
            available_positions = available_positions[:-1]  # Remove last position
        
        positions = random.sample(available_positions, num_colored)
        
        # Select random colors for each position
        colors = random.sample(available_colors, num_colored)
        
        # Color the cells
        for pos, color in zip(positions, colors):
            grid[0, pos] = color
            
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        
        # Find distinct colors (excluding background 0)
        distinct_colors = set(grid[grid != 0])
        m = len(distinct_colors)
        
        # Create output grid of size (m×n) × (m×n)
        output_size = m * n
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Copy input grid to bottom-left corner
        # Bottom-left means starting at row (output_size - 1) and column 0
        start_row = output_size - 1
        for col in range(n):
            output_grid[start_row, col] = grid[0, col]
        
        # For each colored cell in the output grid, extend along anti-diagonal
        for row in range(output_size):
            for col in range(output_size):
                if output_grid[row, col] != 0:
                    color = output_grid[row, col]
                    # Extend along anti-diagonal (where row + col = constant)
                    diagonal_sum = row + col
                    
                    # Fill all cells where r + c = diagonal_sum
                    for r in range(output_size):
                        for c in range(output_size):
                            if r + c == diagonal_sum:
                                output_grid[r, c] = color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random grid width between 5 and 30
        # But since output size will be m*n × m*n, we need to be careful about the maximum
        # With worst case m=9 (if all cells have different colors), we need n such that 9*n ≤ 30
        # So n ≤ 3 for safety, but that's too restrictive
        # Let's use a more reasonable range
        n = random.randint(3, 6)  # This ensures output won't exceed 30×30 even with many colors
        
        taskvars = {'n': n}
        
        # Generate 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        def generate_examples(count):
            examples = []
            for _ in range(count):
                input_grid = self.create_input(taskvars, {})
                output_grid = self.transform_input(input_grid, taskvars)
                examples.append({
                    'input': input_grid,
                    'output': output_grid
                })
            return examples
        
        train_examples = generate_examples(num_train)
        test_examples = generate_examples(1)
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
