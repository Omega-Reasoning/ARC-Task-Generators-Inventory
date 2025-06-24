from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taskea786f4a(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each input grid is entirely filled with a single random color value, except for the center cell, which is marked as empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All cells located on the main diagonal and anti-diagonal are then replaced with empty (0) values."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        color = gridvars['color']
        
        # Create grid filled with the specified color
        grid = np.full((n, n), color, dtype=int)
        
        # Set center cell to empty (0)
        center = n // 2
        grid[center, center] = 0
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        output_grid = grid.copy()
        
        # Clear main diagonal (top-left to bottom-right)
        for i in range(n):
            output_grid[i, i] = 0
        
        # Clear anti-diagonal (top-right to bottom-left)
        for i in range(n):
            output_grid[i, n - 1 - i] = 0
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Random grid size between 5 and 30 (must be odd to have a clear center)
        n = random.choice([i for i in range(5, 31) if i % 2 == 1])
        
        # Random number of training examples (3-6)
        num_train = random.randint(3, 6)
        num_test = 1
        
        # Task variables
        taskvars = {'n': n}
        
        # Generate training examples with different colors
        train_examples = []
        used_colors = set()
        
        for _ in range(num_train):
            # Pick a random color (excluding 0 which is background)
            available_colors = [c for c in range(1, 10) if c not in used_colors]
            if not available_colors:  # If we've used all colors, reset
                used_colors.clear()
                available_colors = list(range(1, 10))
            
            color = random.choice(available_colors)
            used_colors.add(color)
            
            gridvars = {'color': color}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with a new color
        available_colors = [c for c in range(1, 10) if c not in used_colors]
        if not available_colors:
            available_colors = [random.randint(1, 9)]
        
        test_color = random.choice(available_colors)
        test_gridvars = {'color': test_color}
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
