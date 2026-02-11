from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taskdae9d2b5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {2*vars['n']}.",
            "The right half of the input grid contains some random number of cells of color {color('color_1')}.",
            "The left half of the input grid contains some random number of cells of color {color('color_2')}."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {vars['n']} x {vars['n']}.",
            "The output grid is constructed by taking the union of the colored cells from the left and right halves of the input grid.",
            "All cells in this union are recolored with {color('color_3')} in the output."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        
        # Create input grid of size n x 2n
        grid = np.zeros((n, 2 * n), dtype=int)
        
        # Fill left half with color_2 cells (random density between 0.1 and 0.6)
        left_half = grid[:, :n]
        density_left = random.uniform(0.1, 0.6)
        random_cell_coloring(left_half, color_2, density=density_left, background=0)
        
        # Fill right half with color_1 cells (random density between 0.1 and 0.6)
        right_half = grid[:, n:]
        density_right = random.uniform(0.1, 0.6)
        random_cell_coloring(right_half, color_1, density=density_right, background=0)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        color_3 = taskvars['color_3']
        
        # Create output grid of size n x n
        output = np.zeros((n, n), dtype=int)
        
        # Get left and right halves
        left_half = grid[:, :n]
        right_half = grid[:, n:]
        
        # Find union of colored cells (any non-zero cell from either half)
        union_mask = (left_half != 0) | (right_half != 0)
        
        # Color the union cells with color_3
        output[union_mask] = color_3
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        n = random.randint(5, 15)  # Keep reasonable size for union operation
        
        # Select three different colors
        available_colors = list(range(1, 10))  # Exclude 0 (background)
        selected_colors = random.sample(available_colors, 3)
        
        taskvars = {
            'n': n,
            'color_1': selected_colors[0],
            'color_2': selected_colors[1], 
            'color_3': selected_colors[2]
        }
        
        # Generate training and test examples
        num_train = random.randint(3, 6)
        
        def generate_valid_example():
            """Generate an example ensuring at least one cell can be filled"""
            max_attempts = 100
            for _ in range(max_attempts):
                input_grid = self.create_input(taskvars, {})
                output_grid = self.transform_input(input_grid, taskvars)
                
                # Check if output has at least one colored cell
                if np.any(output_grid != 0):
                    return {'input': input_grid, 'output': output_grid}
            
            # Fallback: create a minimal valid example
            input_grid = np.zeros((n, 2 * n), dtype=int)
            input_grid[0, 0] = taskvars['color_2']  # At least one cell in left half
            output_grid = self.transform_input(input_grid, taskvars)
            return {'input': input_grid, 'output': output_grid}
        
        train_examples = [generate_valid_example() for _ in range(num_train)]
        test_examples = [generate_valid_example()]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
