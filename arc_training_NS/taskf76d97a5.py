from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskf76d97a5(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares of different sizes.",
            "In each input grid a random number of the cells in the grid are of color {color('base_color')}.",
            "The remaining cells are of a random color.",
            "The remaining cells are filled with a randomly chosen color, which differs from one input grid to another."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is the same size as the input grid.",
            "The output grid is constructed by first identifying all the cells and their colors in the input grid.",
            "In the output grid, cells corresponding to those colored {color('base_color')} in the input grid are replaced with the other color present in that input grid.",
            "All other cells in the output grid are empty(0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        base_color = random.randint(1, 9)
        taskvars = {
            'base_color': base_color
        }
        
        # Generate 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        train_examples = []
        test_examples = []
        
        # Create training examples
        for _ in range(num_train):
            # Generate grid-specific variables
            grid_size = random.randint(5, 30)
            # Choose random color different from base_color
            available_colors = [c for c in range(1, 10) if c != base_color]
            random_color = random.choice(available_colors)
            
            gridvars = {
                'grid_size': grid_size,
                'random_color': random_color
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_grid_size = random.randint(5, 30)
        available_colors = [c for c in range(1, 10) if c != base_color]
        test_random_color = random.choice(available_colors)
        
        test_gridvars = {
            'grid_size': test_grid_size,
            'random_color': test_random_color
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples.append({
            'input': test_input,
            'output': test_output
        })
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = gridvars['grid_size']
        base_color = taskvars['base_color']
        random_color = gridvars['random_color']
        
        # Create a grid filled with the random color as background
        grid = np.full((grid_size, grid_size), random_color, dtype=int)
        
        # Use input_library to randomly place base_color cells
        # Use a random density between 0.2 and 0.8 to ensure both colors are present
        base_density = random.uniform(0.2, 0.8)
        
        # Apply random cell coloring with base_color, treating random_color as background
        random_cell_coloring(
            grid=grid,
            color_palette=base_color,
            density=base_density,
            background=random_color,
            overwrite=True  # Allow overwriting the random_color cells
        )
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        base_color = taskvars['base_color']
        
        # Create output grid filled with zeros (empty)
        output = np.zeros_like(grid)
        
        # Find the random color (any color that's not base_color and not 0)
        unique_colors = np.unique(grid)
        random_color = None
        for color in unique_colors:
            if color != base_color and color != 0:
                random_color = color
                break
        
        # Replace base_color cells with the random color
        if random_color is not None:
            output[grid == base_color] = random_color
        
        return output

