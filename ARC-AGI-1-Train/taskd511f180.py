from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taskd511f180Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each cell is colored with one of multiple distinct colors, where the total number of colors is an integer greater than 2.",
            "Among these colors, one is {color('color_1')} and another is {color('color_2')}."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "The output grid is constructed by copying the input grid.",
            "All cells colored {color('color_1')} are changed to {color('color_2')}, and all cells colored {color('color_2')} are changed to {color('color_1')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {}
        
        # Choose two distinct colors to swap (excluding background color 0)
        available_colors = list(range(1, 10))
        swap_colors = random.sample(available_colors, 2)
        taskvars['color_1'] = swap_colors[0]
        taskvars['color_2'] = swap_colors[1]
        
        # Generate 3-6 training examples
        num_train = random.randint(3, 6)
        
        train_data = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Generate one test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Random grid size between 5 and 30
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        
        # Determine number of colors (greater than 2, at most 9)
        num_colors = random.randint(3, 9)
        
        # Always include the two swap colors
        colors_to_use = [taskvars['color_1'], taskvars['color_2']]
        
        # Add additional colors if needed
        available_colors = [c for c in range(1, 10) if c not in colors_to_use]
        additional_colors_needed = num_colors - 2
        if additional_colors_needed > 0:
            colors_to_use.extend(random.sample(available_colors, 
                                             min(additional_colors_needed, len(available_colors))))
        
        # Create grid filled with random colors (no background/0)
        # Each cell gets a random color from colors_to_use
        grid = np.random.choice(colors_to_use, size=(height, width))
        
        # Ensure both swap colors are present in the grid
        if taskvars['color_1'] not in grid:
            # Place at least one cell of color_1
            r, c = random.randint(0, height-1), random.randint(0, width-1)
            grid[r, c] = taskvars['color_1']
        
        if taskvars['color_2'] not in grid:
            # Place at least one cell of color_2
            r, c = random.randint(0, height-1), random.randint(0, width-1)
            grid[r, c] = taskvars['color_2']
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Copy the input grid
        output_grid = grid.copy()
        
        # Swap the two colors
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        
        # Create masks for each color
        mask_1 = (grid == color_1)
        mask_2 = (grid == color_2)
        
        # Perform the swap
        output_grid[mask_1] = color_2
        output_grid[mask_2] = color_1
        
        return output_grid