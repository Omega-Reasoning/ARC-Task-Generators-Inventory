from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects, GridObject
from Framework.input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taska68b268eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Each input grid is of size {vars['n']} × {vars['n']}, where {vars['n']} is an odd number.",
            "The grid has a central row (at index {vars['n']} // 2) and a central column (at index {vars['n']} // 2), both consistently filled with {color('middle_color')}, dividing the grid into four distinct regions.",
            "The top-left region contains a random number of cells colored with {color('color_1')}.",
            "The top-right region contains a random number of cells colored with {color('color_2')}.",
            "The bottom-left region contains a random number of cells colored with {color('color_3')}.",
            "The bottom-right region contains a random number of cells colored with {color('color_4')}."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is initialized by copying the top-left region of the input grid.",
            "Next, all cells in the top-right region of the input grid are considered. For each such cell, if the corresponding position in the output grid is empty, it is filled with the same color.",
            "Then, all cells in the bottom-left region of the input grid are considered. For each such cell, if the corresponding position in the output grid is empty, it is filled with the same color.",
            "Finally, all cells in the bottom-right region of the input grid are considered. For each such cell, if the corresponding position in the output grid is empty, it is filled with the same color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        middle_color = taskvars['middle_color']
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2'] 
        color_3 = taskvars['color_3']
        color_4 = taskvars['color_4']
        
        # Initialize grid with background (0)
        grid = np.zeros((n, n), dtype=int)
        
        # Fill central row and column with middle color
        center = n // 2
        grid[center, :] = middle_color  # Central row
        grid[:, center] = middle_color  # Central column
        
        # Define regions (excluding central row/column)
        top_left = grid[:center, :center]
        top_right = grid[:center, center+1:]
        bottom_left = grid[center+1:, :center]
        bottom_right = grid[center+1:, center+1:]
        
        # Randomly color cells in each region
        region_size = center * center
        if region_size > 0:
            # Use different densities to ensure variety
            densities = [0.1, 0.2, 0.3, 0.4, 0.5]
            
            random_cell_coloring(top_left, color_1, 
                                density=random.choice(densities))
            random_cell_coloring(top_right, color_2, 
                                density=random.choice(densities))
            random_cell_coloring(bottom_left, color_3, 
                                density=random.choice(densities))
            random_cell_coloring(bottom_right, color_4, 
                                density=random.choice(densities))
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        center = n // 2
        
        # Initialize output grid as a copy of the top-left region
        output = np.zeros((center, center), dtype=int)
        output[:, :] = grid[:center, :center]
        
        # Overlay top-right region
        top_right = grid[:center, center+1:]
        for r in range(center):
            for c in range(top_right.shape[1]):
                if c < output.shape[1] and top_right[r, c] != 0:
                    if output[r, c] == 0:  # Only fill if empty
                        output[r, c] = top_right[r, c]
        
        # Overlay bottom-left region
        bottom_left = grid[center+1:, :center]
        for r in range(bottom_left.shape[0]):
            for c in range(center):
                if r < output.shape[0] and bottom_left[r, c] != 0:
                    if output[r, c] == 0:  # Only fill if empty
                        output[r, c] = bottom_left[r, c]
        
        # Overlay bottom-right region
        bottom_right = grid[center+1:, center+1:]
        for r in range(bottom_right.shape[0]):
            for c in range(bottom_right.shape[1]):
                if r < output.shape[0] and c < output.shape[1] and bottom_right[r, c] != 0:
                    if output[r, c] == 0:  # Only fill if empty
                        output[r, c] = bottom_right[r, c]
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {}
        
        # Grid size - must be odd and between 5 and 30
        odd_sizes = [i for i in range(5, 31) if i % 2 == 1]
        taskvars['n'] = random.choice(odd_sizes)
        
        # Choose 5 distinct colors (including middle color)
        available_colors = list(range(1, 10))  # Exclude 0 (background)
        chosen_colors = random.sample(available_colors, 5)
        
        taskvars['middle_color'] = chosen_colors[0]
        taskvars['color_1'] = chosen_colors[1]
        taskvars['color_2'] = chosen_colors[2]
        taskvars['color_3'] = chosen_colors[3]
        taskvars['color_4'] = chosen_colors[4]
        
        # Create train and test examples
        num_train = random.randint(3, 6)
        num_test = 1
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        test_examples = []
        for _ in range(num_test):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

