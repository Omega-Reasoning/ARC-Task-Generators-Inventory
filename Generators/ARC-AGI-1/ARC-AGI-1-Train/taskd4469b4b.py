from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskd4469b4bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "A single random color is selected from {color('color_1')}, {color('color_2')}, or {color('color_3')}.",
            "The grid contains a random number of cells filled with the chosen color; all other cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is of size 3x3.",
            "The output grid is constructed by identifying the color of the cells in the input grid.",
            "Based on the identified color, a specific shape is filled with {color('color_fill')}, ensuring it fits completely within the bounds of the 3×3 output grid.",
            "If the identified color is {color('color_1')}, a cross shape is filled—spanning the middle row and middle column of the grid.",
            "If the identified color is {color('color_2')}, a horizontally flipped L shape is filled—formed by a vertical segment on the rightmost column and a horizontal segment across the bottom row.",
            "If the identified color is {color('color_3')}, a T shape is filled—made up of a horizontal line across the top row and a vertical line extending downward from its center."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        input_color = gridvars['input_color']
        
        # Create empty grid
        grid = np.zeros((n, n), dtype=int)
        
        # Randomly fill some cells with the chosen color
        density = random.uniform(0.1, 0.4)  # 10-40% of cells
        random_cell_coloring(grid, input_color, density=density)
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Create 3x3 output grid
        output = np.zeros((3, 3), dtype=int)
        
        # Find the color used in the input grid
        input_color = None
        for color in [taskvars['color_1'], taskvars['color_2'], taskvars['color_3']]:
            if np.any(grid == color):
                input_color = color
                break
        
        if input_color is None:
            return output  # Should not happen in valid inputs
        
        fill_color = taskvars['color_fill']
        
        # Fill the appropriate shape based on the detected color
        if input_color == taskvars['color_1']:
            # Cross shape: middle row and middle column
            output[1, :] = fill_color  # Middle row
            output[:, 1] = fill_color  # Middle column
            
        elif input_color == taskvars['color_2']:
            # Horizontally flipped L shape: rightmost column + bottom row
            output[:, 2] = fill_color  # Rightmost column
            output[2, :] = fill_color  # Bottom row
            
        elif input_color == taskvars['color_3']:
            # T shape: top row + vertical line from center
            output[0, :] = fill_color  # Top row
            output[:, 1] = fill_color  # Middle column (vertical line from center)
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'n': random.randint(5, 30),
            'color_1': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'color_2': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'color_3': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'color_fill': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        # Ensure all colors are different
        colors = [taskvars['color_1'], taskvars['color_2'], taskvars['color_3'], taskvars['color_fill']]
        while len(set(colors)) != 4:
            taskvars['color_1'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            taskvars['color_2'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            taskvars['color_3'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            taskvars['color_fill'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
            colors = [taskvars['color_1'], taskvars['color_2'], taskvars['color_3'], taskvars['color_fill']]
        
        # Create training examples ensuring all three colors appear
        num_train = random.randint(3, 6)
        train_examples = []
        
        # Ensure we have at least one example of each color
        required_colors = [taskvars['color_1'], taskvars['color_2'], taskvars['color_3']]
        colors_used = []
        
        for i in range(num_train):
            if i < 3:
                # For first 3 examples, use required colors in order
                input_color = required_colors[i]
            else:
                # For additional examples, choose randomly
                input_color = random.choice(required_colors)
            
            colors_used.append(input_color)
            gridvars = {'input_color': input_color}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_input_color = random.choice(required_colors)
        test_gridvars = {'input_color': test_input_color}
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

