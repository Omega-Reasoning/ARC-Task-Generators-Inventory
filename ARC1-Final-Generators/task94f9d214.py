from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects, GridObjects

class Task94f9d214Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid with dimension {vars['rows']} X {2 * vars['rows']}.",
            "The upper half of the input grid of size {vars['rows']} x {vars['rows']} has a 4-way connected object of color {color('upper_color')}.",
            "The lower half of the input grid has another 4-way connected object of color {color('lower_color')}."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same width but half the height of the input grid.",
            "The top and bottom halves of the input grid are combined using an NOR operation,i.e. Colored cells are considered 1 and empty cells are considered 0.",
            "The output cells which given value 1 from the above operation is colored {color('xor')},",
            "The remaining cells are empty(0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        rows = 2 * random.randint(2, 7)  # Ensure even rows between 10-14
        
        # Choose three distinct colors
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.shuffle(available_colors)
        upper_color, lower_color, xor = available_colors[:3]
        
        taskvars = {
            'rows': rows,
            'upper_color': upper_color,
            'lower_color': lower_color,
            'xor': xor
        }
        
        # Generate 3-4 training examples and 1 test example
        num_train = random.randint(3, 4)
        
        # Create training examples
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Extract task variables
        rows = taskvars['rows']
        upper_color = taskvars['upper_color']
        lower_color = taskvars['lower_color']
        
        # Initialize the input grid
        input_grid = np.zeros((2 * rows, rows), dtype=int)
        
        # Create the upper half object with 4-way connectivity
        upper_half = create_object(
            height=rows,
            width=rows,
            color_palette=upper_color,
            contiguity=Contiguity.FOUR,
            background=0
        )
        
        # Create the lower half object with 4-way connectivity
        lower_half = create_object(
            height=rows,
            width=rows,
            color_palette=lower_color,
            contiguity=Contiguity.FOUR,
            background=0
        )
        
        # Place objects in the grid
        input_grid[:rows, :] = upper_half
        input_grid[rows:, :] = lower_half
        
        return input_grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Extract task variables
        rows = taskvars['rows']
        xor_color = taskvars['xor']
        
        # Split the grid into upper and lower halves
        upper_half = grid[:rows, :]
        lower_half = grid[rows:, :]
        
        # Create output grid with the same width but half the height
        output_grid = np.zeros((rows, grid.shape[1]), dtype=int)
        
        # Apply NOR operation correctly
        # For each position in the output grid:
        for r in range(rows):
            for c in range(grid.shape[1]):
                # In NOR, result is 1 only if both inputs are 0
                if upper_half[r, c] == 0 and lower_half[r, c] == 0:
                    output_grid[r, c] = xor_color
                else:
                    output_grid[r, c] = 0
        
        return output_grid