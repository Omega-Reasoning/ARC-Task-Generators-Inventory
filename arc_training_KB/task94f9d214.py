from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects, GridObjects

class ARCTask94f9d214Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has {2 * vars['rows']} rows and {vars['rows']} columns.",
            "The upper half of the input grid is of size {vars['rows']} x {vars['rows']} and has a 4-way connected object of {color('upper_color')} color.",
            "The lower half of the input grid has another 4-way connected object of {color('lower_color')} color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same number of columns as the input grid but only half its number of rows.",
            "Each output cell is obtained by applying a NOR operation to the corresponding cells in the top and bottom halves of the input grid (treat colored cells as 1 and empty cells as 0).",
            "Cells that evaluate to 1 in this NOR operation are colored {color('xor')}.",
            "All remaining cells are left empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        rows = 2 * random.randint(2, 7)  # even rows between 4 and 14
        
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
        
        # Generate 3–4 training examples and 1 test example
        num_train = random.randint(3, 4)
        
        train_examples: List[GridPair] = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples: List[GridPair] = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        upper_color = taskvars['upper_color']
        lower_color = taskvars['lower_color']

        # We keep regenerating until we have at least 2 positions
        # where upper_half == 0 and lower_half == 0 at the same (r, c).
        while True:
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
            
            # Count positions where both halves are empty at the same coordinates
            shared_empty_mask = (upper_half == 0) & (lower_half == 0)
            if np.sum(shared_empty_mask) >= 2:
                # Condition satisfied: place them into the full grid and return
                input_grid[:rows, :] = upper_half
                input_grid[rows:, :] = lower_half
                return input_grid
            # Otherwise loop and regenerate upper_half and lower_half

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        xor_color = taskvars['xor']
        
        # Split the grid into upper and lower halves
        upper_half = grid[:rows, :]
        lower_half = grid[rows:, :]
        
        # Create output grid with the same width but half the height
        output_grid = np.zeros((rows, grid.shape[1]), dtype=int)
        
        # Apply NOR: 1 only if both inputs are 0
        for r in range(rows):
            for c in range(grid.shape[1]):
                if upper_half[r, c] == 0 and lower_half[r, c] == 0:
                    output_grid[r, c] = xor_color
                else:
                    output_grid[r, c] = 0
        
        return output_grid
