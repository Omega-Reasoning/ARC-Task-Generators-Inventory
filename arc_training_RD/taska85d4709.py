from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import retry
import numpy as np
import random

class Taska85d4709Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size 3x3.",
            "Each input grid contains exactly one shaded cell per row, resulting in a total of three shaded cells.",
            "All the shaded cells are of {color('input_color')} color but are positioned such that every row includes precisely one shaded cell."
        ]
        
        transformation_reasoning_chain = [
            "The output grid retains the same dimensions as the input grid.",
            "The transformation is based on the position of the shaded cells in each row.",
            "For each row, the column index of the shaded cell determines the color that should be applied to the entire row in the output grid.",
            "Suppose, in a row, if the shaded cell is appeared on the first column then then that entire row gets the color {color('color1')}, similarly, in a row, if the shaded cell is present on the second column then that entire row gets the color {color('color2')}, lastly, in a row, if the shaded cell is present on the last column then then that entire row gets the color {color('color3')}.",
            "This mapping is performed independently for each row by examining the position (i.e., column index) of the shaded cell within that row. The specific column in which the shaded cell appears determines the color assigned to the entire row. This means that the color transformation does not depend on the order of rows or the pattern in other rows, it strictly relies on the column location of the shaded cell in that particular row alone. Each row is processed in isolation, using only its own shaded cells position to decide the final row color.",
            "The core logic is to detect the column index of the shaded cell in each row and use that index to determine the color that fills the entire row in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict = None) -> np.ndarray:
        """Create a 3x3 input grid with exactly one shaded cell per row and no column having more than one shaded cell."""
        input_color = taskvars['input_color']
        
        def generate_valid_grid():
            # Create empty 3x3 grid
            grid = np.zeros((3, 3), dtype=int)
            
            # Generate a random permutation of columns [0, 1, 2] for rows [0, 1, 2]
            # This ensures exactly one shaded cell per row and at most one per column
            column_positions = random.sample([0, 1, 2], 3)
            
            # Place shaded cells
            for row in range(3):
                col = column_positions[row]
                grid[row, col] = input_color
                
            return grid
        
        # Generate grid (should always be valid with our approach, but using retry for consistency)
        return retry(
            generate_valid_grid,
            lambda grid: (
                # Check exactly one shaded cell per row
                all(np.sum(grid[row] != 0) == 1 for row in range(3)) and
                # Check at most one shaded cell per column
                all(np.sum(grid[:, col] != 0) <= 1 for col in range(3)) and
                # Check total of exactly 3 shaded cells
                np.sum(grid != 0) == 3
            )
        )
    
    def transform_input(self, input_grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input grid by coloring entire rows based on column position of shaded cells."""
        input_color = taskvars['input_color']
        color1 = taskvars['color1']  # Column 0 -> color1
        color2 = taskvars['color2']  # Column 1 -> color2  
        color3 = taskvars['color3']  # Column 2 -> color3
        
        # Create output grid
        output_grid = np.zeros_like(input_grid)
        
        # Column to color mapping
        column_to_color = {0: color1, 1: color2, 2: color3}
        
        # Process each row
        for row in range(3):
            # Find the column index of the shaded cell in this row
            shaded_positions = np.where(input_grid[row] == input_color)[0]
            
            if len(shaded_positions) == 1:  # Should always be exactly 1
                col_index = shaded_positions[0]
                row_color = column_to_color[col_index]
                # Fill entire row with the corresponding color
                output_grid[row, :] = row_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # Generate random colors ensuring they're all different
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        
        input_color = all_colors[0]
        color1 = all_colors[1]  # Column 0 color
        color2 = all_colors[2]  # Column 1 color  
        color3 = all_colors[3]  # Column 2 color
        
        # Store task variables
        taskvars = {
            'input_color': input_color,
            'color1': color1,
            'color2': color2,
            'color3': color3
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Create training examples
        for i in range(num_train_examples):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

