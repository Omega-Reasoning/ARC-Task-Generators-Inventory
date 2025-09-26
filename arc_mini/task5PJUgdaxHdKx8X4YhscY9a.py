# my_task_generator.py

import numpy as np
import random
from typing import Dict, Any, Tuple
# Required imports from the provided base classes and libraries
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry  # we use retry for demonstration if needed
# (We do not need create_object or others here, but you could import them.)
# from transformation_library import find_connected_objects   # not needed here

class Task5PJUgdaxHdKx8X4YhscY9aGenerator(ARCTaskGenerator):
    def __init__(self):
        # Step 1: Define the input reasoning chain
        observation_chain = [
            "Input grids are of size 3x{vars['columns']}.",
            "Each input grid has the top row filled with cells of {color('row_color')} color and the bottom row filled with cells of a different color (1-9).",
            "The middle row contains one {color('fill_color')} cell."
        ]

        # Step 2: Define the transformation reasoning chain
        transformation_chain = [
            "To create the output grid, copy the input grid.",
            "Append two additional rows, the fourth row is identical to the first row of the input grid and the fifth row is identical to the third row of the input grid."
        ]

        # The super-constructor call with the observation and reasoning chains
        super().__init__(observation_chain, transformation_chain)

    def create_grids(self):
        """
        Initialise task variables and create train and test grids.
        Returns (taskvars, train_test_data).
        """

        # 1) Randomize the task variables
        #    'columns': an odd integer between 5 and 19 (inclusive).
        columns_candidates = [c for c in range(5, 30) if c % 2 == 1]
        columns = random.choice(columns_candidates)
        
        # row_color and fill_color from 1..9, must be different
        row_color = random.randint(1, 9)
        fill_color_candidates = [c for c in range(1, 10) if c != row_color]
        fill_color = random.choice(fill_color_candidates)
        
        taskvars = {
            "columns": columns,
            "row_color": row_color,
            "fill_color": fill_color
        }

        # 2) Create the training and test examples
        # We need 3 train examples and 1 test example.
        # Each example differs in the bottom row color.
        def generate_examples(n):
            examples = []
            used_bottom_colors = set()
            
            for _ in range(n):
                # Pick a bottom color different from row_color
                possible_bottom_colors = [
                    c for c in range(1, 10) if c != row_color
                ]
                # Optionally ensure variety in bottom row across examples:
                # remove ones already used
                possible_bottom_colors = [bc for bc in possible_bottom_colors if bc not in used_bottom_colors]
                if not possible_bottom_colors:
                    # If we run out, we relax and allow reuse (unlikely in practice)
                    possible_bottom_colors = [c for c in range(1, 10) if c != row_color]

                bottom_color = random.choice(possible_bottom_colors)
                used_bottom_colors.add(bottom_color)
                
                gridvars = {"bottom_color": bottom_color}
                
                # Create input and output
                input_grid = self.create_input(taskvars, gridvars)
                output_grid = self.transform_input(input_grid, taskvars)
                
                examples.append(GridPair(
                    input=input_grid,
                    output=output_grid
                ))
            return examples
        
        train_examples = generate_examples(3)
        test_examples = generate_examples(1)
        
        train_test_data = TrainTestData(train=train_examples, test=test_examples)
        
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create a 3 x columns input grid:
          - top row fully in row_color
          - bottom row fully in bottom_color
          - middle row with exactly one cell in fill_color
        """
        columns = taskvars['columns']
        row_color = taskvars['row_color']
        fill_color = taskvars['fill_color']
        bottom_color = gridvars['bottom_color']
        
        # Create a 3x(columns) grid initialized to 0
        grid = np.zeros((3, columns), dtype=int)
        
        # Fill top row
        grid[0, :] = row_color
        
        # Fill bottom row
        grid[2, :] = bottom_color
        
        # Place exactly one fill_color cell in the middle row
        middle_fill_col = random.randrange(columns)
        grid[1, middle_fill_col] = fill_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the 3x(columns) input grid into a 5x(columns) output grid:
         - The first 3 rows are copied from the input.
         - The 4th row is identical to the 1st row of the input.
         - The 5th row is identical to the 3rd row of the input.
        """
        rows_in = grid.shape[0]
        cols_in = grid.shape[1]
        
        # Output has 5 rows, same number of columns
        output = np.zeros((5, cols_in), dtype=int)
        
        # Copy original 3 rows
        output[0:3, :] = grid
        
        # 4th row = 1st row of input
        output[3, :] = grid[0, :]
        # 5th row = 3rd row of input
        output[4, :] = grid[2, :]
        
        return output


