from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taskd4a91cb9(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "The grid contains exactly two uniquely colored cells: one of {color('color_1')} and one of {color('color_2')}.",
            "These two colored cells are positioned such that they do not share the same row or the same column.",
            "All other cells in the grid are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The locations of the two single colored cells are identified.",
            "A path is drawn to connect them using {color('color_3')} color, forming an L-shape by filling all cells in the row of the {color('color_1')} cell between its column and the column of the {color('color_2')} cell, as well as all cells in the column of the {color('color_2')} cell between the row of the {color('color_1')} cell and its own row.",
            "The two original single colored cells retain their initial colors."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Set up colors for the task
        taskvars = {
            'color_1': random.randint(1, 9),
            'color_2': 0,
            'color_3': 0
        }
        
        # Make sure colors are unique
        while taskvars['color_2'] == 0 or taskvars['color_2'] == taskvars['color_1']:
            taskvars['color_2'] = random.randint(1, 9)
            
        while taskvars['color_3'] == 0 or taskvars['color_3'] == taskvars['color_1'] or taskvars['color_3'] == taskvars['color_2']:
            taskvars['color_3'] = random.randint(1, 9)
        
        # Generate train and test examples
        train_examples = []
        num_train_examples = random.randint(3, 6)
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate one test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{
                'input': test_input,
                'output': test_output
            }]
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Choose a random grid size between 5x5 and 10x10
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Place the first colored cell
        r1 = random.randint(0, rows - 1)
        c1 = random.randint(0, cols - 1)
        grid[r1, c1] = taskvars['color_1']
        
        # Place the second colored cell in a different row and column
        r2 = random.choice([r for r in range(rows) if r != r1])
        c2 = random.choice([c for c in range(cols) if c != c1])
        grid[r2, c2] = taskvars['color_2']
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find the positions of the two colored cells
        color_1_positions = np.where(grid == taskvars['color_1'])
        color_2_positions = np.where(grid == taskvars['color_2'])
        
        r1, c1 = color_1_positions[0][0], color_1_positions[1][0]
        r2, c2 = color_2_positions[0][0], color_2_positions[1][0]
        
        # Fill the L-shape connecting the two points
        # First, fill the horizontal part (row of color_1)
        start_col = min(c1, c2)
        end_col = max(c1, c2)
        for c in range(start_col, end_col + 1):
            if (r1, c) != (r1, c1) and (r1, c) != (r2, c2):
                output_grid[r1, c] = taskvars['color_3']
        
        # Then, fill the vertical part (column of color_2)
        start_row = min(r1, r2)
        end_row = max(r1, r2)
        for r in range(start_row, end_row + 1):
            if (r, c2) != (r1, c1) and (r, c2) != (r2, c2):
                output_grid[r, c2] = taskvars['color_3']
        
        return output_grid
