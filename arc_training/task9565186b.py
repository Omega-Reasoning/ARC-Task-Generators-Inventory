from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class ARCTask9565186bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has dimension {vars['rows']} X {vars['rows']}.",
            "At least one row or column or both are filled with the cells of color in_color(between 1-9).",
            "The remaining cells are filled with random color random_color(between 1-9)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "The rows or columns which have the same colored cells remain unchanged.",
            "The cells which have the same color as the in_color are not changed.",
            "The remaining cells are colored with color {color('out_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'rows': random.randint(10, 30),
            'out_color': random.randint(1, 9)
        }
        
        # Create train and test data
        num_train_examples = random.randint(3, 4)
        train_data = []
        
        # Create train examples
        for _ in range(num_train_examples):
            # Randomly choose if we'll have uniform rows or columns and colors
            gridvars = {
                'use_rows': random.choice([True, False]),
                'num_uniform': random.randint(1, max(1, taskvars['rows'] // 3)),  # At least 1, but not too many
                'in_color': random.randint(1, 9)                
            }
            
            # Ensure input and output colors are different
            while taskvars['out_color'] == gridvars['in_color']:
                taskvars['out_color'] = random.randint(1, 9)
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {
            'use_rows': random.choice([True, False]),
            'num_uniform': random.randint(1, max(1, taskvars['rows'] // 3)),
            'in_color': random.randint(1, 9)
        }
        
        # Ensure input and output colors are different
        while taskvars['out_color'] == test_gridvars['in_color']:
            taskvars['out_color'] = random.randint(1, 9)
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_data = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_data,
            'test': test_data
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        in_color = gridvars['in_color']
        
        # Create an empty grid
        grid = np.zeros((rows, rows), dtype=int)
        
        # Fill the grid with random colors initially (can include in_color)
        for r in range(rows):
            for c in range(rows):
                grid[r, c] = random.randint(1, 9)
        
        # Decide whether to fill rows, columns, or both
        fill_type = random.choice(['rows', 'columns', 'both'])
        num_uniform = gridvars['num_uniform']
        
        # Select the indices to make uniform
        if fill_type == 'rows' or fill_type == 'both':
            uniform_rows = random.sample(range(rows), min(num_uniform, rows))
            for r in uniform_rows:
                grid[r, :] = in_color
                
        if fill_type == 'columns' or fill_type == 'both':
            uniform_cols = random.sample(range(rows), min(num_uniform, rows))
            for c in uniform_cols:
                grid[:, c] = in_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows = grid.shape[0]
        out_color = taskvars['out_color']
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find rows and columns that are filled with a uniform color
        uniform_rows = []
        uniform_cols = []
        uniform_colors = set()
        
        # Check for uniform rows
        for r in range(rows):
            row_values = grid[r, :]
            if np.all(row_values == row_values[0]):
                uniform_rows.append(r)
                uniform_colors.add(int(row_values[0]))
        
        # Check for uniform columns
        for c in range(rows):
            col_values = grid[:, c]
            if np.all(col_values == col_values[0]):
                uniform_cols.append(c)
                uniform_colors.add(int(col_values[0]))
        
        # Fill all cells that don't have a uniform color with out_color
        for r in range(rows):
            for c in range(rows):
                cell_color = grid[r, c]
                # Keep cells that have the same color as any uniform row/column
                if cell_color in uniform_colors:
                    continue
                # Change all other cells to out_color
                output_grid[r, c] = out_color
        
        return output_grid