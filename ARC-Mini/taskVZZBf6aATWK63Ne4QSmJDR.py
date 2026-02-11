import numpy as np
import random
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData

class TaskVZZBf6aATWK63Ne4QSmJDRGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are of fixed size.",
            "Each input matrix contains three completely filled columns appearing in the order {color('column_color1')}, {color('column_color2')}, and {color('column_color3')}.",
            "The positions of these colored columns vary across examples, keeping the color order intact.",
            "All remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix",
            "Expand each colored column horizontally to the right until another colored column is encountered or the matrix edge is reached"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows'] 
        columns = taskvars['columns']
        input_matrix = np.zeros((rows, columns), dtype=int)

        # Get column positions from gridvars
        column_positions = gridvars['column_positions']
        
        # Use the color variables defined in taskvars
        colors = [
            taskvars['column_color1'], 
            taskvars['column_color2'], 
            taskvars['column_color3']
        ]
        
        for i, col in enumerate(column_positions):
            if 0 <= col < columns:  # Ensure column index is within bounds
                input_matrix[:, col] = colors[i]

        return input_matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any], gridvars: Dict[str, Any] = None) -> np.ndarray:
        rows, columns = grid.shape
        output_matrix = grid.copy()
        
        # Find columns that have non-zero values (colored columns)
        colored_cols = [col for col in range(columns) if np.any(grid[:, col] != 0)]
        
        # For each colored column, expand to the right
        for i, col in enumerate(colored_cols):
            color = grid[0, col]  # Get the color (assumed same for the whole column)
            
            # Determine how far to expand (to next colored column or edge)
            end_col = colored_cols[i+1] if i < len(colored_cols)-1 else columns
            
            # Fill all cells between this colored column and the next
            for r in range(rows):
                for c in range(col+1, end_col):
                    output_matrix[r, c] = color
        
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Set grid size between 5 and 25
        rows = random.randint(5, 30)
        columns = max(random.randint(5, 30), 8)  # Ensure at least 8 columns
        
        # Define specific colors for the columns
        column_color1 = random.randint(1, 3)
        column_color2 = random.randint(4, 6)
        column_color3 = random.randint(7, 9)
        
        taskvars = {
            'rows': rows,
            'columns': columns,
            'column_color1': column_color1,
            'column_color2': column_color2,
            'column_color3': column_color3
        }

        train_data = []
        train_column_sets = [
            [0, 3, 6],  # First example columns 
            [1, 4, 7],  # Second example columns
            [2, 5, 7]   # Third example columns
        ]
        
        for col_positions in train_column_sets:
            gridvars = {'column_positions': col_positions}
            input_matrix = self.create_input(taskvars, gridvars)
            output_matrix = self.transform_input(input_matrix, taskvars, gridvars)
            train_data.append({
                'input': input_matrix,
                'output': output_matrix
            })

        test_data = []
        test_gridvars = {'column_positions': [1, 3, 5]}  # Original test position
        test_input_matrix = self.create_input(taskvars, test_gridvars)
        test_output_matrix = self.transform_input(test_input_matrix, taskvars, test_gridvars)
        test_data.append({
            'input': test_input_matrix,
            'output': test_output_matrix
        })

        return taskvars, {'train': train_data, 'test': test_data}