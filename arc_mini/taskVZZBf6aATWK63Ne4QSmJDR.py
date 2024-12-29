import numpy as np
import random
from typing import Dict, List, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData

class TasktaskVZZBf6aATWK63Ne4QSmJDRGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains completely filled column {vars['column1']}, {vars['column2']} and {vars['column3']} ,with same-colored cells in each column and colors being {color('column_color1')}, {color('column_color2')} and {color('column_color3')} respectively.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix",
            "Expand each colored column horizontally to the right until, another colored column is encountered or the matrix edge is reached"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        input_matrix = np.zeros((rows, columns), dtype=int)

        for i, col in enumerate([taskvars['column1'], taskvars['column2'], taskvars['column3']]):
            if col <= columns:  # Ensure column index is within bounds
                color = taskvars[f'column_color{i+1}']
                input_matrix[:, col - 1] = color

        return input_matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = grid.shape
        output_matrix = grid.copy()

        for col_idx in range(columns):
            if grid[0, col_idx] != 0:  # Identify a colored column
                color = grid[0, col_idx]
                for row in range(rows):
                    for expand_col in range(col_idx + 1, columns):
                        if grid[row, expand_col] != 0:  # Stop expanding if another colored column is encountered
                            break
                        output_matrix[row, expand_col] = color

        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'rows': random.randint(5, 15),
            'columns': random.randint(5, 15),
            'column1': random.randint(1, 5),
            'column2': random.randint(1, 5),
            'column3': random.randint(1, 5),
            'column_color1': random.randint(1, 9),
            'column_color2': random.randint(1, 9),
            'column_color3': random.randint(1, 9)
        }

        # Ensure columns and colors are distinct
        while len({taskvars['column1'], taskvars['column2'], taskvars['column3']}) < 3:
            taskvars['column3'] = random.randint(1, taskvars['columns'])

        while len({taskvars['column_color1'], taskvars['column_color2'], taskvars['column_color3']}) < 3:
            taskvars['column_color3'] = random.randint(1, 9)

        train_data = []
        for _ in range(random.randint(3, 5)):
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({
                'input': input_matrix,
                'output': output_matrix
            })

        test_data = []
        test_input_matrix = self.create_input(taskvars, {})
        test_output_matrix = self.transform_input(test_input_matrix, taskvars)
        test_data.append({
            'input': test_input_matrix,
            'output': test_output_matrix
        })

        return taskvars, {'train': train_data, 'test': test_data}

