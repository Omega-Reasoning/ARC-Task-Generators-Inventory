import numpy as np
import random
import numpy as np
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData

class Tasktaskby2YUGnMUpGwiL2365k779_1Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains one {color('cell_color1')} and one {color('cell_color2')} cell, in both row no. {vars['row1']} and row no. {vars['row2']}, with the remaining cells being empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and expanding each colored cell horizontally from left to right with the same color until another colored cell is encountered or the matrix border is reached."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = taskvars['rows'], taskvars['columns']
        row1, row2 = taskvars['row1'], taskvars['row2']
        cell_color1, cell_color2 = taskvars['cell_color1'], taskvars['cell_color2']

        matrix = np.zeros((rows, columns), dtype=int)
        
        # Randomize positions for cell_color1 and cell_color2 in row1 and row2
        col1, col2 = random.sample(range(columns), 2)
        matrix[row1, col1] = cell_color1
        matrix[row1, col2] = cell_color2

        col3, col4 = random.sample(range(columns), 2)
        while col3 == col1 or col4 == col2:  # Ensure different columns for row2
            col3, col4 = random.sample(range(columns), 2)

        matrix[row2, col3] = cell_color1
        matrix[row2, col4] = cell_color2

        return matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = grid.shape
        output_matrix = grid.copy()

        for row in range(rows):
            for col in range(columns):
                if grid[row, col] != 0:  # Non-empty cell
                    color = grid[row, col]
                    # Expand horizontally to the right
                    for right_col in range(col + 1, columns):
                        if grid[row, right_col] != 0:  # Stop at another colored cell
                            break
                        output_matrix[row, right_col] = color
        
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Randomize task variables
        rows = random.randint(5, 30)
        columns = random.randint(5, 30)
        cell_color1, cell_color2 = random.sample(range(1, 10), 2)
        row1, row2 = random.sample(range(rows), 2)

        taskvars = {
            'rows': rows,
            'columns': columns,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2,
            'row1': row1,
            'row2': row2
        }

        train_data = []
        for _ in range(random.randint(3, 6)):  # Generate 3-6 training examples
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({'input': input_matrix, 'output': output_matrix})

        # Create one test example
        input_matrix = self.create_input(taskvars, {})
        output_matrix = self.transform_input(input_matrix, taskvars)

        test_data = [{'input': input_matrix, 'output': output_matrix}]

        return taskvars, {'train': train_data, 'test': test_data}
