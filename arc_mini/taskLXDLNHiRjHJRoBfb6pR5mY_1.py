import numpy as np
from typing import Dict, Any, Tuple
import random
from arc_task_generator import ARCTaskGenerator, TrainTestData


class TaskLXDLNHiRjHJRoBfb6pR5mY_1Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains several cells of {color('cell_color')} color in the left half of the matrix and the middle column is entirely filled with grey (5) cells, with the remaining cells being empty (0)."
        ]
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and reflecting cells from the left half to the right half, using the grey (5) middle column as the line of reflection."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        cell_color = taskvars['cell_color']
        middle_col = columns // 2
        
        # Initialize matrix with zeros
        matrix = np.zeros((rows, columns), dtype=int)

        # Fill middle column with grey (5)
        matrix[:, middle_col] = 5

        # Randomly place colored cells in the left half
        num_colored_cells = random.randint(1, rows * middle_col // 2)
        for _ in range(num_colored_cells):
            row = random.randint(0, rows - 1)
            col = random.randint(0, middle_col - 1)
            matrix[row, col] = cell_color
        
        return matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = grid.shape
        middle_col = columns // 2

        # Copy input matrix
        output_matrix = grid.copy()

        # Reflect left half to the right half across the middle column
        for col in range(middle_col):
            reflected_col = columns - col - 1
            output_matrix[:, reflected_col] = grid[:, col]
        
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Define task variables
        rows = random.randint(5, 30)
        columns = random.choice([c for c in range(5, 31) if c % 2 == 1])  # Only odd columns
        cell_color = random.choice([c for c in range(1, 10) if c != 5])  # Exclude grey

        taskvars = {'rows': rows, 'columns': columns, 'cell_color': cell_color}
        
        train_data = []
        for _ in range(random.randint(3, 6)):
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({'input': input_matrix, 'output': output_matrix})

        # Generate one test case
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_data, 'test': test_data}
