import numpy as np
import random
from typing import Dict, Any, Tuple, List
import numpy as np
from typing import Dict, List, Any, Tuple, TypedDict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from arc_task_generator import ARCTaskGenerator


class MatrixPair(TypedDict):
    input: np.ndarray
    output: np.ndarray

class TrainTestData(TypedDict):
    train: List[MatrixPair]
    test: List[MatrixPair]


class TaskGenerator(ARCTaskGenerator):
    def __init__(self):
        # Input and transformation reasoning chains
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix has; either the first and last columns completely filled, or the first and last rows completely filled with {color('cell_color')}.",
        ]
        transformation_reasoning_chain = [
            "Create an empty matrix of the same size as input matrix",
            "Check if the input matrix has filled columns,if yes, fill the middle cell of the first and last column with {color('fill_color')} of the output matrix.",
            "Check if the input has filled rows,if yes, fill the middle cell of the first and last row {color('fill_color')} of the output matrix.",
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            "rows": random.randint(5, 30),
            "columns": random.randint(5, 30),
            "cell_color": random.choice([8, 6, 1]),
        }
        fill_color_map = {8: 7, 6: 4, 1: 2}
        taskvars["fill_color"] = fill_color_map[taskvars["cell_color"]]
        
        train_matrices = []
        num_train = random.randint(3, 6)
        for _ in range(num_train):
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_matrices.append({"input": input_matrix, "output": output_matrix})
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_matrices = [{"input": test_input, "output": test_output}]
        
        return taskvars, {"train": train_matrices, "test": test_matrices}
    
    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        cell_color = taskvars['cell_color']
        matrix = np.zeros((rows, columns), dtype=int)
        
        # Randomly decide to fill rows or columns
        fill_type = random.choice(['rows', 'columns'])
        if fill_type == 'columns':
            matrix[:, 0] = cell_color
            matrix[:, -1] = cell_color
        else:
            matrix[0, :] = cell_color
            matrix[-1, :] = cell_color
        
        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = matrix.shape
        fill_color = taskvars['fill_color']
        output_matrix = np.zeros((rows, columns), dtype=int)
        
        # Check filled columns
        if np.all(matrix[:, 0] != 0) and np.all(matrix[:, -1] != 0):
            mid_row = rows // 2
            output_matrix[mid_row, 0] = fill_color
            output_matrix[mid_row, -1] = fill_color
        
        # Check filled rows
        if np.all(matrix[0, :] != 0) and np.all(matrix[-1, :] != 0):
            mid_col = columns // 2
            output_matrix[0, mid_col] = fill_color
            output_matrix[-1, mid_col] = fill_color
        
        return output_matrix

# Test the implementation
generator = TaskGenerator()
taskvars, train_test_data = generator.create_matrices()
print("Task Variables:", taskvars)
ARCTaskGenerator.visualize_train_test_data(train_test_data)
