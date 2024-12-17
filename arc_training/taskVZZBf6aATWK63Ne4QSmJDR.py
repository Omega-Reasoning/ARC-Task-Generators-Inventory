import numpy as np
from abc import ABC
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


class MyARCTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}",
            "Each input matrix contains several same {color('object_color')} columns, with the remaining cells being empty (0)."
        ]
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix",
            "Expand each colored column horizontally to the right until, another colored column is encountered or the matrix edge is reached"
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = taskvars['rows'], taskvars['columns']
        num_colors = taskvars['num_colors']
        matrix = np.zeros((rows, columns), dtype=int)

        column_indices = np.random.choice(columns, num_colors, replace=False)
        colors = np.random.choice(range(1, 10), num_colors, replace=False)

        for col_idx, color in zip(column_indices, colors):
            matrix[:, col_idx] = color

        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = matrix.shape
        transformed = matrix.copy()

        for col in range(columns):
            if np.any(matrix[:, col] > 0):
                color = matrix[0, col]
                for right_col in range(col + 1, columns):
                    if np.any(matrix[:, right_col] > 0):
                        break
                    transformed[:, right_col] = color

        return transformed

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'rows': np.random.randint(5, 21),
            'columns': np.random.randint(5, 21),
            'num_colors': np.random.randint(2, 5)
        }

        train = []
        for _ in range(np.random.randint(3, 6)):
            input_matrix = self.create_input(taskvars, None)
            output_matrix = self.transform_input(input_matrix, taskvars)
            train.append({'input': input_matrix, 'output': output_matrix})

        test_input = self.create_input(taskvars, None)
        test_output = self.transform_input(test_input, taskvars)
        test = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train, 'test': test}

# Test code
generator = MyARCTaskGenerator()
_, train_test_data = generator.create_matrices()
ARCTaskGenerator.visualize_train_test_data(train_test_data)
