import numpy as np
from typing import Dict, List, Any, Tuple, TypedDict
from random import randint, choice
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
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


class CustomARCTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains rows of the same {color('cell_color')} cells, where the color can vary between rows. The remaining cells are empty (0)."
        ]
        
        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix.",
            "For each row, change the color of its cells based on the number of filled cells:"
            "2 cells → Yellow (4),3 cells → Grey (5) and 4 cells → Pink (6)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = taskvars['rows'], taskvars['columns']
        cell_color = taskvars['cell_color']
        matrix = np.zeros((rows, columns), dtype=int)

        for r in range(rows):
            filled_cells = randint(2, 4)  # Number of filled cells in the row
            indices = np.random.choice(columns, filled_cells, replace=False)
            matrix[r, indices] = cell_color
            cell_color = randint(1, 9)  # Change color for next row
        
        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = matrix.shape
        output_matrix = matrix.copy()

        for r in range(rows):
            filled_cells = np.count_nonzero(matrix[r])
            if filled_cells == 2:
                output_matrix[r, output_matrix[r] != 0] = 4  # Yellow
            elif filled_cells == 3:
                output_matrix[r, output_matrix[r] != 0] = 5  # Grey
            elif filled_cells == 4:
                output_matrix[r, output_matrix[r] != 0] = 6  # Pink
        
        return output_matrix

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'rows': randint(5, 10),  # Random rows (5-10)
            'columns': randint(5, 10),  # Random columns (5-10)
            'cell_color': randint(1, 9)
        }

        train_data = []
        for _ in range(randint(3, 6)):  # 3-6 train matrices
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({'input': input_matrix, 'output': output_matrix})

        test_data = []
        input_matrix = self.create_input(taskvars, {})
        output_matrix = self.transform_input(input_matrix, taskvars)
        test_data.append({'input': input_matrix, 'output': output_matrix})

        train_test_data = {'train': train_data, 'test': test_data}
        return taskvars, train_test_data

# Test the generator
generator = CustomARCTaskGenerator()
taskvars, train_test_data = generator.create_matrices()
print("Task Variables:", taskvars)
generator.visualize_train_test_data(train_test_data)
