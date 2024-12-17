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



class SingleColorColumnExpansionTask(ARCTaskGenerator):
    def __init__(self):
        # Initialize reasoning chains
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
        rows = taskvars['rows']
        columns = taskvars['columns']
        num_columns = taskvars['num_columns']
        colors = taskvars['colors']
        
        # Initialize an empty matrix
        matrix = np.zeros((rows, columns), dtype=int)
        
        # Place colored columns at random positions
        column_positions = np.random.choice(range(columns), size=num_columns, replace=False)
        for i, col in enumerate(column_positions):
            matrix[:, col] = colors[i % len(colors)]
        
        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = matrix.shape
        output_matrix = matrix.copy()
        
        # Expand each colored column to the right until blocked
        for col in range(cols):
            if np.all(matrix[:, col] != 0):  # If the column is colored
                color = matrix[0, col]
                for c in range(col + 1, cols):
                    if np.any(matrix[:, c] != 0):  # Stop if another colored column is encountered
                        break
                    output_matrix[:, c] = color
        
        return output_matrix

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'rows': 5,
            'columns': 5,
            'num_columns': np.random.randint(2, 4),  # Between 2 and 3 colored columns
            'colors': np.random.choice(range(1, 10), size=3, replace=False).tolist(),
            'object_color': np.random.choice(range(1, 10))  # Added object_color for templates
        }
        
        train_data = []
        for _ in range(3):  # Create 3 training examples
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({'input': input_matrix, 'output': output_matrix})
        
        test_data = []
        input_matrix = self.create_input(taskvars, {})
        output_matrix = self.transform_input(input_matrix, taskvars)
        test_data.append({'input': input_matrix, 'output': output_matrix})
        
        return taskvars, {'train': train_data, 'test': test_data}

# Test the generator
generator = SingleColorColumnExpansionTask()
taskvars, train_test_data = generator.create_matrices()
ARCTaskGenerator.visualize_train_test_data(train_test_data)
