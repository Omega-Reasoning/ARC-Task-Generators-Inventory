import numpy as np
import random
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


class DiagonalExpansionTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}",
            "Each input matrix contains exactly one {color('cell_color')} cell, with the rest of the cells being empty (0)."
        ]

        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix.",
            "Extend the {color('cell_color')} cell diagonally downwards (bottom-right direction) with the same {color('cell_color')} until the edge of the matrix is reached."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = taskvars['rows'], taskvars['columns']
        cell_color = taskvars['cell_color']
        matrix = np.zeros((rows, columns), dtype=int)

        # Place a single cell with the specified color at a random position
        x, y = np.random.randint(0, rows), np.random.randint(0, columns)
        matrix[x, y] = cell_color
        matrixvars['start_position'] = (x, y)
        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = matrix.shape
        cell_color = taskvars['cell_color']
        output_matrix = matrix.copy()

        # Find the position of the single cell
        x, y = np.argwhere(matrix == cell_color)[0]

        # Extend diagonally until edge
        while x < rows and y < columns:
            output_matrix[x, y] = cell_color
            x += 1
            y += 1
        return output_matrix

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        num_train = np.random.randint(3, 6)  # 3-6 train examples
        num_test = 1  # One test example

        # Define task variables
        taskvars = {
            'rows': np.random.randint(5, 31),
            'columns': np.random.randint(5, 31),
            'cell_color': np.random.randint(1, 10)
        }

        train_data = []
        for _ in range(num_train):
            matrixvars = {}
            input_matrix = self.create_input(taskvars, matrixvars)
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({
                'input': input_matrix,
                'output': output_matrix
            })

        # Create test input/output pair
        matrixvars = {}
        test_input = self.create_input(taskvars, matrixvars)
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_data, 'test': test_data}

# Test code
task_generator = DiagonalExpansionTaskGenerator()
taskvars, train_test_data = task_generator.create_matrices()
print("Task Variables:", taskvars)
ARCTaskGenerator.visualize_train_test_data(train_test_data)

class ARCAGITaskGenerator(ARCTaskGenerator):
    def __init__(self):
        # Input and transformation reasoning chains
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains exactly one {color('cell_color')} cell, with the rest of the cells being empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix.",
            "Extend the {color('cell_color')} cell diagonally downwards (bottom-right direction) with the same {color('cell_color')} until the edge of the matrix is reached."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        cell_color = taskvars['cell_color']
        
        # Initialize an empty matrix of the given size
        matrix = np.zeros((rows, columns), dtype=int)
        
        # Place exactly one colored cell at a random location
        x = random.randint(0, rows - 1)
        y = random.randint(0, columns - 1)
        matrix[x, y] = cell_color
        
        matrixvars['cell_position'] = (x, y)  # Save position for consistency
        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        cell_color = taskvars['cell_color']
        rows, cols = matrix.shape
        output_matrix = matrix.copy()
        
        # Locate the position of the colored cell
        x, y = np.argwhere(matrix == cell_color)[0]
        
        # Extend the cell diagonally downwards (bottom-right)
        while x < rows and y < cols:
            output_matrix[x, y] = cell_color
            x += 1
            y += 1
        
        return output_matrix

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'rows': random.randint(5, 30),
            'columns': random.randint(5, 30),
            'cell_color': random.randint(1, 9)  # Random color between 1 and 9
        }
        
        train_examples = []
        for _ in range(random.randint(3, 6)):  # Generate 3-6 training matrices
            matrixvars = {}
            input_matrix = self.create_input(taskvars, matrixvars)
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_examples.append({'input': input_matrix, 'output': output_matrix})
        
        # Generate one test example
        matrixvars = {}
        test_input_matrix = self.create_input(taskvars, matrixvars)
        test_output_matrix = self.transform_input(test_input_matrix, taskvars)
        test_examples = [{'input': test_input_matrix, 'output': test_output_matrix}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

# Test the implementation
generator = ARCAGITaskGenerator()
taskvars, train_test_data = generator.create_matrices()
print("Task Variables:", taskvars)
ARCTaskGenerator.visualize_train_test_data(train_test_data)
