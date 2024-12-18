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



class FillEmptyCellsTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices can have different sizes.",
            "Each input matrix contains exactly one object, made of 4-way connected cells of the same {color('object_color')}, with the remaining cells being empty (0)."
        ]
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and filling empty (0) cells with {color('fill_color')}"
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = np.random.randint(5, 15), np.random.randint(5, 15)
        matrix = np.zeros((rows, cols), dtype=int)
        object_color = np.random.choice([1, 2, 4, 5, 6, 7, 8, 9])
        start_row, start_col = np.random.randint(0, rows), np.random.randint(0, cols)
        matrix[start_row, start_col] = object_color
        frontier = [(start_row, start_col)]
        for _ in range(np.random.randint(5, 20)):
            if not frontier: break
            row, col = frontier.pop()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols and matrix[new_row, new_col] == 0:
                    matrix[new_row, new_col] = object_color
                    frontier.append((new_row, new_col))
        taskvars['object_color'] = object_color
        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        fill_color = 3  # Hardcoded fill color (green)
        output_matrix = matrix.copy()
        output_matrix[output_matrix == 0] = fill_color
        taskvars['fill_color'] = fill_color
        return output_matrix

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {}
        train_data = []
        for _ in range(np.random.randint(3, 6)):
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({'input': input_matrix, 'output': output_matrix})
        test_input_matrix = self.create_input(taskvars, {})
        test_output_matrix = self.transform_input(test_input_matrix, taskvars)
        test_data = [{'input': test_input_matrix, 'output': test_output_matrix}]
        return taskvars, {'train': train_data, 'test': test_data}

generator = FillEmptyCellsTaskGenerator()
taskvars, train_test_data = generator.create_matrices()
ARCTaskGenerator.visualize_train_test_data(train_test_data)
