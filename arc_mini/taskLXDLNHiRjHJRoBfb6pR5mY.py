import numpy as np
from random import randint, choice
from typing import Dict, List, Any, Tuple, TypedDict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from arc_task_generator import ARCTaskGenerator

# Define types
class MatrixPair(TypedDict):
    input: np.ndarray
    output: np.ndarray

class TrainTestData(TypedDict):
    train: List[MatrixPair]
    test: List[MatrixPair]

class ARCAGITaskGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input matrix is of size 3x5.",
            "Each input matrix contains red (2) and green (3) cells in the left half of the matrix, ",
            "and the middle column is entirely filled with grey (5) cells, separating the left and right halves of the matrix."
        ]
        transformation_reasoning_chain = [
            "Copy the input matrix.",
            "Reflect the red (2) and green (3) cells from the left half to the right half, using the middle column as the line of reflection.",
            "During the reflection, the Red (2) cells of the input matrix turn into yellow (4) in the output matrix.",
            "Similarly the Green (3) cells of the input matrix turn into blue (1) in the output."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        matrix = np.zeros((3, 5), dtype=int)
        matrix[:, 2] = 5  # Middle column grey
        red_positions = set()
        green_positions = set()

        while len(red_positions) < 1:
            red_positions.add((randint(0, 2), randint(0, 1)))

        while len(green_positions) < 1:
            pos = (randint(0, 2), randint(0, 1))
            if pos not in red_positions:
                green_positions.add(pos)

        for row, col in red_positions:
            matrix[row, col] = 2
        for row, col in green_positions:
            matrix[row, col] = 3

        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_matrix = matrix.copy()
        for row in range(3):
            for col in range(2):
                if matrix[row, col] == 2:
                    output_matrix[row, 4 - col] = 4
                elif matrix[row, col] == 3:
                    output_matrix[row, 4 - col] = 1
        return output_matrix

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        train = []
        for _ in range(2):  # Generate 2 train examples
            input_matrix = self.create_input({}, {})
            output_matrix = self.transform_input(input_matrix, {})
            train.append({"input": input_matrix, "output": output_matrix})

        test_input = self.create_input({}, {})
        test_output = self.transform_input(test_input, {})
        test = [{"input": test_input, "output": test_output}]

        return {}, {"train": train, "test": test}


# Test the generator
generator = ARCAGITaskGenerator()
_, data = generator.create_matrices()
generator.visualize_train_test_data(data)
