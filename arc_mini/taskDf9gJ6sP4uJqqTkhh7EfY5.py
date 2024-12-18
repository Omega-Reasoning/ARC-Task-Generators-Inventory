import numpy as np
import random
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData


class TaskDf9gJ6sP4uJqqTkhh7EfY5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices can have different sizes.",
            "In each input matrix there are exactly two colored (1-9) cells, which are located in two different columns, with the remaining cells being empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and for each column containing a colored cell, fill the entire column with the same color as the cell."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        size = random.randint(5, 30)
        matrix = np.zeros((size, size), dtype=int)

        col1, col2 = random.sample(range(size), 2)
        color1, color2 = random.sample(range(1, 10), 2)

        row1, row2 = random.randint(0, size - 1), random.randint(0, size - 1)
        matrix[row1, col1] = color1
        matrix[row2, col2] = color2

        matrixvars.update({"col1": col1, "col2": col2, "color1": color1, "color2": color2})

        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_matrix = matrix.copy()
        non_zero_columns = np.where(matrix.any(axis=0))[0]

        for col in non_zero_columns:
            color = matrix[np.nonzero(matrix[:, col])[0][0], col]
            output_matrix[:, col] = color

        return output_matrix

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {}
        train_pairs = []

        for _ in range(random.randint(3, 6)):
            matrixvars = {}
            input_matrix = self.create_input(taskvars, matrixvars)
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_pairs.append({"input": input_matrix, "output": output_matrix})

        test_pairs = []
        for _ in range(1):
            matrixvars = {}
            input_matrix = self.create_input(taskvars, matrixvars)
            output_matrix = self.transform_input(input_matrix, taskvars)
            test_pairs.append({"input": input_matrix, "output": output_matrix})

        train_test_data = {"train": train_pairs, "test": test_pairs}

        return taskvars, train_test_data

