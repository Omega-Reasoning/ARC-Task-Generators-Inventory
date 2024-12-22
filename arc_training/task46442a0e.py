import numpy as np
import random
from typing import Dict, Any, Tuple

from arc_task_generator import ARCTaskGenerator, TrainTestData

class ARCTask46442a0eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are square matrices with 2, 3 or 4 rows/columns.",
            "There are at least two different colors in the input matrix."
        ]
        transformation_reasoning_chain = [
            "Output matrices are always square and have double the dimensions of the input matrices.",
            "The top-left quadrant of the output matrix is identical to the input matrix.",
            "The top-right quadrant is a copy of the input matrix after a 90 degree clockwise rotation.",
            "Similarly, the bottom-right quadrant is a result of copying the input matrix and rotating it 180 degrees.",
            "Finally, the same is done for the bottom left quadrant with a 270 degree rotation."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        size = random.choice([2, 3, 4])  # Choose size between 2x2, 3x3, or 4x4
        matrix = np.zeros((size, size), dtype=int)

        # Ensure at least two different colors in the input matrix
        color_1, color_2 = random.sample(range(1, 10), 2)
        num_cells = size * size

        # Randomly assign colors to at least two cells
        color_positions = random.sample(range(num_cells), 2)
        matrix[color_positions[0] // size, color_positions[0] % size] = color_1
        matrix[color_positions[1] // size, color_positions[1] % size] = color_2

        # Randomly populate the rest of the matrix with colors or remain empty
        for i in range(num_cells):
            if i not in color_positions:
                matrix[i // size, i % size] = random.choice([0, color_1, color_2])

        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        size = matrix.shape[0]
        new_size = size * 2
        output = np.zeros((new_size, new_size), dtype=int)

        # Top-left quadrant: identical to the input matrix
        output[:size, :size] = matrix

        # Top-right quadrant: 90 degree rotation
        output[:size, size:] = np.rot90(matrix, k=-1)

        # Bottom-right quadrant: 180 degree rotation
        output[size:, size:] = np.rot90(matrix, k=2)

        # Bottom-left quadrant: 270 degree rotation
        output[size:, :size] = np.rot90(matrix, k=1)

        return output

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        train_data = []
        num_train = random.randint(3, 5)  # Create 3-5 training examples

        for _ in range(num_train):
            taskvars = {}
            matrixvars = {}

            input_matrix = self.create_input(taskvars, matrixvars)
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({"input": input_matrix, "output": output_matrix})

        # Create a single test example
        taskvars = {}
        matrixvars = {}
        test_input = self.create_input(taskvars, matrixvars)
        test_output = self.transform_input(test_input, taskvars)

        test_data = [{"input": test_input, "output": test_output}]

        return {}, {"train": train_data, "test": test_data}
