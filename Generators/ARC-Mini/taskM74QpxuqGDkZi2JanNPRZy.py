import numpy as np
from typing import Dict, Any, Tuple
from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData

class TaskM74QpxuqGDkZi2JanNPRZyGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices can have different sizes.",
            "They only contain one colored (1-9) cell in each row, with the remaining cells being empty (0)."
        ]
        transformation_reasoning_chain = [
            "To construct the output matrix, initialise a zero-filled matrix of the same size as the input matrix.",
            "Copy colored cell from each row of the input matrix and paste it to the first column of the same row in the output matrix."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = gridvars['size']
        matrix = np.zeros((rows, cols), dtype=int)
        for row in range(rows):
            color = np.random.randint(1, 10)  # Random color (1-9)
            col_position = np.random.randint(0, cols)  # Random column position
            matrix[row, col_position] = color
        return matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = grid.shape
        output_matrix = np.zeros((rows, cols), dtype=int)
        for row in range(rows):
            color_position = np.where(grid[row] != 0)[0][0]  # Find the position of the colored cell
            output_matrix[row, 0] = grid[row, color_position]  # Move it to the first column
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        num_matrices = np.random.randint(3, 7)  # Create 3-6 matrices
        matrix_size = (
            np.random.randint(5, 31),  # Random rows (5-30)
            np.random.randint(5, 31)   # Random columns (5-30)
        )

        taskvars = {
            "size": matrix_size
        }

        train_data = []
        for _ in range(num_matrices - 1):  # Train matrices
            input_matrix = self.create_input(taskvars, {"size": matrix_size})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({"input": input_matrix, "output": output_matrix})

        # Test matrix
        input_matrix = self.create_input(taskvars, {"size": matrix_size})
        output_matrix = self.transform_input(input_matrix, taskvars)
        test_data = [{"input": input_matrix, "output": output_matrix}]

        return taskvars, {"train": train_data, "test": test_data}

