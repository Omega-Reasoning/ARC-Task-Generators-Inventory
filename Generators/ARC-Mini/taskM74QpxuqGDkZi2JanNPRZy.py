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
        
        # Collect all colored (non-zero) cells from the input, reading row by row, left to right
        colored_cells = []
        for row in range(rows):
            for col in range(cols):
                if grid[row, col] != 0:
                    colored_cells.append(grid[row, col])
        
        # Place collected colored cells in the first column of the output, one per row
        for row in range(min(rows, len(colored_cells))):
            output_matrix[row, 0] = colored_cells[row]
        
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        num_matrices = np.random.randint(3, 7)  # Create 3-6 matrices
        size_rows = np.random.randint(5, 31)  # Random rows (5-30)
        size_cols = np.random.randint(5, 31)  # Random columns (5-30)
        matrix_size = (size_rows, size_cols)

        taskvars = {
            "size_rows": size_rows,
            "size_cols": size_cols
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

