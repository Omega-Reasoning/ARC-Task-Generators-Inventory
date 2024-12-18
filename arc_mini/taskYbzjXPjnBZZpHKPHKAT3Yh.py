import numpy as np
import random
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData


class TasktaskYbzjXPjnBZZpHKPHKAT3YhGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains exactly one blue (1) cell.",
            "The blue (1) cell is positioned such that there are always at least two empty (0) cells in all four directions (up, down, left, and right)."
        ]
        transformation_reasoning_chain = [
            "To construct the output matrix, copy the input matrix.",
            "Color all empty (0) cells adjacent to the blue (1) cell (up, down, left, right) with green (3).",
            "Then, color all empty (0) cells adjacent to the green (3) cells (up, down, left, right) with red (2)."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        matrix = np.zeros((rows, columns), dtype=int)

        # Ensure blue cell placement
        min_margin = 2  # At least two empty cells in all directions
        row = random.randint(min_margin, rows - min_margin - 1)
        col = random.randint(min_margin, columns - min_margin - 1)
        matrix[row, col] = 1  # Place blue cell

        return matrix

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = matrix.shape
        output_matrix = matrix.copy()

        def is_valid(r, c):
            return 0 <= r < rows and 0 <= c < columns

        def color_adjacent(target_value, new_value):
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            to_process = []

            for r in range(rows):
                for c in range(columns):
                    if output_matrix[r, c] == target_value:
                        for dr, dc in directions:
                            nr, nc = r + dr, c + dc
                            if is_valid(nr, nc) and output_matrix[nr, nc] == 0:
                                to_process.append((nr, nc))
            for r, c in to_process:
                output_matrix[r, c] = new_value

        # Step 1: Color cells adjacent to blue (1) with green (3)
        color_adjacent(1, 3)
        # Step 2: Color cells adjacent to green (3) with red (2)
        color_adjacent(3, 2)

        return output_matrix

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        num_train = random.randint(3, 6)  # 3-6 training matrices
        num_test = 1  # One test matrix

        rows = random.randint(5, 30)
        columns = random.randint(5, 30)

        taskvars = {
            'rows': rows,
            'columns': columns
        }

        train_matrices = []
        for _ in range(num_train):
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_matrices.append({'input': input_matrix, 'output': output_matrix})

        test_matrices = []
        input_matrix = self.create_input(taskvars, {})
        output_matrix = self.transform_input(input_matrix, taskvars)
        test_matrices.append({'input': input_matrix, 'output': output_matrix})

        data: TrainTestData = {
            'train': train_matrices,
            'test': test_matrices
        }

        return taskvars, data

