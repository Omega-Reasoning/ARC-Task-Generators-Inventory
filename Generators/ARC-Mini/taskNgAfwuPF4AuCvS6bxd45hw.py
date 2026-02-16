import numpy as np
from typing import Dict, Any, Tuple
import random
from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData

class TaskNgAfwuPF4AuCvS6bxd45hwGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "They only contain a single object (4-way connected cells), which is a 2x2 block of either red (2) or blue (1) color and the remaining cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and moving the object one column left, if it is red (2) or one column right, if it is blue (1).",
            "The color of the object is changed; if it is red (2) in the input matrix it becomes blue (1) in the output matrix and if it is blue (1) in the input matrix it becomes red (2) in the output matrix."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        color = gridvars['color']

        # Create an empty matrix
        matrix = np.zeros((rows, columns), dtype=int)

        # Randomly position the 2x2 block within constraints
        row_start = random.randint(1, rows - 3)
        col_start = random.randint(1, columns - 3)

        # Place the block
        matrix[row_start:row_start+2, col_start:col_start+2] = color

        return matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        color = taskvars['color']
        rows, columns = grid.shape

        # Locate the block
        block_pos = np.argwhere(grid == color)
        if block_pos.size == 0:
            raise ValueError("Block not found in input matrix")

        row_start, col_start = block_pos[0]

        # Create the output matrix
        output_matrix = np.zeros_like(grid)
        new_color = 1 if color == 2 else 2

        # Determine the new column position
        new_col_start = col_start - 1 if color == 2 else col_start + 1

        # Place the block in the new position
        output_matrix[row_start:row_start+2, new_col_start:new_col_start+2] = new_color

        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Randomly define task variables
        rows = random.randint(5, 30)
        columns = random.randint(5, 30)
        color = random.choice([1, 2])

        taskvars = {'rows': rows, 'columns': columns, 'color': color}
        train_examples = []

        # Generate train matrices
        for _ in range(random.randint(3, 6)):
            input_matrix = self.create_input(taskvars, {'color': color})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_examples.append({'input': input_matrix, 'output': output_matrix})

        # Generate test matrix
        test_input_matrix = self.create_input(taskvars, {'color': color})
        test_output_matrix = self.transform_input(test_input_matrix, taskvars)
        test_examples = [{'input': test_input_matrix, 'output': test_output_matrix}]

        train_test_data = {'train': train_examples, 'test': test_examples}

        return taskvars, train_test_data
