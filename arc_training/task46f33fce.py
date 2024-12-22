import numpy as np
import random

from arc_task_generator import ARCTaskGenerator

class ARCTask46f33fceGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are square matrices with an even dimension of {vars['dimension']}.",
            "All even rows and columns are empty (row/column counter starts from 0)."
        ]
        transformation_reasoning_chain = [
            "Output matrices are always square.",
            "To create the output matrices, first all even rows and columns of the input matrix are removed.",
            "All remaining cells are scaled by {vars['scale_factor']}, i.e. a single cell becomes a {vars['scale_factor']}x{vars['scale_factor']} submatrix."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, matrixvars):
        dimension = taskvars['dimension']
        matrix = np.zeros((dimension, dimension), dtype=int)
        min_non_empty_cells = random.randint(3, int(dimension*dimension/4-1))
        
        # Fill some cells randomly, avoiding odd rows and columns
        non_empty_cells = 0
        while non_empty_cells < min_non_empty_cells:
            row = random.choice(range(1, dimension, 2))  # Choose even row
            col = random.choice(range(1, dimension, 2))  # Choose even column
            if matrix[row, col] == 0:  # Ensure the cell is empty
                matrix[row, col] = random.randint(1, 9)  # Assign a random color
                non_empty_cells += 1
        return matrix

    def transform_input(self, matrix, taskvars):
        scale_factor = taskvars['scale_factor']

        # Remove odd rows and columns
        reduced_matrix = matrix[1::2, 1::2]

        # Scale remaining cells
        scaled_matrix = np.kron(reduced_matrix, np.ones((scale_factor, scale_factor), dtype=int))
        return scaled_matrix

    def create_matrices(self):
        taskvars = {
            'dimension': random.choice([6, 8, 10, 12, 14]),
            'scale_factor': random.randint(2, 4)  # Ensure the output stays within 30x30
        }

        train = []
        for _ in range(random.randint(3, 5)):
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train.append({'input': input_matrix, 'output': output_matrix})

        # Generate test input-output pair
        test_input_matrix = self.create_input(taskvars, {})
        test_output_matrix = self.transform_input(test_input_matrix, taskvars)

        test = [{'input': test_input_matrix, 'output': test_output_matrix}]

        return taskvars, {'train': train, 'test': test}
