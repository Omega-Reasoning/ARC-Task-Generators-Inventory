import numpy as np
from typing import Dict, List, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData

class ARCTask007bbfb7Generator(ARCTaskGenerator):

    def __init__(self):
        # we use a general version of the task in which the input martrix is copied #columns times horizontally and 
        # rows times vertically
        observation_chain = [
            "The inputs are {vars['row_blocks']}x{vars['column_blocks']} matrices.",
            "Some cells are empty (value 0) and all other cells have a single other color value (1-9)."
        ]
        reasoning_chain = [
            "The output matrix is a {vars['row_blocks']*vars['row_blocks']}x{vars['column_blocks']*vars['column_blocks']} matrix.",
            "It can be subdivided into {vars['row_blocks']}x{vars['column_blocks']} blocks.",
            "If the input matrix contains a non-empty cell in coordinate (i,j) then the corresponding block is a copy of the input matrix."
        ]
        super().__init__(observation_chain, reasoning_chain)

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {}
        taskvars['row_blocks'] = np.random.randint(2, 4)
        taskvars['column_blocks'] = np.random.randint(2, 4)
        num_train = np.random.randint(2, 6)
        train = []

        # Choose three distinct colors
        colors = np.random.choice(range(1, 9), size=3, replace=False)

        # Create training examples
        for _ in range(num_train):
            color = np.random.choice(colors)
            input_matrix = self.create_input(taskvars, { 'color' : color})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train.append({
                "input": input_matrix,
                "output": output_matrix
            })

        # Create test example
        test_color = np.random.choice(colors)
        test_input = self.create_input(taskvars, { 'color' : test_color})
        test_output = self.transform_input(test_input, taskvars)
        test = [{
            "input": test_input,
            "output": test_output
        }]

        return (taskvars, { "train": train, "test": test } )

    def create_input(self, taskvars: Dict[str, Any], matrixvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['row_blocks']
        columns = taskvars['column_blocks']
        color = matrixvars['color']
        
        # Create empty matrix
        matrix = np.zeros((rows, columns), dtype=int)
        
        # Generate random number of positions to fill (at least 1 filled and at least 1 empty)
        num_filled = np.random.randint(1, rows * columns)  
        
        # Generate random unique positions
        positions = np.random.choice(rows * columns, size=num_filled, replace=False)
        
        # Convert positions to 2D indices and fill matrix
        row_indices = positions // columns
        col_indices = positions % columns
        matrix[row_indices, col_indices] = color
        
        return matrix


    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        row_blocks = taskvars['row_blocks']
        column_blocks = taskvars['column_blocks']

        # Create output matrix of appropriate size
        output = np.zeros((row_blocks * row_blocks, column_blocks * column_blocks), dtype=int)

        # Iterate through each position in the input matrix
        for i in range(row_blocks):
            for j in range(column_blocks):
                if matrix[i, j] != 0:
                    # Calculate the starting position of the corresponding block
                    block_start_row = row_blocks * i
                    block_start_col = column_blocks * j

                    # Copy the input matrix into the corresponding block
                    output[block_start_row:block_start_row+row_blocks,
                           block_start_col:block_start_col+column_blocks] = matrix

        return output
