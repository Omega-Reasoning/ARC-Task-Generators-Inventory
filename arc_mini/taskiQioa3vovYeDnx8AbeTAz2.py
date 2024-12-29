import numpy as np
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData

# Subclassing the ARCTaskGenerator
class TasktaskiQioa3vovYeDnx8AbeTAz2Generator(ARCTaskGenerator):
    def __init__(self):
        # Input and transformation reasoning chains
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}",
            "Each input matrix contains one blue (1) rectangle and one red (2) cell, with the remaining cells being empty (0)."
        ]
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix.",
            "Replace red (2) single-cell with a 2x2 green (3) square.",
            "The top-left corner of the green square should align with the location of the red cell."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, columns = taskvars['rows'], taskvars['columns']
        matrix = np.zeros((rows, columns), dtype=int)
        
        # Random placement of blue rectangle (1)
        rect_height, rect_width = gridvars['rect_height'], gridvars['rect_width']
        rect_row = np.random.randint(0, rows - rect_height + 1)
        rect_col = np.random.randint(0, columns - rect_width + 1)
        matrix[rect_row:rect_row+rect_height, rect_col:rect_col+rect_width] = 1
        
        # Place red (2) cell ensuring adjacency constraints
        valid = False
        while not valid:
            red_row = np.random.randint(1, rows - 1)
            red_col = np.random.randint(1, columns - 1)
            if np.all(matrix[red_row-1:red_row+2, red_col-1:red_col+2] == 0):
                matrix[red_row, red_col] = 2
                valid = True
        return matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_matrix = grid.copy()
        red_row, red_col = np.argwhere(grid == 2)[0]
        # Replace red cell with a 2x2 green square
        output_matrix[red_row:red_row+2, red_col:red_col+2] = 3
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'rows': np.random.randint(5, 31),
            'columns': np.random.randint(5, 31)
        }
        gridvars = {
            'rect_height': np.random.randint(1, taskvars['rows'] // 2),
            'rect_width': np.random.randint(1, taskvars['columns'] // 2)
        }
        
        train = []
        for _ in range(np.random.randint(3, 7)):
            input_matrix = self.create_input(taskvars, gridvars)
            output_matrix = self.transform_input(input_matrix, taskvars)
            train.append({'input': input_matrix, 'output': output_matrix})
        
        # Create one test example
        test_input_matrix = self.create_input(taskvars, gridvars)
        test_output_matrix = self.transform_input(test_input_matrix, taskvars)
        test = [{'input': test_input_matrix, 'output': test_output_matrix}]
        
        return taskvars, {'train': train, 'test': test}

