import numpy as np
import random
from typing import Dict, Any, Tuple
from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData

class TaskankHoMoAtA9yJX7B57KeDdGenerator(ARCTaskGenerator):
    def __init__(self):
        # Input and transformation reasoning chains
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains exactly one {color('cell_color')} cell, with the rest of the cells being empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix.",
            "Extend the {color('cell_color')} cell diagonally downwards (bottom-right direction) with the same color until the edge of the matrix is reached."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        cell_color = taskvars['cell_color']
        
        # Initialize an empty matrix of the given size
        matrix = np.zeros((rows, columns), dtype=int)
        
        # Place exactly one colored cell at a random location
        x = random.randint(0, rows - 1)
        y = random.randint(0, columns - 1)
        matrix[x, y] = cell_color
        
        gridvars['cell_position'] = (x, y)  # Save position for consistency
        return matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        cell_color = taskvars['cell_color']
        rows, cols = grid.shape
        output_matrix = grid.copy()
        
        # Locate the position of the colored cell
        x, y = np.argwhere(grid == cell_color)[0]
        
        # Extend the cell diagonally downwards (bottom-right)
        while x < rows and y < cols:
            output_matrix[x, y] = cell_color
            x += 1
            y += 1
        
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'rows': random.randint(5, 30),
            'columns': random.randint(5, 30),
            'cell_color': random.randint(1, 9)  # Random color between 1 and 9
        }
        
        train_examples = []
        for _ in range(random.randint(3, 6)):  # Generate 3-6 training matrices
            gridvars = {}
            input_matrix = self.create_input(taskvars, gridvars)
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_examples.append({'input': input_matrix, 'output': output_matrix})
        
        # Generate one test example
        gridvars = {}
        test_input_matrix = self.create_input(taskvars, gridvars)
        test_output_matrix = self.transform_input(test_input_matrix, taskvars)
        test_examples = [{'input': test_input_matrix, 'output': test_output_matrix}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

