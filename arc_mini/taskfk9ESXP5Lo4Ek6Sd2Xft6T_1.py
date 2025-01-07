import numpy as np
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData

class Taskfk9ESXP5Lo4Ek6Sd2Xft6T_1Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices can have different sizes.",
            "Each input matrix contains exactly one object, made of 4-way connected {color('object_color')} cells, with the remaining cells being empty (0)."
        ]
        transformation_reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and filling empty (0) cells with {color('fill_color')}."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = np.random.randint(5, 15), np.random.randint(5, 15)
        matrix = np.zeros((rows, cols), dtype=int)
        start_row, start_col = np.random.randint(0, rows), np.random.randint(0, cols)
        matrix[start_row, start_col] = taskvars['object_color']
        frontier = [(start_row, start_col)]
        for _ in range(np.random.randint(5, 20)):
            if not frontier: break
            row, col = frontier.pop()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < rows and 0 <= new_col < cols and matrix[new_row, new_col] == 0:
                    matrix[new_row, new_col] = taskvars['object_color']
                    frontier.append((new_row, new_col))
        
        return matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_matrix = grid.copy()
        output_matrix[output_matrix == 0] = taskvars['fill_color']
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {}
        object_color = np.random.choice(range(1, 10))
        available_fill_colors = [c for c in range(1, 10) if c != object_color]
        fill_color = np.random.choice(available_fill_colors)
        taskvars['object_color'] = object_color
        taskvars['fill_color'] = fill_color

        train_data = []
        for _ in range(np.random.randint(3, 6)):
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({'input': input_matrix, 'output': output_matrix})
        test_input_matrix = self.create_input(taskvars, {})
        test_output_matrix = self.transform_input(test_input_matrix, taskvars)
        test_data = [{'input': test_input_matrix, 'output': test_output_matrix}]
        return taskvars, {'train': train_data, 'test': test_data}
