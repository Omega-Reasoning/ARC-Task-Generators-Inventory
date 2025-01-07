import numpy as np
from random import randint, choice
from typing import Dict, Any, Tuple
from arc_task_generator import ARCTaskGenerator, TrainTestData


class TaskjonFUmqwVVkzaeDRu5q2gSGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains red (2) and blue (1) objects (4-way connected cells of the same color), which are 2x2 blocks and 1x2 rectangles."
        ]
        transformation_reasoning_chain = [
            "To construct the output matrix, copy the input matrix.",
            "Check, if a red (2) 2x2 block and a blue (1) 2x2 block are horizontally connected at every edge point, if yes, replace them with a single green (3) rectangle covering their combined area.",
            "All other objects remain unchanged."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['columns']
        matrix = np.zeros((rows, cols), dtype=int)

        # Generate red and blue objects randomly
        num_objects = randint(2, min(rows, cols) // 2)
        objects = [
            {'type': 2, 'shape': '2x2'},  # Red blocks
            {'type': 1, 'shape': '2x2'},  # Blue blocks
            {'type': 2, 'shape': '1x2'},  # Red rectangles
            {'type': 1, 'shape': '1x2'},  # Blue rectangles
        ]

        placed_positions = []
        for _ in range(num_objects):
            obj = choice(objects)
            color, shape = obj['type'], obj['shape']
            if shape == '2x2':
                while True:
                    x, y = randint(0, rows - 2), randint(0, cols - 2)
                    if self._is_free(matrix, x, y, 2, 2, placed_positions):
                        matrix[x:x+2, y:y+2] = color
                        placed_positions.append((x, y, 2, 2))
                        break
            elif shape == '1x2':
                while True:
                    x, y = randint(0, rows - 1), randint(0, cols - 2)
                    if self._is_free(matrix, x, y, 1, 2, placed_positions):
                        matrix[x:x+1, y:y+2] = color
                        placed_positions.append((x, y, 1, 2))
                        break

        # Ensure at least one horizontally connected red-blue 2x2 pair
        self._place_horizontally_connected_pair(matrix)
        return matrix

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_matrix = grid.copy()
        rows, cols = grid.shape

        for x in range(rows - 1):
            for y in range(cols - 3):
                if self._is_connected_pair(grid, x, y):
                    output_matrix[x:x+2, y:y+4] = 3  # Place green rectangle
        return output_matrix

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'rows': randint(5, 30),
            'columns': randint(5, 30)
        }

        train_data, test_data = [], []
        for _ in range(randint(3, 6)):
            input_matrix = self.create_input(taskvars, {})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train_data.append({'input': input_matrix, 'output': output_matrix})

        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data.append({'input': test_input, 'output': test_output})

        return taskvars, {'train': train_data, 'test': test_data}

    def _is_free(self, matrix, x, y, height, width, positions):
        for px, py, ph, pw in positions:
            if not (x + height <= px or px + ph <= x or y + width <= py or py + pw <= y):
                return False
        return True

    def _place_horizontally_connected_pair(self, grid: np.ndarray):
        rows, cols = grid.shape
        while True:
            x, y = randint(0, rows - 2), randint(0, cols - 4)
            if np.all(grid[x:x+2, y:y+2] == 0) and np.all(grid[x:x+2, y+2:y+4] == 0):
                grid[x:x+2, y:y+2] = 2  # Red
                grid[x:x+2, y+2:y+4] = 1  # Blue
                break

    def _is_connected_pair(self, grid: np.ndarray, x: int, y: int) -> bool:
        return (
            np.all(grid[x:x+2, y:y+2] == 2) and  # Red 2x2 block
            np.all(grid[x:x+2, y+2:y+4] == 1)    # Blue 2x2 block
        )

