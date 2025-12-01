from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry
from transformation_library import GridObject, GridObjects
import numpy as np
from typing import Dict, Any, List, Tuple
import random

class Task1fad071eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "Multiple 2x2 squares are present in the input grid.",
            "Few of the 2x2 squares are of color {color('color1')} and others are color {color('color2')}.",
            "Few of the cells are also colored {color('color1')} and {color('color2')}, remaining cells are empty(0)."
        ]
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "The output grid has size 1 X (floor({vars['rows']}/2) + 1).",
            "First find all the 2x2 squares with color {color('color1')}.",
            "The cells in the output grid are filled with the color {color('color1')}, corresponding to the same number identified in the previous step."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        rows = random.randint(9, 22)
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)
        taskvars = {
            'rows': rows,
            'color1': color1,
            'color2': color2
        }
        nr_train = random.randint(3, 4)
        train_test_data = self.create_grids_default(nr_train, 1, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        color1 = taskvars['color1']
        color2 = taskvars['color2']
        count_color1 = random.randint(1, (rows // 2) + 1)

        def generate_grid():
            grid = np.zeros((rows, rows), dtype=int)
            possible_positions = [(i, j) for i in range(rows-1) for j in range(rows-1)]
            random.shuffle(possible_positions)
            placed_color1 = []
            for pos in possible_positions:
                i, j = pos
                overlap = False
                for (x, y) in placed_color1:
                    if (i-1 <= x+1 and i+1 >= x-1) and (j-1 <= y+1 and j+1 >= y-1):
                        overlap = True
                        break
                if not overlap:
                    placed_color1.append((i, j))
                    grid[i:i+2, j:j+2] = color1
                    if len(placed_color1) == count_color1:
                        break
            if len(placed_color1) != count_color1:
                return None

            max_color2 = min(3, rows - 2 - len(placed_color1))
            if max_color2 < 1:
                return None
            count_color2 = random.randint(1, max_color2)

            possible_positions_color2 = [(i, j) for i in range(rows-1) for j in range(rows-1)]
            random.shuffle(possible_positions_color2)
            placed_color2 = []
            for pos in possible_positions_color2:
                i, j = pos
                overlap = False
                for (x, y) in placed_color1 + placed_color2:
                    if (i-1 <= x+1 and i+1 >= x-1) and (j-1 <= y+1 and j+1 >= y-1):
                        overlap = True
                        break
                if not overlap:
                    placed_color2.append((i, j))
                    grid[i:i+2, j:j+2] = color2
                    if len(placed_color2) == count_color2:
                        break
            if len(placed_color2) != count_color2:
                return None

            for i in range(rows):
                for j in range(rows):
                    if grid[i, j] == 0 and random.random() < 0.1:
                        color = color1 if random.random() < 0.5 else color2
                        allowed = True
                        # Check if any adjacent cells (including diagonals) have a color
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if (di != 0 or dj != 0):  # Skip the cell itself
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < rows and 0 <= nj < rows and grid[ni, nj] != 0:
                                        allowed = False
                                        break
                            if not allowed:
                                break
                                
                        # Original 2x2 square check
                        if allowed:
                            squares = [
                                (i-1, j-1),
                                (i-1, j),
                                (i, j-1),
                                (i, j)
                            ]
                            for (a, b) in squares:
                                if a >= 0 and b >= 0 and a+1 < rows and b+1 < rows:
                                    count = 0
                                    for x in range(a, a+2):
                                        for y in range(b, b+2):
                                            if (x, y) != (i, j) and grid[x, y] == color:
                                                count += 1
                                    if count >= 3:
                                        allowed = False
                                        break
                        if allowed:
                            grid[i, j] = color
            return grid

        return retry(generate_grid, lambda g: g is not None)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        color1 = taskvars['color1']
        output_cols = (rows // 2) + 1
        
        # Count the number of 2x2 squares of color1
        count_color1_squares = 0
        for i in range(rows-1):
            for j in range(rows-1):
                if (grid[i:i+2, j:j+2] == color1).all():
                    count_color1_squares += 1
        
        # Create output grid with zeros
        output = np.zeros((1, output_cols), dtype=int)
        # Fill first count_color1_squares cells with color1
        output[0, :count_color1_squares] = color1   
        
        return output

if __name__ == "__main__":
    generator = ARCTask1fad071eGenerator()
    taskvars, train_test_data = generator.create_grids()
    ARCTaskGenerator.visualize_train_test_data(train_test_data)