from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity

class TaskeV8NEV3ouyuG6eUVBGznv2Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a single group of 8-way connected cells of {color('cell_color1')} and {color('cell_color2')} colors.",
            "All other cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid is created by reflecting each 8-way connected cell horizontally to its opposite side, using the middle column as the line of reflection.",
            "Any cell on the middle column remains in place, but its color changes during the reflection.",
            "During the reflection, the 8-way connected cells of {color('cell_color1')} and {color('cell_color2')} are transformed into {color('cell_color3')} and {color('cell_color4')}, respectively."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        rows = random.randint(5, 10)
        cols = random.choice([c for c in range(5, 16) if c % 2 == 1])
        colors = random.sample(range(1, 10), 4)
        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': colors[0],
            'cell_color2': colors[1],
            'cell_color3': colors[2],
            'cell_color4': colors[3],
        }

        num_train = random.choice([3, 4])
        train_examples = []
        middle_col_included = False

        for _ in range(num_train):
            grid_in = self.create_input(taskvars, {}, ensure_middle_col=not middle_col_included)
            grid_out = self.transform_input(grid_in, taskvars)
            train_examples.append({'input': grid_in, 'output': grid_out})
            if not middle_col_included and np.any(grid_in[:, taskvars['cols'] // 2] != 0):
                middle_col_included = True

        test_input = self.create_input(taskvars, {}, ensure_middle_col=False)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars, gridvars, ensure_middle_col=False):
        rows, cols = taskvars['rows'], taskvars['cols']
        color1, color2 = taskvars['cell_color1'], taskvars['cell_color2']
        grid = np.zeros((rows, cols), dtype=int)

        total_size = random.randint(10, 15)
        current_size = 0
        cell_positions = set()

        # Decide majority side
        place_left = random.choice([True, False])
        min_col = 0 if place_left else cols // 2
        max_col = cols // 2 if place_left else cols

        # Create a connected group of cells
        start_row = random.randint(0, rows - 1)
        start_col = random.randint(min_col, max_col - 1)
        stack = [(start_row, start_col)]

        while current_size < total_size and stack:
            r, c = stack.pop()

            if (r, c) not in cell_positions and 0 <= r < rows and min_col <= c < max_col:
                if ensure_middle_col and c == cols // 2:
                    ensure_middle_col = False  # Ensure at least one cell in the middle column

                chosen_color = color1 if current_size % 2 == 0 else color2
                grid[r, c] = chosen_color
                cell_positions.add((r, c))
                current_size += 1

                # Add neighbors to the stack
                neighbors = [(r + dr, c + dc) for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]]
                random.shuffle(neighbors)
                stack.extend(neighbors)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        rows, cols = grid.shape
        mid_col = cols // 2
        out_grid = np.zeros_like(grid)

        color1, color2 = taskvars['cell_color1'], taskvars['cell_color2']
        color3, color4 = taskvars['cell_color3'], taskvars['cell_color4']

        for r in range(rows):
            for c in range(cols):
                original_color = grid[r, c]
                if original_color == 0:
                    continue

                if c == mid_col:
                    if original_color == color1:
                        out_grid[r, c] = color3
                    elif original_color == color2:
                        out_grid[r, c] = color4
                else:
                    reflected_c = (cols - 1) - c
                    if original_color == color1:
                        out_grid[r, reflected_c] = color3
                    elif original_color == color2:
                        out_grid[r, reflected_c] = color4

        return out_grid


