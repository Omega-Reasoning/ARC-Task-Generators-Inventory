from Framework.arc_task_generator import ARCTaskGenerator
import numpy as np
import random

class TaskAA18DE87(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid contains a wave or zigzag pattern drawn with a single color.",
            "This wave pattern connects diagonally from one column to the next forming V-like shapes.",
            "All non-pattern cells are black (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid is identical to the input except that the region between the arms of each V shape is filled with an inner color.",
            "The inner color is chosen to be different from the wave color.",
            "Only the region bounded by the arms of each V is filled, matching the structure of the zigzag pattern."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        inner_color = random.randint(1, 9)
        taskvars = {'inner_color': inner_color}

        num_train = random.randint(3, 5)
        train_examples = []
        for _ in range(num_train):
            input_grid, wave_color = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid.copy(), {'wave_color': wave_color, 'inner_color': inner_color})
            train_examples.append({'input': input_grid, 'output': output_grid})

        test_input, wave_color = self.create_input(taskvars)
        test_output = self.transform_input(test_input.copy(), {'wave_color': wave_color, 'inner_color': inner_color})
        test_examples = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars, gridvars=None):
        rows = random.randint(4, 6)
        cols = random.randint(8, 14)
        grid = np.zeros((rows, cols), dtype=int)

        inner_color = taskvars['inner_color']
        wave_color_candidates = [c for c in range(1, 10) if c != inner_color]
        wave_color = random.choice(wave_color_candidates)

        row = 0
        direction = 1
        for col in range(cols):
            grid[row, col] = wave_color
            if direction == 1 and row == rows - 1:
                direction = -1
            elif direction == -1 and row == 0:
                direction = 1
            row += direction

        return grid, wave_color

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()
        rows, cols = grid.shape
        wave_color = ['wave_color']
        inner_color = taskvars['inner_color']

        # For each column, find the wave's row and fill above it
        for col in range(cols):
            wave_rows = np.where(grid[:, col] == wave_color)[0]
            if len(wave_rows) > 0:
                wave_row = wave_rows[0]
                for row in range(wave_row):
                    if output_grid[row, col] == 0:
                        output_grid[row, col] = inner_color

        return output_grid

    def create_grids_default(self):
        return self.create_grids()
