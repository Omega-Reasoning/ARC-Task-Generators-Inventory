from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import find_connected_objects
import numpy as np
import random

class Task178fcbfbGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "Each grid contains several cells colored with one of three specific colors {color('color_1')}, {color('color_2')}, and {color('color_3')}, randomly placed, with the remaining cells being empty (0).",
            "No two cells of {color('color_1')} color are in the same column.",
            "No two cells of {color('color_2')} and {color('color_3')} colors are in the same row."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid." ,
            "For each column in the input grid that contains a cell of {color('color_1')} color, fill the entire column in the output grid with color {color('color_1')}.",
            "For each row in the input grid that contains a cell of {color('color_2')} or {color('color_3')} color , are filled with respective colors {color('color_2')} and {color('color_3')} in the output grid.",
            "The {color('color_2')} and {color('color_3')} colors in the output grid override any previous color assignments from {color('color_1')}."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        color_3 = taskvars['color_3']

        grid = np.zeros((rows, cols), dtype=int)

        def place_cells():
            nonlocal grid
            # First, ensure one cell of each color is placed
            for color in [color_1, color_2, color_3]:
                while True:
                    row = random.randint(0, rows - 1)
                    col = random.randint(0, cols - 1)
                    if grid[row, col] == 0:
                        # For color_1, check if the column is empty
                        if color == color_1 and not any(grid[:, col] == color_1):
                            grid[row, col] = color
                            break
                        # For color_2 and color_3, check if the row is empty of both colors
                        elif color == color_2 and not any(grid[row] == color_2) and not any(grid[row] == color_3):
                            grid[row, col] = color
                            break
                        elif color == color_3 and not any(grid[row] == color_3) and not any(grid[row] == color_2):
                            grid[row, col] = color
                            break

            # Then place remaining cells randomly
            for _ in range(gridvars['num_colored_cells'] - 3):
                while True:
                    row = random.randint(0, rows - 1)
                    col = random.randint(0, cols - 1)
                    color = random.choice([color_1, color_2, color_3])
                    if grid[row, col] == 0:
                        # For color_1, check if the column is empty
                        if color == color_1 and not any(grid[:, col] == color_1):
                            grid[row, col] = color
                            break
                        # For color_2 and color_3, check if the row is empty of both colors
                        elif color == color_2 and not any(grid[row] == color_2) and not any(grid[row] == color_3):
                            grid[row, col] = color
                            break
                        elif color == color_3 and not any(grid[row] == color_3) and not any(grid[row] == color_2):
                            grid[row, col] = color
                            break

        place_cells()
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output = grid.copy()
        rows, cols = grid.shape

        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        color_3 = taskvars['color_3']

        # Fill columns with color_1
        for col in range(cols):
            if color_1 in grid[:, col]:
                output[:, col] = color_1

        # Fill rows with color_2
        for row in range(rows):
            if color_2 in grid[row, :]:
                output[row, :] = color_2

        # Fill rows with color_3
        for row in range(rows):
            if color_3 in grid[row, :]:
                output[row, :] = color_3

        return output

    def create_grids(self) -> tuple:
        rows = random.randint(8, 30)
        cols = random.randint(8, 30)
        color_1, color_2, color_3 = random.sample(range(1, 10), 3)

        taskvars = {
            'rows': rows,
            'cols': cols,
            'color_1': color_1,
            'color_2': color_2,
            'color_3': color_3,
        }

        train_examples = []
        for _ in range(random.randint(3, 4)):
            gridvars = {}
            gridvars['num_colored_cells'] = random.randint(3, 10)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        gridvars = {}
        gridvars['num_colored_cells'] = random.randint(3, 10)
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)

        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }

        return taskvars, train_test_data

