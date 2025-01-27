from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
import numpy as np
import random

class ARCTask10fcaaa3Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}",
            "The input grid has some cells filled with color input_color(between 1-9) and the remaining cells are empty(0)."
        ]

        transformation_reasoning_chain = [
            "The dimension of the output grid is 2 times the input grid along both rows and columns.",
            "The input grid is copied to all the four quadrants in the output grid.",
            "If there is a cell in output grid with color input_color at position (i,j) the cells at diagonal positions (i-1,j-1), (i+1,j-1),(i-1,j+1),(i+1,j+1) are filled with {color('output_color')} only if these diagonal elements are already not filled with input_color.",
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        input_color = gridvars['input_color']
        max_cells = int(0.15 * rows * cols)
        min_cells = max(1, int(0.05 * rows * cols))

        def grid_generator():
            grid = np.zeros((rows, cols), dtype=int)
            num_cells = random.randint(min_cells, max_cells)
            positions = random.sample([(i, j) for i in range(rows) for j in range(cols)], num_cells)
            for i, j in positions:
                grid[i, j] = input_color
            return grid

        def valid_grid(grid):
            colored_cells = np.count_nonzero(grid == input_color)
            return min_cells <= colored_cells <= max_cells

        return retry(grid_generator, valid_grid)

    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        input_color = next(color for color in set(grid.flatten()) if color != 0)
        output_color = taskvars['output_color']
        output_grid = np.zeros((2 * rows, 2 * cols), dtype=int)

        for i in range(2):
            for j in range(2):
                output_grid[i*rows:(i+1)*rows, j*cols:(j+1)*cols] = grid

        for r in range(output_grid.shape[0]):
            for c in range(output_grid.shape[1]):
                if output_grid[r, c] == input_color:
                    for dr, dc in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < output_grid.shape[0] and 0 <= nc < output_grid.shape[1]:
                            if output_grid[nr, nc] == 0:
                                output_grid[nr, nc] = output_color

        return output_grid

    def create_grids(self):
        rows = random.randint(5, 8)
        cols = random.randint(5, 8)
        output_color = random.randint(1, 9)
        
        taskvars = {
            'rows': rows,
            'cols': cols,
            'output_color': output_color
        }

        train_pairs = []
        num_train = random.randint(3, 4)

        # Keep track of used input colors to avoid repetition
        used_input_colors = set()
        # Create list of available colors (excluding output_color)
        available_colors = list(set(range(1, 10)) - {output_color})

        for _ in range(num_train):
            # Select from remaining available colors
            remaining_colors = list(set(available_colors) - used_input_colors)
            input_color = random.choice(remaining_colors)
            used_input_colors.add(input_color)

            gridvars = taskvars.copy()
            gridvars['input_color'] = input_color
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # For test pair, generate another unique input color
        remaining_colors = list(set(available_colors) - used_input_colors)
        input_color = random.choice(remaining_colors)

        gridvars = taskvars.copy()
        gridvars['input_color'] = input_color
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)

        return taskvars, {
            'train': train_pairs,
            'test': [GridPair(input=test_input, output=test_output)]
        }

