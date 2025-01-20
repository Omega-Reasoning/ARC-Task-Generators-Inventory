# Filename: arc_agi_task_generator_example.py


import numpy as np
import random

# Required imports from the base ARCTaskGenerator and the libraries:
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects  # You can import others if needed
from input_library import Contiguity, create_object, retry

class TaskmhTE37F5cVpMjzQX8f75nHGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "The input grids contain a completely filled border of one color (1-9) along with a single {color('cell_color1')} or {color('cell_color2')} cell in the interior.",
            "The remaining interior cells are empty (0)."
        ]

        # 2) The transformation reasoning chain
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid, and fill in the empty (0) interior cells according to the color of the filled interior cell.",
            "If the grid contains {color('cell_color1')} interior cell, fill all empty (0) cells with {color('cell_color3')} color.",
            "If the grid contains {color('cell_color2')} interior cell, fill all the empty (0) cells with {color('cell_color4')} color."
        ]

        # 3) Call the superclass constructor with the given reasoning chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        1) Randomly choose distinct colors for cell_color1, cell_color2, cell_color3, cell_color4, and
           optionally a fifth color for the border (also distinct).
        2) Create a number of training grids (3-6) and 2 test grids.
           Make sure:
             - At least one training grid has cell_color1 in its interior,
             - At least one training grid has cell_color2 in its interior,
             - One test grid has cell_color1,
             - Another test grid has cell_color2.
        3) Return the (taskvars, TrainTestData).
        """
        # 1) Pick 5 distinct colors in [1..9]
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        cell_color1 = all_colors[0]
        cell_color2 = all_colors[1]
        cell_color3 = all_colors[2]
        cell_color4 = all_colors[3]
        border_color = all_colors[4]

        # Store them in a dictionary so the template can resolve color('cell_colorX')
        taskvars = {
            'cell_color1': cell_color1,
            'cell_color2': cell_color2,
            'cell_color3': cell_color3,
            'cell_color4': cell_color4,
            'border_color': border_color,
        }

        # Decide how many training examples we want: 3 to 6
        num_train = random.randint(3, 6)

        # We need to ensure at least one training grid with interior cell_color1
        # and at least one training grid with interior cell_color2.
        # We'll explicitly construct at least two training examples for these scenarios.
        train_data = []

        # Force one training example with cell_color1
        input_grid_1 = self.create_input(taskvars, {'interior_color': cell_color1})
        output_grid_1 = self.transform_input(input_grid_1, taskvars)
        train_data.append({'input': input_grid_1, 'output': output_grid_1})

        # Force one training example with cell_color2
        input_grid_2 = self.create_input(taskvars, {'interior_color': cell_color2})
        output_grid_2 = self.transform_input(input_grid_2, taskvars)
        train_data.append({'input': input_grid_2, 'output': output_grid_2})

        # Create the remaining training examples
        for _ in range(num_train - 2):
            # Randomly pick interior color among {cell_color1, cell_color2}
            chosen_interior_color = random.choice([cell_color1, cell_color2])
            in_grid = self.create_input(taskvars, {'interior_color': chosen_interior_color})
            out_grid = self.transform_input(in_grid, taskvars)
            train_data.append({'input': in_grid, 'output': out_grid})

        # For the test data, we want 2 test grids:
        #  - one with cell_color1
        #  - one with cell_color2
        test_data = []
        test_input_1 = self.create_input(taskvars, {'interior_color': cell_color1})
        test_output_1 = self.transform_input(test_input_1, taskvars)
        test_data.append({'input': test_input_1, 'output': test_output_1})

        test_input_2 = self.create_input(taskvars, {'interior_color': cell_color2})
        test_output_2 = self.transform_input(test_input_2, taskvars)
        test_data.append({'input': test_input_2, 'output': test_output_2})

        train_test_data = {
            'train': train_data,
            'test': test_data
        }
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid following the input reasoning chain:
          - Grid size between 5x5 and 30x30 (chosen randomly).
          - Entire border is filled with `border_color`.
          - Exactly one interior cell with `gridvars['interior_color']`, the rest of the interior is 0.
        """
        border_color = taskvars['border_color']
        interior_color = gridvars['interior_color']

        # Randomly choose grid dimensions between 5 and 30
        rows = random.randint(5, 10)      # smaller upper bound for demonstration
        cols = random.randint(5, 10)

        grid = np.zeros((rows, cols), dtype=int)

        # Fill the border
        grid[0, :] = border_color
        grid[-1, :] = border_color
        grid[:, 0] = border_color
        grid[:, -1] = border_color

        # If there is enough interior space, place the interior cell
        # Choose a random interior row and column
        if rows > 2 and cols > 2:
            int_row = random.randint(1, rows - 2)
            int_col = random.randint(1, cols - 2)
            grid[int_row, int_col] = interior_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Implement the transformation reasoning chain:
          - Copy the input grid
          - If the interior cell is cell_color1, fill all 0 cells with cell_color3
          - If the interior cell is cell_color2, fill all 0 cells with cell_color4
        """
        new_grid = np.copy(grid)

        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']
        cell_color3 = taskvars['cell_color3']
        cell_color4 = taskvars['cell_color4']

        # Determine which interior color is present (assuming exactly one interior cell is non-zero
        # and different from the border; if multiple, the first found interior color decides).
        # We'll simply scan for cell_color1 or cell_color2 in the interior:
        found_color = None
        rows, cols = new_grid.shape
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                val = new_grid[r, c]
                if val == cell_color1:
                    found_color = cell_color1
                    break
                elif val == cell_color2:
                    found_color = cell_color2
                    break
            if found_color is not None:
                break

        if found_color == cell_color1:
            fill_color = cell_color3
        else:
            fill_color = cell_color4

        # Fill all 0 cells with fill_color
        # The instructions say "fill the empty (0) interior cells," but typically
        # that means all 0 cells in the entire grid. Because the border is
        # not zero, it won't be affected anyway.
        new_grid[new_grid == 0] = fill_color

        return new_grid


