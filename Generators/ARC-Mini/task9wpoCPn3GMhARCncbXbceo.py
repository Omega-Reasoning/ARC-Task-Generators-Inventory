from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task9wpoCPn3GMhARCncbXbceokGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each input grid has {int((vars['grid_size'] + 1) / 2)} same-colored (1-9) cells in the last row, with the remaining cells being empty (0).",
            "These {int((vars['grid_size'] + 1) / 2)} cells are evenly spaced in the last row, starting from the first cell."
        ]

        transformation_reasoning_chain = [
            "The output grids are created by copying the input grids and expanding each colored cell diagonally upwards from its top-left and top-right corners, maintaining the same color.",
            "These colored cells extend diagonally until they reach the edges of the grid, overlapping any colored cells if present."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates train and test grids with randomly chosen odd sizes and distinct colors
        while following the input/transformation reasoning chains.
        """
        nr_train_examples = random.randint(3, 4)
        nr_test_examples = 1
        total_examples = nr_train_examples + nr_test_examples

        possible_sizes = [s for s in range(7, 31, 2)]  # Odd sizes between 7 and 30
        chosen_grid_size = random.choice(possible_sizes)

        all_colors = list(range(1, 10))  # possible colors are 1..9
        random.shuffle(all_colors)
        chosen_colors = all_colors[:total_examples]

        train_pairs = []
        test_pairs = []

        for i in range(nr_train_examples):
            color = chosen_colors[i]
            input_grid = self.create_input({'grid_size': chosen_grid_size}, {'color': color})
            output_grid = self.transform_input(input_grid, {'color': color})
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # Test example
        color_test = chosen_colors[nr_train_examples]
        test_input_grid = self.create_input({'grid_size': chosen_grid_size}, {'color': color_test})
        test_output_grid = self.transform_input(test_input_grid, {'color': color_test})
        test_pairs.append(GridPair(input=test_input_grid, output=test_output_grid))

        # Return task variables including grid_size to avoid KeyError
        return {'grid_size': chosen_grid_size}, TrainTestData(train=train_pairs, test=test_pairs)

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid of size grid_size x grid_size with uniformly spaced colored cells in the last row.
        """
        grid_size = taskvars['grid_size']
        color = gridvars['color']
        
        grid = np.zeros((grid_size, grid_size), dtype=int)

        num_cells = (grid_size + 1) // 2
        for i in range(num_cells):
            grid[grid_size - 1, i * 2] = color  # Place color every other cell

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Copy the input grid and expand colored cells diagonally upwards.
        """
        output_grid = grid.copy()
        rows, cols = output_grid.shape

        for c in range(cols):
            color = output_grid[rows - 1, c]
            if color != 0:
                r_up, c_left = rows - 2, c - 1
                while r_up >= 0 and c_left >= 0:
                    output_grid[r_up, c_left] = color
                    r_up -= 1
                    c_left -= 1

                r_up, c_right = rows - 2, c + 1
                while r_up >= 0 and c_right < cols:
                    output_grid[r_up, c_right] = color
                    r_up -= 1
                    c_right += 1

        return output_grid


