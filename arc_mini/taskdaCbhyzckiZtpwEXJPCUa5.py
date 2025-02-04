from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# Optional (but encouraged) imports from the provided libraries:
from input_library import retry
from transformation_library import find_connected_objects

class TaskdaCbhyzckiZtpwEXJPCUa5Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "All input grids are squares.",
            "Let n be the height and width of an input grid.",
            "Each input grid is entirely empty (0), and its size varies across examples."
        ]
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids are created by copying the input grid and filling n number of diagonals, starting from the main diagonal (top-left to bottom-right) and extending towards the top-right until reaching the end.",
            "Each diagonal is filled with a single color. The main diagonal is always {color('object_color1')}, followed by {color('object_color2')}, {color('object_color3')}, {color('object_color4')}, {color('object_color5')}, {color('object_color6')}, {color('object_color7')}, and {color('object_color8')}.",
            "The number of colors used is determined by the grid size, with diagonals extending towards the top-right until they reach the edge of the grid."
        ]
        # 3) Call superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create a square grid of size n x n, filled entirely with 0.
        n is provided in gridvars['size'].
        """
        n = gridvars['size']
        # Create the n x n grid filled with zeros
        grid = np.zeros((n, n), dtype=int)
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:

        """
        Fill the diagonals of the grid (copied) with different colors.
        Diagonal 0 uses object_color1, diagonal 1 uses object_color2, etc.
        Only as many diagonals as the grid size n are used.
        """
        n = grid.shape[0]
        out_grid = grid.copy()

        # Prepare the list of up to 8 color variables in order
        color_vars = [
            taskvars['object_color1'],
            taskvars['object_color2'],
            taskvars['object_color3'],
            taskvars['object_color4'],
            taskvars['object_color5'],
            taskvars['object_color6'],
            taskvars['object_color7'],
            taskvars['object_color8']
        ]

        # Fill each of the n diagonals with a unique color (from color_vars)
        for i in range(n):
            color = color_vars[i]
            for row in range(n - i):
                col = row + i
                out_grid[row, col] = color

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        1) Select distinct sizes for each train and test grid between 3 and 8.
        2) Pick 8 distinct colors from 1..9 and assign them to object_color1..8.
        3) Create 2 or 3 train grids (random) plus 1 test grid, each with a distinct size.
        4) Return the taskvars dictionary and the resulting train/test data.
        """
        # 1) Randomly choose how many training examples we want: 2 or 3
        nr_train_examples = random.randint(2, 3)
        nr_test_examples = 1

        # 2) Choose distinct sizes for these examples (nr_train_examples + nr_test_examples).
        #    We sample from [3..8] (i.e. range(3,9)) which gives possible sizes 3,4,5,6,7,8
        total_needed = nr_train_examples + nr_test_examples
        possible_sizes = list(range(3, 9))  # [3,4,5,6,7,8]
        chosen_sizes = random.sample(possible_sizes, total_needed)

        # 3) Pick 8 distinct colors from 1..9
        all_colors = list(range(1, 10))  # [1..9]
        chosen_colors = random.sample(all_colors, 8)

        taskvars = {
            'object_color1': chosen_colors[0],
            'object_color2': chosen_colors[1],
            'object_color3': chosen_colors[2],
            'object_color4': chosen_colors[3],
            'object_color5': chosen_colors[4],
            'object_color6': chosen_colors[5],
            'object_color7': chosen_colors[6],
            'object_color8': chosen_colors[7],
        }

        # 4) Build train/test data
        train_test_data: TrainTestData = {
            'train': [],
            'test': []
        }

        # Generate train examples
        for i in range(nr_train_examples):
            sz = chosen_sizes[i]
            input_grid = self.create_input(taskvars, {'size': sz})
            output_grid = self.transform_input(input_grid, taskvars)
            train_test_data['train'].append(
                GridPair(input=input_grid, output=output_grid)
            )

        # Generate test examples
        for i in range(nr_test_examples):
            sz = chosen_sizes[nr_train_examples + i]
            input_grid = self.create_input(taskvars, {'size': sz})
            output_grid = self.transform_input(input_grid, taskvars)
            train_test_data['test'].append(
                GridPair(input=input_grid, output=output_grid)
            )

        return taskvars, train_test_data



