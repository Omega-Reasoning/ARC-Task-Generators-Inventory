from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class TaskfZZdqLV3JHHUUxBZ7dmJtHGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialize the input reasoning chain
        input_reasoning_chain = [
            "All input grids are squares.",
            "Let n be the height and width of an input grid, where n is an even number.",
            "They contain a completely filled main diagonal (top-left to bottom-right) with either {color('cell_color1')} or {color('cell_color2')} cells, and the remaining cells are empty (0)."
        ]

        # 2) Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and completely filling the inverse diagonal (top-right to bottom-left) with a different color.",
            "The inverse diagonal is filled with {color('cell_color3')} if the main diagonal is {color('cell_color1')}, otherwise, it is filled with {color('cell_color4')}."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
         - Square grid of even dimension n
         - All cells = 0 except main diagonal = color1 or color2
        Uses gridvars to specify the chosen size (n) and which color is used for the diagonal.
        """
        n = gridvars['size']
        diag_color = gridvars['diag_color']  # either taskvars['cell_color1'] or taskvars['cell_color2']

        # Create an n x n array of zeros
        grid = np.zeros((n, n), dtype=int)

        # Fill main diagonal
        for i in range(n):
            grid[i, i] = diag_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
         - Copy the input grid
         - If the main diagonal color is cell_color1, fill the inverse diagonal with cell_color3
           otherwise fill it with cell_color4.
        """
        # Copy the grid
        out_grid = grid.copy()

        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']
        color3 = taskvars['cell_color3']
        color4 = taskvars['cell_color4']

        n = out_grid.shape[0]
        # Check which color is on the main diagonal (assume the diagonal is consistently color1 or color2)
        diagonal_color = out_grid[0, 0]  # top-left corner is the main diag color

        if diagonal_color == color1:
            fill_color = color3
        else:
            fill_color = color4

        # Fill the inverse diagonal
        for i in range(n):
            out_grid[i, n - 1 - i] = fill_color

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        We need to:
         - Pick distinct colors cell_color1..4 in [1..9], all different.
         - Generate 3-4 training examples (we choose 4) and 2 test examples.
         - Each input grid is square, even dimension, distinct sizes.
         - We want at least one training and one test example with main diagonal = color1,
           and at least one with main diagonal = color2.
         - The transformation logic: fill inverse diagonal with cell_color3 or cell_color4 
           depending on the main diagonal color.
        """
        # 1) Randomly choose 4 distinct colors from 1..9
        colors = random.sample(range(1, 10), 4)
        taskvars = {
            'cell_color1': colors[0],
            'cell_color2': colors[1],
            'cell_color3': colors[2],
            'cell_color4': colors[3]
        }

        # 2) Pick 6 distinct even grid sizes from [4..30].
        #    We want 4 training examples + 2 test examples => 6 distinct sizes.
        #    (We exclude 2 because the instructions say the grid dimension >= 3, plus must be even.)
        possible_even_sizes = [n for n in range(4, 31) if n % 2 == 0]
        chosen_sizes = random.sample(possible_even_sizes, 6)

        # We'll produce 4 training examples:
        #   2 with main diagonal = color1, 2 with main diagonal = color2
        # We'll produce 2 test examples:
        #   1 with main diagonal = color1, 1 with main diagonal = color2
        # This satisfies the constraints that training and test each contain both color1 and color2 diagonal examples.

        train_pairs = []
        # First 2 training grids: main diagonal = color1
        for i in range(2):
            gridvars = {
                'size': chosen_sizes[i],
                'diag_color': taskvars['cell_color1']
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})

        # Next 2 training grids: main diagonal = color2
        for i in range(2, 4):
            gridvars = {
                'size': chosen_sizes[i],
                'diag_color': taskvars['cell_color2']
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})

        # Now 2 test grids
        test_pairs = []
        # 1 with color1 diagonal
        gridvars_test1 = {
            'size': chosen_sizes[4],
            'diag_color': taskvars['cell_color1']
        }
        input_test1 = self.create_input(taskvars, gridvars_test1)
        output_test1 = self.transform_input(input_test1, taskvars)
        test_pairs.append({'input': input_test1, 'output': output_test1})

        # 1 with color2 diagonal
        gridvars_test2 = {
            'size': chosen_sizes[5],
            'diag_color': taskvars['cell_color2']
        }
        input_test2 = self.create_input(taskvars, gridvars_test2)
        output_test2 = self.transform_input(input_test2, taskvars)
        test_pairs.append({'input': input_test2, 'output': output_test2})

        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }

        return taskvars, train_test_data

