# my_arc_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects  # We may or may not need other imports
from Framework.input_library import np, random

class TaskYznSsp9hnPBY5wG3townqvGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input (observation) reasoning chain
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input matrix contains exactly one {color('cell_color1')} cell, with the remaining cells being empty (0)."
        ]
        
        # 2) Transformation reasoning chain
        reasoning_chain = [
            "The output matrix is constructed by copying the input matrix and coloring all empty (0) cells that are adjacent (up, down, left, right) to the {color('cell_color1')} cell with {color('cell_color2')} color."
        ]
        
        # 3) Initialize the superclass
        super().__init__(observation_chain, reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain given the task and grid variables.
        - The grid size is random between 5x5 and 30x30.
        - Exactly one cell has color cell_color1, and it's not on the border of the grid.
        - The rest are empty (0).
        """
        # Randomly choose grid dimensions
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # cell_color1 must not be on the border
        row = random.randint(1, height - 2)
        col = random.randint(1, width - 2)
        
        # Place exactly one cell of color_color1
        cell_color1 = taskvars['cell_color1']
        grid[row, col] = cell_color1
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
          1. Copy the input grid.
          2. Color all empty (0) cells adjacent (up, down, left, right) to the {color('cell_color1')} cell with {color('cell_color2')} color.
        """
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']

        # 1) Copy the grid
        out_grid = grid.copy()

        # 2) Find the coordinates of the single cell_color1 cell
        coords = np.argwhere(out_grid == cell_color1)
        # There should be exactly one, but let's handle any possibility
        for (r, c) in coords:
            # Check neighbors up/down/left/right
            for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < out_grid.shape[0] and 0 <= cc < out_grid.shape[1]:
                    if out_grid[rr, cc] == 0:
                        out_grid[rr, cc] = cell_color2
        
        return out_grid

    def create_grids(self):
        """
        1) Randomly choose valid task variables:
            - cell_color1 != 0
            - cell_color2 != 0
            - cell_color1 != cell_color2
        2) Create 3-6 training examples + 1 test example by default.
        3) Return (vars_dict, TrainTestData).
        """
        # Randomly pick two different colors from 1..9
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)
        
        taskvars = {
            'cell_color1': color1,
            'cell_color2': color2
        }

        # For simplicity, we'll produce between 3 and 6 training examples and 1 test
        num_train = random.randint(3, 6)
        num_test = 1
        
        # Use the helper method from the abstract class to generate train/test data
        data = self.create_grids_default(nr_train_examples=num_train,
                                         nr_test_examples=num_test,
                                         taskvars=taskvars)
        
        return taskvars, data


