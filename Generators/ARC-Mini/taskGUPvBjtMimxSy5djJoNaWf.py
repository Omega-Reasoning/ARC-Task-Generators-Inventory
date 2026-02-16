

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random


class TaskGUPvBjtMimxSy5djJoNaWfGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain three {color('cell_color')} diagonally connected cells and empty cells.",
            "The three connected {color('cell_color')} cells must follow a single diagonal direction within each example, but their positions and diagonal direction can vary across examples."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling the three columns that contain {color('cell_color')} cells entirely with {color('cell_color')} color."
        ]
        
        # 3) Call parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create a grid of size (rows, cols) with exactly three diagonally connected cells 
        of color cell_color, all other cells empty (0).
        The diagonal may be '\' or '/' direction, chosen randomly.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        cell_color = taskvars['cell_color']

        # Start with all empty
        grid = np.zeros((rows, cols), dtype=int)

        # Decide which diagonal direction to use
        direction = random.choice(["\\", "/"])
        
        if direction == "\\":
            # We place cells at (r, c), (r+1, c+1), (r+2, c+2)
            # so we need r+2 < rows and c+2 < cols
            r = random.randint(0, rows - 3)
            c = random.randint(0, cols - 3)
            for i in range(3):
                grid[r + i, c + i] = cell_color
        else:
            # We place cells at (r, c), (r+1, c-1), (r+2, c-2)
            # so we need r+2 < rows and c-2 >= 0
            r = random.randint(0, rows - 3)
            c = random.randint(2, cols - 1)
            for i in range(3):
                grid[r + i, c - i] = cell_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        The transformation: 
        1) Copy the input grid.
        2) Find all columns where at least one cell has color == cell_color.
        3) Fill those columns entirely with cell_color.
        4) Return the result.
        """
        cell_color = taskvars['cell_color']
        out_grid = grid.copy()
        rows, cols = out_grid.shape
        
        # Identify which columns have the target color
        columns_to_fill = set()
        for r in range(rows):
            for c in range(cols):
                if out_grid[r, c] == cell_color:
                    columns_to_fill.add(c)
        
        # Fill these columns
        for col_idx in columns_to_fill:
            out_grid[:, col_idx] = cell_color
        
        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Creates the task variables and train/test data.
         - rows, cols randomly in [5..10]
         - cell_color randomly in [1..9]
         - 3 to 5 train examples, 1 test example
         - Each example must have exactly three diagonally connected colored cells
        """
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        cell_color = random.randint(1, 9)
        
        # Put these in taskvars (used for template expansions)
        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color': cell_color
        }
        
        nr_train = random.randint(3, 5)
        nr_test = 1
        
        train_examples = []
        for _ in range(nr_train):
            in_grid = self.create_input(taskvars, {})
            out_grid = self.transform_input(in_grid, taskvars)
            train_examples.append(GridPair(input=in_grid, output=out_grid))
        
        test_examples = []
        for _ in range(nr_test):
            in_grid = self.create_input(taskvars, {})
            out_grid = self.transform_input(in_grid, taskvars)
            test_examples.append(GridPair(input=in_grid, output=out_grid))
        
        return taskvars, TrainTestData(train=train_examples, test=test_examples)

