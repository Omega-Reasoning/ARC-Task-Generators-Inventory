# filename: block_repeat_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random


class TaskYP3iGGttPxFFL2qWyBArRrGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Define the input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain only four multi-colored (1-9) cells and empty (0) cells.",
            "The four colored cells are positioned in the top-left corner of the grid at (0,0), (0,1), (1,0), and (1,1), forming a multi-colored 2x2 block."
        ]

        # 2) Define the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and duplicating the multi-colored 2x2 block from the top-left corner, repeating it horizontally and vertically until the entire grid is filled."
        ]

        # 3) Call the superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        """
        Create a single input grid according to the instructions:
        - Size (rows x cols) is stored in taskvars.
        - Four distinct colors are stored in gridvars['color_block'].
        - Place them at (0,0), (0,1), (1,0), (1,1). The rest is 0.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']

        # The 2x2 block's colors are passed via gridvars:
        color_block = gridvars['color_block']  # [color1, color2, color3, color4]

        # Create the grid and place the 2x2 colored block
        grid = np.zeros((rows, cols), dtype=int)
        grid[0, 0] = color_block[0]
        grid[0, 1] = color_block[1]
        grid[1, 0] = color_block[2]
        grid[1, 1] = color_block[3]

        return grid

    def transform_input(self, grid, taskvars):
        """
        Transform the input grid according to the transformation reasoning chain:
        - Copy the 2x2 block from top-left corner.
        - Tile it across the entire grid.
        """
        rows, cols = grid.shape
        output = grid.copy()

        # Extract the 2x2 block
        top_left_block = grid[0:2, 0:2]

        # Fill the entire grid by repeating the 2x2 pattern
        for r in range(rows):
            for c in range(cols):
                output[r, c] = top_left_block[r % 2, c % 2]

        return output

    def create_grids(self):
        """
        Create multiple (3-4) training examples and 1 test example.
        Each example:
          - random even rows, random even cols in [5..30].
          - 4 distinct colors from 1..9 in the top-left 2x2 block.
        The same rows & cols are used across all examples in a single task.
        """
        # Choose random even dimensions for the entire task
        rows = random.choice([r for r in range(5, 31) if r % 2 == 0])
        cols = random.choice([c for c in range(5, 31) if c % 2 == 0])

        taskvars = {
            'rows': rows,
            'cols': cols
        }

        # Number of training examples is 3 or 4
        num_train = random.randint(3, 4)
        train_data = []

        # Generate training examples
        for _ in range(num_train):
            # 4 distinct colors from 1..9
            color_block = random.sample(range(1, 10), 4)
            gridvars_example = {'color_block': color_block}

            input_grid = self.create_input(taskvars, gridvars_example)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))

        # Generate the test example
        color_block_test = random.sample(range(1, 10), 4)
        gridvars_test = {'color_block': color_block_test}

        test_input_grid = self.create_input(taskvars, gridvars_test)
        test_output_grid = self.transform_input(test_input_grid, taskvars)
        test_data = [GridPair(input=test_input_grid, output=test_output_grid)]

        # Package the results
        train_test_data: TrainTestData = {
            "train": train_data,
            "test": test_data
        }
        return taskvars, train_test_data



