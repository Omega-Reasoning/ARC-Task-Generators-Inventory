from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring
from Framework.transformation_library import find_connected_objects
import numpy as np
import random
from typing import Dict, Any, Tuple
class TaskTKboHv7Euxfcxf3HzF8JPBGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (as given):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each input grid contains several 2x2 blocks of different colors (0-9)."
        ]

        # 2) Transformation reasoning chain (as given):
        transformation_reasoning_chain = [
            "Output grids are of size {vars['grid_size2']}x{vars['grid_size2']}.",
            "The output grid condenses each 2x2 block in the input grid into a single cell, preserving the color and relative position of the color."
        ]

        # 3) Initialize the base ARCTaskGenerator:
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain, i.e.,
        a grid of size grid_size x grid_size containing uniform 2x2 color blocks.
        """

        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Fill the grid in 2x2 blocks with random colors from 0-9
        for r in range(0, grid_size, 2):
            for c in range(0, grid_size, 2):
                color = random.randint(0, 9)  # Each 2x2 block can be color 0-9
                grid[r:r+2, c:c+2] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid by condensing each 2x2 block into a single cell,
        preserving the 2x2 block's color.
        """
        grid_size = taskvars['grid_size']
        grid_size2 = taskvars['grid_size2']

        out_grid = np.zeros((grid_size2, grid_size2), dtype=int)
        for i in range(grid_size2):
            for j in range(grid_size2):
                # Top-left corner of each 2x2 block
                out_grid[i, j] = grid[2*i, 2*j]

        return out_grid

    def create_grids(self):
        """
        Creates 3 training examples and 1 test example for an ARC-AGI task.
        The grid size is chosen at random among even numbers between 5 and 10 (inclusive),
        with the transformation always halving the dimension.
        """

        # Randomly pick an even grid size in [6,8,10].
        grid_size = random.choice([6, 8, 10,12,14,16,18,20,22,24,26,28,30])
        grid_size2 = grid_size // 2

        # Store task variables
        taskvars = {
            'grid_size': grid_size,
            'grid_size2': grid_size2,
        }

        # Create 3 training pairs and 1 test pair
        # You could add extra logic to ensure variety if needed.
        train_test_data = self.create_grids_default(nr_train_examples=3,
                                                    nr_test_examples=1,
                                                    taskvars=taskvars)
        return taskvars, train_test_data



