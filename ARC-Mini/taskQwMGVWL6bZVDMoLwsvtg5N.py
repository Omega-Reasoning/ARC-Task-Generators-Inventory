# my_arc_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# Imports from the provided libraries
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects

class TaskQwMGVWL6bZVDMoLwsvtg5NGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) The "input reasoning chain" describing how the input is generated.
        observation_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each input grid contains a single object, 4-way connected cells of the same color (1-9) with the remaining cells being empty (0)."
        ]
        # 2) The "transformation reasoning chain" describing how to get from input to output.
        reasoning_chain = [
            "The output grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "The output grid has a single-colored cell at the exact center of the grid.",
            "The color of this cell matches the color of the object in the input grid."
        ]
        # 3) Call superclass constructor with the chain definitions.
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        We generate task variables (currently only 'grid_size') 
        and create training & test data using create_grids_default().
        """
        # Pick a random odd grid size between 4 and 20
        # (as per instructions, it must be an odd number).
        possible_sizes = [s for s in range(5, 31, 2)]  # 5,7,9,11,13,15,17,19
        grid_size = random.choice(possible_sizes)

        # Store the task variables in a dictionary
        taskvars = {
            'grid_size': grid_size
        }
        # We create 4 train examples and 1 test example.
        # create_grids_default() will call create_input() and transform_input()
        # for each example. 
        train_test_data = self.create_grids_default(nr_train_examples=4,
                                                    nr_test_examples=1,
                                                    taskvars=taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create a single input grid with:
          1) Size: taskvars['grid_size'] x taskvars['grid_size']
          2) One single-color, 4-way connected object of color ∈ [1..9],
             placed at a random position, random shape.
        """
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Random color for the object
        color = random.randint(1, 9)

        # Randomly pick object dimensions (at least 1x1, up to grid_size x grid_size)
        # Typically smaller objects add variety.
        obj_height = random.randint(1, grid_size)
        obj_width = random.randint(1, grid_size)

        # Create the object matrix using the input_library
        obj = create_object(
            height=obj_height,
            width=obj_width,
            color_palette=color,
            contiguity=Contiguity.FOUR,
            background=0
        )

        # Choose a random top-left placement so the entire object fits
        row_offset = random.randint(0, grid_size - obj_height)
        col_offset = random.randint(0, grid_size - obj_width)

        # Paste the generated object into the grid
        for r in range(obj_height):
            for c in range(obj_width):
                if obj[r, c] != 0:
                    grid[row_offset + r, col_offset + c] = obj[r, c]

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transformation: 
        1) Detect the single object's color (since it's uniformly colored, 
           we can just take max(grid) or find non-zero color).
        2) Create an output grid of the same shape with 0s.
        3) Place a single cell at the center with the object color.
        """
        # Extract the object's color 
        # (assuming exactly one object and it's uniform, 
        #  we can do either max(...) or unique(...) to find it).
        nonzero_vals = grid[grid != 0]
        color = int(nonzero_vals[0]) if len(nonzero_vals) > 0 else 0

        # Create empty output grid
        out_grid = np.zeros_like(grid)

        # Place a single cell of 'color' at the center
        n = grid.shape[0]
        center = n // 2  # integer division
        out_grid[center, center] = color

        return out_grid


