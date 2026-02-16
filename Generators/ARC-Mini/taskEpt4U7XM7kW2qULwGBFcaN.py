# my_arc_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from typing import Dict, Any, Tuple
import numpy as np
import random

# Optional: we may import find_connected_objects, etc. from the transformation library if needed
# from Framework.transformation_library import find_connected_objects, GridObject, GridObjects

class TaskEpt4U7XM7kW2qULwGBFcaNGenerator(ARCTaskGenerator):

    def __init__(self):
        """
        Constructor:
          1) Sets the input reasoning chain (observation_chain).
          2) Sets the transformation reasoning chain.
          3) Calls the parent constructor with these chains.
        """
        # 1) Input (observation) reasoning chain
        observation_chain = [
            "Input grids are of size 4x4.",
            "Each input grid contains four 2x2 objects, 4-way connected cells of the same color, which are placed in the four corners of the grid.",
            "Each 2x2 object is of a different color. The colors are {color('object_color1')}, {color('object_color2')}, {color('object_color3')} and {color('object_color4')}."
        ]

        # 2) Transformation reasoning chain
        transformation_chain = [
            "To construct the output grid, copy the input grid and modify the colors of the objects according to the following rule.",
            "If the 2x2 object is of {color('object_color1')} or {color('object_color2')} color, change its color to {color('object_color5')}.",
            "Otherwise keep the object as it is."
        ]

        # 3) Call the superclass constructor
        super().__init__(observation_chain, transformation_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Initialise task variables used in templates and create train/test data grids.
        
        We:
         1) Select 5 distinct colors out of {1..9}.
         2) Create two training examples and one test example.
         3) Each example is a 4x4 grid with 4 corner blocks (each 2x2) 
            in a random color assignment.
        """
        # Step 1: Pick 5 distinct colors
        distinct_colors = random.sample(range(1, 10), 5)
        taskvars = {
            "object_color1": distinct_colors[0],
            "object_color2": distinct_colors[1],
            "object_color3": distinct_colors[2],
            "object_color4": distinct_colors[3],
            "object_color5": distinct_colors[4],
        }

        # Create two training pairs, one test pair by default
        # We do not need separate gridvars; corner color assignment is handled in create_input()
        # using random permutations each time. So we can use create_grids_default() with 2 train, 1 test.
        train_test_data = self.create_grids_default(nr_train_examples=2, nr_test_examples=1, taskvars=taskvars)
        return taskvars, train_test_data

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input 4x4 grid with four distinct 2x2 color blocks in the corners.
        Each block uses one of the colors among object_color1..4. Which color goes to which corner
        is randomized so we get diverse examples.
        """
        # Initialize a 4x4 grid of zeros (empty).
        grid = np.zeros((4, 4), dtype=int)

        # We read the four "object colors" from taskvars
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]
        color3 = taskvars["object_color3"]
        color4 = taskvars["object_color4"]

        # Create a random permutation of these four colors
        color_permutation = [color1, color2, color3, color4]
        random.shuffle(color_permutation)

        # Coordinates for the four corners (each is a 2x2 region)
        corner_coords = [
            [(0,0), (0,1), (1,0), (1,1)],  # top-left
            [(0,2), (0,3), (1,2), (1,3)],  # top-right
            [(2,0), (2,1), (3,0), (3,1)],  # bottom-left
            [(2,2), (2,3), (3,2), (3,3)],  # bottom-right
        ]

        # Assign each of the 4 colors in the permutation to one corner
        for block_idx, coords_list in enumerate(corner_coords):
            block_color = color_permutation[block_idx]
            for (r, c) in coords_list:
                grid[r, c] = block_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
          - Copy the grid
          - Any cell of color object_color1 or object_color2 => recolor to object_color5
          - Other cells remain unchanged
        """
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]
        color5 = taskvars["object_color5"]

        # Make a copy
        output_grid = grid.copy()

        # Replace object_color1 or object_color2 with object_color5
        mask = (output_grid == color1) | (output_grid == color2)
        output_grid[mask] = color5

        return output_grid

