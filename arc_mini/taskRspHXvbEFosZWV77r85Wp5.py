from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple
# We can import from input_library for the create_input() method:
from input_library import retry
# We could also import from transformation_library if we want to use the helpers for find_connected_objects, etc.
from transformation_library import find_connected_objects

class TaskRspHXvbEFosZWV77r85Wp5Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (copy exactly from input)
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid contains exactly one {color('object_color')} object, 4-way connected cells of the same color, which is located in the top half of the grid.",
            "The {color('object_color')} object has a vertical length of {vars['object_length']} cells.",
            "All other cells are empty."
        ]

        # 2) Transformation reasoning chain (copy exactly from input)
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and reflecting the {color('object_color')} object into the bottom half of the grid, placing the reflected object directly below and connected to the original object."
        ]

        # 3) Call superclass __init__ with the taskvars_definitions as the third argument
        #    Since the prompt mentions "3 statements", we'll pass an empty dictionary
        #    (or you can pass a docstring with instructions). For clarity, we just use {}:
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        color = taskvars['object_color']
        object_length = taskvars['object_length']

        # Start with an all-empty grid
        grid = np.zeros((rows, cols), dtype=int)

        def generate_object():
            # Work on a copy for each attempt
            attempt_grid = np.zeros((rows, cols), dtype=int)

            # Choose a random width for the bounding region in [2, up to cols]
            w = random.randint(2, min(cols, 10))  # Limit max width to prevent too many carve attempts
            # Random horizontal offset
            left_col = random.randint(0, cols - w)

            # Fill top bounding rectangle rows [0..object_length-1], cols [left_col..left_col+w-1]
            for r in range(object_length):
                for c in range(left_col, left_col + w):
                    attempt_grid[r, c] = color

            # Carve out random holes but ensure connectivity
            # First, create a copy to test connectivity after each carve
            working_grid = attempt_grid.copy()
            
            # Try to carve each cell with some probability
            carve_prob = 0.3  # ~30% chance to try carving
            for r in range(object_length):
                for c in range(left_col, left_col + w):
                    if working_grid[r, c] == color and random.random() < carve_prob:
                        # Temporarily remove this cell
                        working_grid[r, c] = 0
                        
                        # Check if object remains connected and spans the required height
                        objects = find_connected_objects(working_grid, 
                                                        diagonal_connectivity=False,
                                                        background=0, 
                                                        monochromatic=True)
                        
                        if len(objects) == 1 and objects[0].height == object_length:
                            # Carving is valid, keep it
                            attempt_grid[r, c] = 0
                        else:
                            # Revert the carve
                            working_grid[r, c] = color

            return attempt_grid

        def is_valid(grid_candidate: np.ndarray) -> bool:
            # Find connected objects
            objects = find_connected_objects(grid_candidate,
                                            diagonal_connectivity=False,
                                            background=0,
                                            monochromatic=True)

            # We want exactly 1 non-background object
            if len(objects) != 1:
                return False

            obj = objects[0]
            
            # Check it has cells in the top rows
            if obj.height != object_length:
                return False
                
            # Ensure the object starts at the top row and spans exactly object_length rows
            row_slice, _ = obj.bounding_box
            if row_slice.start != 0 or row_slice.stop != object_length:
                return False

            return True

        # Attempt to generate a valid object
        object_grid = retry(generator=generate_object,
                            predicate=is_valid,
                            max_attempts=100)
        return object_grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        1) Copy the input grid
        2) Reflect the single object from the top half into the bottom half
           by flipping row -> (rows - 1 - row).
        """
        rows = taskvars['rows']
        # Create a copy
        out_grid = grid.copy()

        # Identify the (single) object in the top half
        objects = find_connected_objects(grid,
                                         diagonal_connectivity=False,
                                         background=0,
                                         monochromatic=True)
        if len(objects) != 1:
            # Edge case: if something unexpected, just return the grid copy
            return out_grid

        the_obj = objects[0]
        # Reflect each cell (r, c, color) into bottom half:
        for (r, c, color) in the_obj.cells:
            reflected_r = rows - 1 - r
            out_grid[reflected_r, c] = color

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        1) Randomly instantiate the task variables:
           - rows (even, between 5 and 30)
           - cols (between 5 and 30)
           - object_length = rows // 2
           - object_color in [1..9]
        2) Create 3-6 train examples + 1 test example, ensuring we vary the shape across calls to create_input().
        3) Return (taskvars, train_test_data).
        """

        # 1) Instantiate the 4 variables:
        # rows must be even and in [5..30]. We'll pick a random even number >=6 for more interesting shape
        rows_candidates = [r for r in range(6, 31) if (r % 2 == 0)]
        rows = random.choice(rows_candidates)
        cols = random.randint(5, 30)
        object_length = rows // 2
        object_color = random.randint(1, 9)

        taskvars = {
            'rows': rows,
            'cols': cols,
            'object_length': object_length,
            'object_color': object_color
        }

        # 2) Decide how many train examples in [3..6]
        nr_train = random.randint(3, 6)
        # We'll always have 1 test example
        nr_test = 1

        # For variety in shapes, we rely on create_input() to randomize the carving each time.
        # We can use the built-in create_grids_default or manually do it. We'll do it manually
        # so you can see exactly what's happening:

        train_pairs = []
        for _ in range(nr_train):
            inp = self.create_input(taskvars, {})
            outp = self.transform_input(inp, taskvars)
            train_pairs.append({'input': inp, 'output': outp})

        test_pairs = []
        for _ in range(nr_test):
            inp = self.create_input(taskvars, {})
            outp = self.transform_input(inp, taskvars)
            test_pairs.append({'input': inp, 'output': outp})

        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }

        return taskvars, train_test_data


