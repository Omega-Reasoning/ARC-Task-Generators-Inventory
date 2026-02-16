# my_arc_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, Contiguity
from Framework.transformation_library import find_connected_objects
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskSkgRdJoKNLdJ47AFPnDVhMGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains a single colored (1-9) object, consisting of 4-way connected cells of the same color.",
            "All other cells are empty (0)."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by initializing a zero-filled grid.",
            "It has a checkerboard pattern with alternating colored and empty (0) cells, starting with the first cell always being colored.",
            "The color used for the checkerboard pattern is the color of the object in the input grid."
        ]
        
        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
         1) Different size (between 5 and 30 rows/columns).
         2) Contains a single 4-way connected object of a single color (1-9).
         3) All other cells are empty (0).
        
        The color, rows, and cols are passed via gridvars.
        """
        color = gridvars['color']
        rows = gridvars['rows']
        cols = gridvars['cols']
        
        # Initialize an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Create a small random object using input_library's create_object
        # with a single color in its palette and 4-way contiguity
        obj_height = random.randint(3, rows)  # object can be up to the grid size
        obj_width = random.randint(2, cols)
        object_matrix = create_object(height=obj_height,
                                      width=obj_width,
                                      color_palette=color,
                                      contiguity=Contiguity.FOUR,
                                      background=0)
        
        # Randomly place this object somewhere in the grid
        max_row_offset = rows - object_matrix.shape[0]
        max_col_offset = cols - object_matrix.shape[1]
        row_offset = random.randint(0, max_row_offset)
        col_offset = random.randint(0, max_col_offset)
        
        # Paste the object into the grid
        for r in range(object_matrix.shape[0]):
            for c in range(object_matrix.shape[1]):
                if object_matrix[r, c] != 0:
                    grid[row_offset + r, col_offset + c] = object_matrix[r, c]
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
         1) Create a zero-filled grid of the same shape.
         2) Fill it in a checkerboard pattern with the input object's color,
            starting with the top-left cell as colored.
         3) Use the color of the single object found in the input.
        """
        # 1) Find the single object and retrieve its color
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        # By design, there's only one object. Get its color:
        if len(objects) == 0:
            # Fallback if somehow no object is found; just return the same grid
            return grid.copy()
        single_obj = objects[0]
        # There's only one color because it's monochromatic
        (obj_color,) = single_obj.colors
        
        # 2) Create a new zero-filled grid of the same shape
        out_grid = np.zeros_like(grid)
        
        # 3) Checkerboard fill using obj_color
        #    We want (row + col) even -> colored cell
        rows, cols = out_grid.shape
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    out_grid[r, c] = obj_color
        
        return out_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        1. Pick how many training examples to create (3-4).
        2. Ensure each input grid has a unique size and color across train + test.
        3. Generate the train and test pairs.
        4. Return (variables_dict, TrainTestData).
        """
        # We do not actually have any templated variable placeholders in the 
        # reasoning chain, so an empty dict is sufficient for taskvars.
        task_variables = {}
        
        # Decide on number of train examples
        nr_train = random.choice([3, 4])
        nr_test = 1  # We always create a single test pair

        # Distinct colors and sizes for each example
        # We need (nr_train + nr_test) distinct colors and sizes
        total_examples = nr_train + nr_test
        
        # Choose distinct colors from 1..9
        possible_colors = list(range(1, 10))
        random.shuffle(possible_colors)
        chosen_colors = possible_colors[:total_examples]
        
        # We'll pick distinct (rows, cols) pairs from [5..30]
        all_sizes = []
        for r in range(5, 31):
            for c in range(5, 31):
                all_sizes.append((r, c))
        random.shuffle(all_sizes)
        chosen_sizes = all_sizes[:total_examples]

        # Build the train/test data
        train_pairs = []
        for i in range(nr_train):
            color_i = chosen_colors[i]
            (rows_i, cols_i) = chosen_sizes[i]
            # Create the input
            input_grid = self.create_input(task_variables, {
                'color': color_i,
                'rows': rows_i,
                'cols': cols_i
            })
            # Transform it
            output_grid = self.transform_input(input_grid, task_variables)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Test pair
        color_test = chosen_colors[-1]
        (rows_test, cols_test) = chosen_sizes[-1]
        test_input_grid = self.create_input(task_variables, {
            'color': color_test,
            'rows': rows_test,
            'cols': cols_test
        })
        test_output_grid = self.transform_input(test_input_grid, task_variables)
        
        test_pairs = [GridPair(input=test_input_grid, output=test_output_grid)]
        
        train_test_data = TrainTestData(train=train_pairs, test=test_pairs)
        
        return task_variables, train_test_data



