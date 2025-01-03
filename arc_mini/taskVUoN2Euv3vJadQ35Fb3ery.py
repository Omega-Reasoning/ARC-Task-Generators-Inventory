from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, BorderBehavior, CollisionBehavior
from input_library import create_object, retry, Contiguity
import numpy as np
import random

class TasktaskVUoN2Euv3vJadQ35Fb3eryGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        observation_chain = [
            "Input grids are square of size {vars['grid_size']} x {vars['grid_size']}.",
            "In each input grid there is exactly one object, 4-way connected cells of the same color, with the remaining cells being empty (0).",
            "The color of the object is {color('object_color_1')}."
        ]
        
        # 2) Transformation reasoning chain
        reasoning_chain = [
            "The output grid is constructed by copying the input grid and changing the color of the object from {color('object_color_1')} to {color('object_color_2')}.",
            "All empty (0) cells are filled according to the number of rows/columns in the grid.",
            "If the number is even, empty cells are filled with {color('even_fill_color')}.",
            "If the number is odd, empty cells are filled with {color('odd_fill_color')}."
        ]
        
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        1) Randomly choose the task variables:
           - grid_size (between 5 and 30)
           - object_color_1, object_color_2, odd_fill_color, even_fill_color
             (between 1..9, with constraints)
        2) Generate train/test data by calling self.create_grids_default(...)
        """
        # Possible colors are 1..9
        possible_colors = list(range(1, 10))
        
        # Choose a random grid size
        grid_size = random.randint(5, 30)
        
        # Choose object_color_1
        object_color_1 = random.choice(possible_colors)
        
        # Choose object_color_2 (different from object_color_1)
        remaining = [c for c in possible_colors if c != object_color_1]
        object_color_2 = random.choice(remaining)
        
        # Choose even_fill_color, odd_fill_color distinct from the above two
        more_remaining = [c for c in remaining if c != object_color_2]
        even_fill_color = random.choice(more_remaining)
        next_remaining = [c for c in more_remaining if c != even_fill_color]
        odd_fill_color = random.choice(next_remaining)
        
        # Prepare the dictionary of task-level variables
        taskvars = {
            'grid_size': grid_size,
            'object_color_1': object_color_1,
            'object_color_2': object_color_2,
            'even_fill_color': even_fill_color,
            'odd_fill_color': odd_fill_color
        }
        
        # Randomly choose the number of train examples in [3..6]
        nr_train = random.randint(3, 6)
        # We will produce exactly 1 test example
        nr_test = 1
        
        # If we do not need special logic to differentiate across train/test
        # we can use the create_grids_default() helper from ARCTaskGenerator
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create a single input grid:
        1) Generate an N x N zero grid (N=taskvars['grid_size']).
        2) Create exactly one 4-connected object with color = object_color_1.
        3) Place it in the grid at a random location.
        4) Return the resulting grid.
        """
        grid_size = taskvars['grid_size']
        object_color_1 = taskvars['object_color_1']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Randomly choose object dimensions
        obj_h = random.randint(1, grid_size)
        obj_w = random.randint(1, grid_size)
        
        # Create the object matrix of random shape
        object_matrix = create_object(
            height=obj_h,
            width=obj_w,
            color_palette=object_color_1,
            contiguity=Contiguity.FOUR,
            background=0
        )
        
        # Position (top-left corner) where we will paste object_matrix
        max_row_offset = grid_size - obj_h
        max_col_offset = grid_size - obj_w
        row_offset = random.randint(0, max_row_offset)
        col_offset = random.randint(0, max_col_offset)
        
        # Paste object_matrix onto our grid
        for r in range(obj_h):
            for c in range(obj_w):
                if object_matrix[r, c] != 0:
                    grid[row_offset + r, col_offset + c] = object_matrix[r, c]
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars):
        """
        Implement the transformation reasoning chain:
        1) Copy input grid
        2) Find all cells of color object_color_1, change them to object_color_2
        3) Fill empty cells (0) with either even_fill_color or odd_fill_color 
           depending on whether grid_size is even or odd
        """
        object_color_1 = taskvars['object_color_1']
        object_color_2 = taskvars['object_color_2']
        even_fill_color = taskvars['even_fill_color']
        odd_fill_color = taskvars['odd_fill_color']
        grid_size = taskvars['grid_size']
        
        # 1) Copy
        out_grid = np.copy(grid)
        
        # 2) Replace object_color_1 with object_color_2
        out_grid[out_grid == object_color_1] = object_color_2
        
        # 3) Fill empty cells
        fill_color = even_fill_color if (grid_size % 2 == 0) else odd_fill_color
        out_grid[out_grid == 0] = fill_color
        
        return out_grid

