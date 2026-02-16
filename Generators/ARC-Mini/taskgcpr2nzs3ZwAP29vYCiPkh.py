from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# We can import from your provided libraries:
from Framework.input_library import create_object, retry, Contiguity
from Framework.transformation_library import find_connected_objects

class Taskgcpr2nzs3ZwAP29vYCiPkhGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input Reasoning Chain (the observation chain)
        observation_chain = [
            "All input and output matrices are of size {vars['rows']}x{vars['columns']}.",
            "Each input matrix contains a completely filled grey (5) middle column and an object, that is made of 4-way connected cells of {color('color1')} color, located in the left half of the matrix."
        ]
        
        # 2. Transformation Reasoning Chain
        reasoning_chain = [
            "To construct the output matrix, copy the input matrix and reflect the {color('color1')} object in the left half onto the right half of the output matrix, using the grey (5) column as the line of reflection."
        ]
        
        # 3. Superclass init
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        We override create_grids to:
         - sample random values for rows, columns, color1 (with columns being odd)
         - create 3-6 training examples and 1 test example
         - return the dictionary of task variables plus the train/test data
        """
        # Sample the task variables:
        # rows in [5..30], columns in [5..30] but forced to be odd
        rows = random.randint(5, 30)
        # Make sure columns is an odd number within [5..30]
        columns = random.choice([c for c in range(5, 31) if c % 2 == 1])
        color1 = random.randint(1, 9)  # from 1..9

        taskvars = {
            "rows": rows,
            "columns": columns,
            "color1": color1
        }
        
        # Decide how many train examples (3-6)
        nr_train = random.randint(3, 6)
        # We will produce 1 test example
        nr_test = 1
        
        # We can either produce train/test data using our custom logic:
        train_data = []
        for _ in range(nr_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({"input": input_grid, "output": output_grid})
        
        test_data = []
        for _ in range(nr_test):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_data.append({"input": input_grid, "output": output_grid})
        
        train_test_data = {
            "train": train_data,
            "test": test_data
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid with:
          - Size rows x columns
          - Middle column is fully colored grey (5)
          - A single 4-way connected object of color1 in the left half
        """
        rows = taskvars["rows"]
        columns = taskvars["columns"]
        color1 = taskvars["color1"]
        
        # Create empty grid
        grid = np.zeros((rows, columns), dtype=int)
        
        # Fill the middle column with grey (5)
        mid_col = columns // 2
        grid[:, mid_col] = 5
        
        # We want to place an object in the left half, i.e. columns < mid_col
        # The object must be contiguous (4-way).
        # We'll randomly choose some region in [0..mid_col-1], and place an object.
        # We'll use create_object to ensure 4-way connectivity.
        
        def place_connected_object():
            # pick a random sub-dimension in the left half
            # to avoid guaranteed overlap with the grey column, we go strictly up to mid_col
            sub_width = random.randint(1, max(1, mid_col))  # subregion width
            sub_height = random.randint(1, rows)
            
            obj_matrix = create_object(
                height=sub_height,
                width=sub_width,
                color_palette=color1,
                contiguity=Contiguity.FOUR,
                background=0
            )
            return obj_matrix
        
        # We'll keep retrying until the object can be placed in the left half
        # in a position that doesn't overlap the middle column and fits in the grid
        def can_place(obj_matrix: np.ndarray) -> bool:
            # We have sub_height, sub_width
            sub_h, sub_w = obj_matrix.shape
            # We want to place it at some top-left corner in the left half
            # so that sub_w <= mid_col
            # let's see if there's any row in 0..(rows - sub_h)
            # and col in 0..(mid_col - sub_w)
            
            if sub_w > mid_col:
                return False
            return True
        
        obj_matrix = retry(
            generator=place_connected_object,
            predicate=can_place
        )
        
        # Now place obj_matrix into the grid at a random valid position in the left half
        sub_h, sub_w = obj_matrix.shape
        row_top = random.randint(0, rows - sub_h)
        col_left = random.randint(0, mid_col - sub_w)
        
        for r in range(sub_h):
            for c in range(sub_w):
                if obj_matrix[r, c] != 0:
                    grid[row_top + r, col_left + c] = obj_matrix[r, c]
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars):
        """
        Create the output by:
          1) Copying the entire input to output
          2) Reflecting the color1 object in the left half of the matrix
             onto the right half across the middle column.
        """
        rows = taskvars["rows"]
        columns = taskvars["columns"]
        color1 = taskvars["color1"]
        
        # 1) Copy the entire input
        output = grid.copy()
        
        # 2) Identify the color1 object specifically in the left half
        #    Then reflect it horizontally about the middle column.
        mid_col = columns // 2
        
        # Find all connected objects (4-way)
        all_objects = find_connected_objects(grid, diagonal_connectivity=False,
                                             background=0, monochromatic=True)
        # Filter by color1
        color1_objects = all_objects.with_color(color1)
        
        # Potentially there could be multiple color1 objects, but the spec suggests 
        # "an {color('color1')} object" in the left half, so we expect only one main object. 
        # We'll proceed with the largest or the left-most if multiple exist.
        if len(color1_objects) > 1:
            # pick the largest
            color1_objects = color1_objects.sort_by_size(reverse=True)
        
        if len(color1_objects) == 0:
            # If, for any reason, we didn't find it, just return the copy
            return output
        
        main_obj = color1_objects[0]
        
        # We'll reflect it across mid_col
        # Reflection formula: new_col = (2 * mid_col) - old_col
        reflected_cells = set()
        for (r, c, col) in main_obj.cells:
            # Only reflect if it's in the left half
            if c < mid_col:
                new_c = 2 * mid_col - c
                if new_c < columns:
                    reflected_cells.add((r, new_c, col))
        
        # Paste these reflected cells into the output
        for (r, c, col) in reflected_cells:
            output[r, c] = col
        
        return output

