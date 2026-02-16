from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
import numpy as np
import random

class TaskjJmWYiqtS4pEHmcMJdbRVtGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a {color('object_color')} 4-way connected object, that is a 3x3 block with the middle cell of either {color('cell_color1')} or {color('cell_color2')} color.",
            "The position of the 3x3 object can vary across examples."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and recoloring the middle cell of the 3x3 object, the cell to its right, and the cell below with a different color.",
            "The new fill color depends on the color of the middle cell of the 3x3 object: if it is {color('cell_color1')}, the fill color is {color('fill_color1')}; if it is {color('cell_color2')}, the fill color is {color('fill_color2')}."
        ]

        # 3) Call the superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)


    def create_grids(self):
        """
        Create the dictionary of task variables (vars) and the train/test data.
        We ensure at least one example in train and test where the middle cell is cell_color1
        and at least one example where the middle cell is cell_color2.
        """
        # 1. Choose distinct color variables
        distinct_colors = random.sample(range(1, 10), 5)
        cell_color1, cell_color2, object_color, fill_color1, fill_color2 = distinct_colors
        
        # 2. Choose random grid dimensions
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        
        # Prepare the dictionary of task variables
        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2,
            'object_color': object_color,
            'fill_color1': fill_color1,
            'fill_color2': fill_color2
        }
        
        # We want 3-5 training examples; let's pick a random number in that range
        nr_train_examples = random.randint(3, 5)
        # We definitely need at least one with center_color = cell_color1 and one with center_color2
        # We'll distribute them accordingly. For simplicity, we'll ensure we have both colors present,
        # and fill the remainder randomly.
        
        # We'll store a small helper function to pick center_color (to guarantee coverage):
        def pick_center_color(i):
            # If we have fewer examples than 4, ensure coverage in the first two
            # If i == 0, pick cell_color1, if i == 1, pick cell_color2,
            # otherwise random among the two.
            if i == 0:
                return cell_color1
            elif i == 1:
                return cell_color2
            else:
                return random.choice([cell_color1, cell_color2])

        train_pairs = []
        for i in range(nr_train_examples):
            center_col = pick_center_color(i)
            input_grid = self.create_input(taskvars, {'center_color': center_col})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create exactly two test examples, one with each center color
        test_pairs = []
        for center_col in [cell_color1, cell_color2]:
            input_grid = self.create_input(taskvars, {'center_color': center_col})
            output_grid = self.transform_input(input_grid, taskvars)
            test_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }
        
        return taskvars, train_test_data


    def create_input(self, taskvars, gridvars):
        """
        Create a grid of size rows x cols = taskvars['rows'] x taskvars['cols'],
        containing one 3x3 block of color taskvars['object_color'] except for the middle cell,
        which is either cell_color1 or cell_color2. The position of the 3x3 block varies.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        object_color = taskvars['object_color']
        center_color = gridvars['center_color']  # Either cell_color1 or cell_color2

        grid = np.zeros((rows, cols), dtype=int)
        
        # Random position for the 3x3 block, ensuring it stays within bounds
        offset_row = random.randint(0, rows - 3)
        offset_col = random.randint(0, cols - 3)
        
        # Fill the 3x3 block with the object color
        for r in range(offset_row, offset_row + 3):
            for c in range(offset_col, offset_col + 3):
                grid[r, c] = object_color
        
        # Overwrite the middle cell with the chosen center color
        grid[offset_row + 1, offset_col + 1] = center_color

        return grid


    def transform_input(self, grid, taskvars):
        """
        Transform the input grid by:
         1) Copying the grid
         2) Finding the single 3x3 block (which is a 4-connected object of size 9)
         3) Identifying its center cell
         4) Changing the center cell, the cell to its right, and the cell below
            to fill_color1 (if the center cell is cell_color1) or fill_color2 (if the center is cell_color2)
        """
        out_grid = grid.copy()
        
        # Find the single 3x3 connected object of size 9
        objects = find_connected_objects(
            out_grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=False
        )
        block_obj_candidates = objects.with_size(9, 9).objects
        if not block_obj_candidates:
            # Fallback if something unexpected
            return out_grid
        block_obj = block_obj_candidates[0]
        
        # Find the bounding box (should be exactly 3x3)
        r_slice, c_slice = block_obj.bounding_box
        center_r = r_slice.start + 1
        center_c = c_slice.start + 1
        
        center_cell_color = out_grid[center_r, center_c]
        
        # Determine fill color based on the center cell color
        if center_cell_color == taskvars['cell_color1']:
            fill_color = taskvars['fill_color1']
        else:
            fill_color = taskvars['fill_color2']
        
        # Recolor the center cell, the cell to its right, and the cell below
        out_grid[center_r, center_c] = fill_color
        
        # Guard checks in case the center is on the right or bottom boundary
        if center_c + 1 < out_grid.shape[1]:
            out_grid[center_r, center_c + 1] = fill_color
        if center_r + 1 < out_grid.shape[0]:
            out_grid[center_r + 1, center_c] = fill_color
        
        return out_grid



