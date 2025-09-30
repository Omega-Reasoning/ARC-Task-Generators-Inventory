from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, BorderBehavior

class Task4938f0c2Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain two colored objects, one {color('objectcol1')} and one {color('objectcol2')}, with the remaining cells being empty (0).",
            "The {color('objectcol1')} object consists of 8-way connected cells, with its shape varying across grids, while the {color('objectcol2')} object is always a 2x2 colored block.",
            "The bottom-right corner of the {color('objectcol1')} object is always connected to the top-left corner of the 2x2 block.",
            "The two objects are positioned to ensure sufficient space is available for attaching three additional {color('objectcol1')} objects, identical in shape and size to the existing one, at the top-right, bottom-left, and bottom-right corners of the 2x2 block."
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the two colored objects.",
            "The {color('objectcol1')} object is reflected to the right by considering the two columns containing the {color('objectcol2')} block as the line of reflection.",
            "Next, the two {color('objectcol1')} objects; original and the mirrored copy of the original, are reflected vertically downward by considering the two rows containing the 2x2 block as the line of reflection."
        ]

        taskvars_definitions = {}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars.get('rows', 15)
        cols = taskvars.get('cols', 15)
        objectcol1 = taskvars['objectcol1']
        objectcol2 = taskvars['objectcol2']
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Create a 2x2 block at a position that allows for reflections
        # Ensure it's not too close to the edges to allow for the reflections
        block_row = np.random.randint(rows // 3, 2 * rows // 3 - 1)
        block_col = np.random.randint(cols // 3, 2 * cols // 3 - 1)
        
        # Create 2x2 block
        grid[block_row:block_row+2, block_col:block_col+2] = objectcol2
        
        # Define the size of the objectcol1 object
        object_height = np.random.randint(3, min(6, block_row))
        object_width = np.random.randint(3, min(6, block_col))
        
        def generate_valid_object():
            # Create a random connected object
            obj_grid = create_object(
                height=object_height,
                width=object_width,
                color_palette=objectcol1,
                contiguity=Contiguity.EIGHT,
                background=0
            )
            
            # Check if the object has cells
            if np.sum(obj_grid != 0) == 0:
                return None
                
            # Ensure the bottom-right cell of the object exists
            if obj_grid[-1, -1] == 0:
                obj_grid[-1, -1] = objectcol1  # Force bottom-right cell to be filled
            
            # Place the object so that its bottom-right corner is diagonally connected 
            # to the top-left corner of the 2x2 block
            tl_row = block_row - object_height
            tl_col = block_col - object_width
            
            # Check if it fits within the grid
            if tl_row < 0 or tl_col < 0:
                return None
                
            # Place the object in the grid
            temp_grid = grid.copy()
            temp_grid[tl_row:block_row, tl_col:block_col] = obj_grid
            
            # Verify we have one properly connected objectcol1 object
            objects = find_connected_objects(temp_grid, diagonal_connectivity=True)
            objectcol1_objects = objects.with_color(objectcol1)
            
            if len(objectcol1_objects) != 1:
                return None
            
            # Verify that the objects are properly diagonally connected
            # The bottom-right of objectcol1 should be at (block_row-1, block_col-1)
            # and the top-left of objectcol2 is at (block_row, block_col)
            if temp_grid[block_row-1, block_col-1] != objectcol1:
                return None
                
            return temp_grid
        
        # Keep trying until we get a valid object
        valid_grid = retry(generate_valid_object, lambda g: g is not None, max_attempts=100)
        
        # Check if we have enough space for reflections
        # We need space on the right and below the 2x2 block
        if block_col + 2 + object_width > cols or block_row + 2 + object_height > rows:
            # If not enough space, try again with a smaller grid (recursively)
            return self.create_input(taskvars, gridvars)
            
        return valid_grid

    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid for output
        output_grid = grid.copy()
        
        objectcol1 = taskvars['objectcol1']
        objectcol2 = taskvars['objectcol2']
        
        # Find the connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=True)
        
        # Get the objectcol1 and objectcol2 objects
        obj1 = objects.with_color(objectcol1)[0]
        obj2 = objects.with_color(objectcol2)[0]
        
        # Find the 2x2 block's bounding box
        block_box = obj2.bounding_box
        block_top = block_box[0].start
        block_left = block_box[1].start
        block_bottom = block_box[0].stop - 1
        block_right = block_box[1].stop - 1
        
        # Convert the objectcol1 object to array for easier manipulation
        obj1_array = obj1.to_array()
        obj1_height, obj1_width = obj1_array.shape
        
        # Get the coordinates of the object in the grid
        obj1_box = obj1.bounding_box
        obj1_top = obj1_box[0].start
        obj1_left = obj1_box[1].start
        
        # Reflect horizontally across the column line of the 2x2 block
        # Calculate the new position for the horizontally reflected object
        h_reflect_left = block_right + 1
        h_reflect_top = obj1_top  # Same top position
        
        # Create the horizontally reflected object
        h_reflected = np.fliplr(obj1_array)
        
        # Place the horizontally reflected object in the output grid
        for r in range(obj1_height):
            for c in range(obj1_width):
                if h_reflected[r, c] != 0:
                    new_r = h_reflect_top + r
                    new_c = h_reflect_left + c
                    if 0 <= new_r < output_grid.shape[0] and 0 <= new_c < output_grid.shape[1]:
                        output_grid[new_r, new_c] = objectcol1
        
        # Now reflect both objects vertically across the row line of the 2x2 block
        # First, reflect the original object
        v_reflect_top = block_bottom + 1
        v_reflect_left = obj1_left  # Same left position
        
        # Create the vertically reflected original object
        v_reflected_orig = np.flipud(obj1_array)
        
        # Place the vertically reflected original object in the output grid
        for r in range(obj1_height):
            for c in range(obj1_width):
                if v_reflected_orig[r, c] != 0:
                    new_r = v_reflect_top + r
                    new_c = v_reflect_left + c
                    if 0 <= new_r < output_grid.shape[0] and 0 <= new_c < output_grid.shape[1]:
                        output_grid[new_r, new_c] = objectcol1
        
        # Next, reflect the horizontally reflected object vertically
        v_reflect_h_top = block_bottom + 1
        v_reflect_h_left = h_reflect_left  # Same left position as horizontally reflected
        
        # Create the vertically reflected horizontally reflected object
        v_h_reflected = np.flipud(h_reflected)
        
        # Place the vertically and horizontally reflected object in the output grid
        for r in range(obj1_height):
            for c in range(obj1_width):
                if v_h_reflected[r, c] != 0:
                    new_r = v_reflect_h_top + r
                    new_c = v_reflect_h_left + c
                    if 0 <= new_r < output_grid.shape[0] and 0 <= new_c < output_grid.shape[1]:
                        output_grid[new_r, new_c] = objectcol1
        
        return output_grid

    def create_grids(self):
        # Define task variables with randomized values
        taskvars = {
            'rows': np.random.randint(15, 30),
            'cols': np.random.randint(15, 30),
        }
        
        # Ensure objectcol1 and objectcol2 are different
        objectcol1 = np.random.randint(1, 10)
        objectcol2 = np.random.randint(1, 10)
        while objectcol2 == objectcol1:
            objectcol2 = np.random.randint(1, 10)
            
        taskvars['objectcol1'] = objectcol1
        taskvars['objectcol2'] = objectcol2
        
        # Create random number of training examples (3-4)
        num_train = np.random.randint(3, 5)
        
        train_data = []
        for _ in range(num_train):
            # Create a new input grid for each example
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create one test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        test_data = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_data,
            'test': test_data
        }

