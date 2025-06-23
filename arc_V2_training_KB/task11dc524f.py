from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, GridObjects, find_connected_objects
import numpy as np
import random

class Task11dc524fGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each grid contains a completely filled background of {color('background_color')} color, one {color('block_color')} 2x2 block, and one {color('object_color')} object made of 8-way connected cells. All remaining cells are empty (0).",
            "The {color('object_color')} object always contains an L-shaped part defined as [[c, 0], [c, c]], which may be rotated or reflected and does not necessarily appear in its original orientation. An additional same-colored cell is added to extend the shape of the {color('object_color')} object. Ensuring that the {color('object_color')} object is always made of exactly 4 cells.",
            "The {color('block_color')} 2x2 block is always positioned such that its top-left corner is at exactly ({(vars['grid_size'] - 1) // 2 - 2}, {(vars['grid_size'] - 1) // 2}) position of grid.",
            "The {color('object_color')} object is positioned in a way that when it is moved vertically or horizontally (up, down, left, or right), it becomes 4-way connected to the {color('block_color')} 2x2 block. Specifically, 2 horizontally or vertically connected {color('object_color')} cells are 4-way connected to 2 horizontally or vertically connected  {color('block_color')} cells.",
            "Ensure the both {color('object_color')} object and the {color('block_color')} 2x2 block are always completely separated from each other by {color('background_color')} cells, by having several rows/cols of background_color in between."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and moving the {color('object_color')} object either horizontally or vertically, depending on whether the {color('block_color')} block is to the left/right or above/below the object.",
            "This movement results in two horizontally or vertically connected {color('object_color')} cells becoming 4-way connected to two horizontally or vertically connected {color('block_color')} cells.",
            "Then, the 2x2 {color('block_color')} block is removed, and a reflection of the {color('object_color')} is placed in the position where the {color('block_color')} block originally appeared. The reflection is either vertical or horizontal, depending on how the {color('block_color')} block was placed.",
            "The reflected copy of the {color('object_color')} in the 2x2 area should appear in {color('block_color')} color.",
            "All remaining cells stay unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Generate random task variables
        taskvars = {
            'grid_size': random.choice([9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]),
            'background_color': random.randint(1, 9),
            'block_color': random.randint(1, 9),
            'object_color': random.randint(1, 9)
        }
        
        # Ensure colors are different
        while taskvars['block_color'] == taskvars['background_color']:
            taskvars['block_color'] = random.randint(1, 9)
        while taskvars['object_color'] in [taskvars['background_color'], taskvars['block_color']]:
            taskvars['object_color'] = random.randint(1, 9)
        
        train_test_data = {'train': [], 'test': []}
        
        # Create training grids with specific placements
        # First: object to the left of block
        gridvars = {'placement': 'left', 'use_test_shape': False}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['train'].append({'input': input_grid, 'output': output_grid})
        
        # Second: object above block
        gridvars = {'placement': 'above', 'use_test_shape': False}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['train'].append({'input': input_grid, 'output': output_grid})
        
        # Third: random placement
        gridvars = {'placement': random.choice(['right', 'below']), 'use_test_shape': False}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['train'].append({'input': input_grid, 'output': output_grid})
        
        # Optional fourth training example
        if random.choice([True, False]):
            gridvars = {'placement': random.choice(['left', 'right', 'above', 'below']), 'use_test_shape': False}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_test_data['train'].append({'input': input_grid, 'output': output_grid})
        
        # Test grid - use new shape and only left/right placement
        gridvars = {'placement': random.choice(['left', 'right']), 'use_test_shape': True}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['test'].append({'input': input_grid, 'output': output_grid})
        
        return taskvars, train_test_data

    def safe_random_gap(self, minimum, maximum):
        if maximum >= minimum:
            return random.randint(minimum, maximum)
        else:
            return minimum

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        background_color = taskvars['background_color']
        block_color = taskvars['block_color']
        object_color = taskvars['object_color']
        placement = gridvars['placement']
        use_test_shape = gridvars.get('use_test_shape', False)

        grid = np.full((grid_size, grid_size), background_color, dtype=int)

        block_row = (grid_size - 1) // 2 - 2
        block_col = (grid_size - 1) // 2
        grid[block_row:block_row+2, block_col:block_col+2] = block_color

        if use_test_shape and placement in ['left', 'right']:
            obj_shape = np.array([[0, 1], [1, 1], [1, 0]]) * object_color
            if placement == 'left':
                max_gap = block_col - 2
                gap = self.safe_random_gap(3, min(5, max_gap))
                obj_row = block_row
                obj_col = block_col - gap - 1
            else:
                max_gap = grid_size - block_col - 4
                gap = self.safe_random_gap(3, min(5, max_gap))
                obj_row = block_row - 1
                obj_col = block_col + 2 + gap
        else:
            if placement == 'left':
                obj_shape = np.array([[1, 0], [0, 1], [1, 1]]) * object_color
                max_gap = block_col - 2
                gap = self.safe_random_gap(3, min(5, max_gap))
                obj_row = max(0, block_row - 1)
                obj_col = block_col - gap - 1
            elif placement == 'right':
                obj_shape = np.array([[1, 0, 0], [1, 1, 1]]) * object_color
                max_gap = grid_size - block_col - 5
                gap = self.safe_random_gap(3, min(5, max_gap))
                obj_row = block_row
                obj_col = block_col + 2 + gap
            elif placement == 'above':
                obj_shape = np.array([[1, 0], [1, 0], [1, 1]]) * object_color
                max_gap = block_row - 3
                gap = self.safe_random_gap(3, min(5, max_gap))
                obj_row = block_row - gap - 3
                obj_col = block_col
            else:
                obj_shape = np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]]) * object_color
                max_gap = grid_size - block_row - 5
                gap = self.safe_random_gap(3, min(5, max_gap))
                obj_row = block_row + 2 + gap
                obj_col = max(0, block_col - 1)

        obj_h, obj_w = obj_shape.shape
        obj_row = max(0, min(obj_row, grid_size - obj_h))
        obj_col = max(0, min(obj_col, grid_size - obj_w))

        for r in range(obj_h):
            for c in range(obj_w):
                if obj_shape[r, c] != 0:
                    grid[obj_row + r, obj_col + c] = obj_shape[r, c]

        return grid

    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        background_color = taskvars['background_color']
        block_color = taskvars['block_color']
        object_color = taskvars['object_color']
        
        # Create output grid as copy
        output = grid.copy()
        
        # Find the 2x2 block position
        block_row = (grid_size - 1) // 2 - 2
        block_col = (grid_size - 1) // 2
        
        # Find the object
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=background_color)
        object_found = None
        for obj in objects:
            if object_color in obj.colors and len(obj) == 4:
                object_found = obj
                break
        
        if object_found is None:
            return output
        
        # Store original object bounding box for reflection positioning
        original_bbox = object_found.bounding_box
        
        # Get object bounding box to determine relative position
        obj_min_row = original_bbox[0].start
        obj_max_row = original_bbox[0].stop - 1
        obj_min_col = original_bbox[1].start
        obj_max_col = original_bbox[1].stop - 1
        
        # Step 1: Determine direction based on where the BLOCK is relative to the OBJECT
        if block_col > obj_max_col:  # Block is to the RIGHT of object
            # Move right to connect and reflect horizontally to the right
            dx = block_col - obj_max_col - 1
            dy = 0
            reflection_type = 'horizontal'
            
        elif block_col + 1 < obj_min_col:  # Block is to the LEFT of object
            # Move left to connect
            dx = -(obj_min_col - block_col - 2)  # Negative because moving left
            dy = 0
            reflection_type = 'horizontal'
            
        elif block_row > obj_max_row:  # Block is BELOW object
            # Move down to connect and reflect vertically downward
            dx = 0
            dy = block_row - obj_max_row - 1
            reflection_type = 'vertical'
            
        else:  # Block is ABOVE object
            # Move up to connect and reflect vertically upward
            dx = 0
            dy = block_row - obj_min_row -2  # Calculate the upward movement
            reflection_type = 'vertical'
        
        # Step 2: Move the object
        object_found.cut(output, background=background_color)
        object_found.translate(dy, dx)
        object_found.paste(output)
        
        # Step 3: Remove the 2x2 block
        output[block_row:block_row+2, block_col:block_col+2] = background_color
        
        # Step 4: Get the moved object as array and create reflection
        moved_bbox = object_found.bounding_box
        obj_array = object_found.to_array()

        if reflection_type == 'horizontal':
            reflected = np.fliplr(obj_array)
            # Position the reflection based on block direction
            ref_row = moved_bbox[0].start
            if dx > 0:  # Block was to the right - reflect to the right
                ref_col = moved_bbox[1].stop  # Start right after the moved object
            else:  # Block was to the left - reflect to the left
                ref_col = moved_bbox[1].start - reflected.shape[1]  # Place before the moved object
        else:
            reflected = np.flipud(obj_array)
            # Position the reflection based on block direction
            ref_col = moved_bbox[1].start
            if dy > 0:  # Block was below - reflect downward
                ref_row = moved_bbox[0].stop  # Start right below the moved object
            else:  # Block was above - reflect upward
                ref_row = moved_bbox[0].start - reflected.shape[0]  #
        
        # Step 5: Change color of reflected object to block_color
        reflected[reflected == object_color] = block_color
        
        # Step 6: Place the reflected object
        ref_h, ref_w = reflected.shape
        for r in range(ref_h):
            for c in range(ref_w):
                if reflected[r, c] != 0:
                    target_r = ref_row + r
                    target_c = ref_col + c
                    if 0 <= target_r < grid_size and 0 <= target_c < grid_size:
                        output[target_r, target_c] = reflected[r, c]
        
        return output




