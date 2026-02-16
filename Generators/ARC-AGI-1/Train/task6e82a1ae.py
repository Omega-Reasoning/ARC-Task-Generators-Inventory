from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity
from Framework.transformation_library import GridObject, GridObjects, find_connected_objects

class Task6e82a1aeGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "They contain several {color('object_color')} objects, with the remaining cells being empty (0).",
            "The objects can have the following specific shapes: 2×1 rectangle; 3×1 rectangle; 2×2 square; an L-shaped object where L is defined as [[c, 0], [c, c]] for a color c; a T-shaped object where C is defined as [[c, c, c],[0, c, 0]] for a color c; and a Z-shaped object where Z is defined as [[c, c, 0], [0, c, c]] for a color c.",
            "These objects may be rotated or reflected, and can slightly differ in orientation, but their original shape and size must not change.",
            "Each input grid must include at least one object made of two cells; one object made of three cells; and one object made of four cells.",
            "The objects must not be connected to each other, and there should be a clear separation between each of them."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grids and identifying the number of cells used for creating each {color('object_color')} object.",
            "Once identified, change the color of each object according to the number of cells it contains.",
            "If an object is made of two cells, its color should change to {color('new_color1')}; if it is made of three cells, its color should change to {color('new_color2')}; and if it is made of four cells, its color should change to {color('new_color3')}.",
            "The transformation does not affect the original shape and size of the object.",
            "All empty (0) cells remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(10, 30),
            'object_color': random.randint(1, 9)
        }
        
        # Ensure all colors are different
        available_colors = list(set(range(1, 10)) - {taskvars['object_color']})
        random.shuffle(available_colors)
        taskvars['new_color1'] = available_colors[0]
        taskvars['new_color2'] = available_colors[1]
        taskvars['new_color3'] = available_colors[2]
        
        # Create train and test examples
        nr_train_examples = random.randint(3, 4)
        
        train_examples = []
        for _ in range(nr_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create one test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        
        # Create an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Define all possible shapes (in their original orientation)
        shapes = {
            # 2-cell shapes
            'rect_2x1': np.array([[object_color, object_color]]),
            
            # 3-cell shapes
            'rect_3x1': np.array([[object_color, object_color, object_color]]),
            
            # 4-cell shapes
            'square_2x2': np.array([[object_color, object_color], 
                                    [object_color, object_color]]),
            'L_shape': np.array([[object_color, 0], 
                                 [object_color, object_color]]),
            'T_shape': np.array([[object_color, object_color, object_color], 
                                 [0, object_color, 0]]),
            'Z_shape': np.array([[object_color, object_color, 0], 
                                 [0, object_color, object_color]])
        }
        
        # Group shapes by cell count
        shapes_by_size = {
            2: ['rect_2x1'],
            3: ['rect_3x1'],
            4: ['square_2x2', 'L_shape', 'T_shape', 'Z_shape']
        }
        
        # Determine number of objects (more than 4, at most grid_size//2+1)
        num_objects = random.randint(5, min(10, grid_size//2+1))
        
        # Ensure at least one of each required size
        object_sizes = [2, 3, 4] + random.choices([2, 3, 4], k=num_objects-3)
        random.shuffle(object_sizes)
        
        # Function to place an object on the grid
        def try_place_object(shape, max_attempts=100):
            original_shape = shape.copy()
            
            for _ in range(max_attempts):
                # Randomly rotate or reflect the shape
                if random.random() < 0.5:
                    # Rotate
                    k = random.choice([0, 1, 2, 3])
                    shape = np.rot90(original_shape, k)
                else:
                    # Reflect
                    axis = random.choice([0, 1, None])
                    if axis is None:
                        shape = np.flip(original_shape)
                    else:
                        shape = np.flip(original_shape, axis)
                
                height, width = shape.shape
                
                # Random position
                r = random.randint(0, grid_size - height)
                c = random.randint(0, grid_size - width)
                
                # Check if we can place the object without overlapping or adjacent to others
                can_place = True
                
                # Define the region to check (include 1-cell buffer around the object)
                check_r_start = max(0, r - 1)
                check_r_end = min(grid_size, r + height + 1)
                check_c_start = max(0, c - 1)
                check_c_end = min(grid_size, c + width + 1)
                
                for i in range(check_r_start, check_r_end):
                    for j in range(check_c_start, check_c_end):
                        if 0 <= i < grid_size and 0 <= j < grid_size and grid[i, j] != 0:
                            can_place = False
                            break
                    if not can_place:
                        break
                
                if can_place:
                    # Place the object
                    for i in range(height):
                        for j in range(width):
                            if shape[i, j] != 0:
                                grid[r+i, c+j] = shape[i, j]
                    return True
            
            return False
        
        # Place objects on the grid
        for size in object_sizes:
            shape_type = random.choice(shapes_by_size[size])
            shape = shapes[shape_type]
            
            if not try_place_object(shape):
                # If failed to place, try a different shape of the same size
                fallback_shapes = shapes_by_size[size].copy()
                fallback_shapes.remove(shape_type)
                if fallback_shapes:
                    shape_type = random.choice(fallback_shapes)
                    shape = shapes[shape_type]
                    if not try_place_object(shape):
                        # If still failed, just create a simpler shape
                        if size == 2:
                            rect = np.array([[object_color, object_color]])
                            try_place_object(rect)
                        elif size == 3:
                            rect = np.array([[object_color, object_color, object_color]])
                            try_place_object(rect)
                        elif size == 4:
                            square = np.array([[object_color, object_color], 
                                              [object_color, object_color]])
                            try_place_object(square)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        new_color1 = taskvars['new_color1']
        new_color2 = taskvars['new_color2']
        new_color3 = taskvars['new_color3']
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # Process each object
        for obj in objects:
            if obj.colors == {object_color}:
                size = obj.size
                new_color = None
                
                # Determine the new color based on size
                if size == 2:
                    new_color = new_color1
                elif size == 3:
                    new_color = new_color2
                elif size == 4:
                    new_color = new_color3
                
                # Change the color of the object in the output grid
                if new_color is not None:
                    for r, c, _ in obj.cells:
                        output_grid[r, c] = new_color
        
        return output_grid

