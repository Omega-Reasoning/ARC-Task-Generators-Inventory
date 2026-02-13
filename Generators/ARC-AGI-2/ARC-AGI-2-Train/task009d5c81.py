from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple

from input_library import Contiguity, create_object, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task009d5c81Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain exactly two different colored objects, object A and object B, with the remaining cells being empty (0).",
            "Object A and B are of colors {color('object_a')} and {color('object_b')} respectively.",
            "Object A and B must be completely separated from each other by empty (0) rows and columns.",
            "Object B can only have three specific shape styles, each confined within a 3x3 grid.",
            "Style 1 is defined as: [[0, c, 0], [c, c, c], [0, c, 0]]; style 2 is defined as: [[c, c, c], [c, 0, c], [0, c, 0]]; style 3 is defined as: [[c, 0, c], [0, c, 0], [c, c, c]] for color c.",
            "Object A is always bigger than object B and is made of 8-way connected cells of {color('object_a')} color."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are constructed by copying the input grids and identifying the object A and B having colors {color('object_a')}  and {color('object_b')} respectively.",
            "Object B can have one of three specific styles: Style 1 [[0, c, 0], [c, c, c], [0, c, 0]], Style 2 [[c, c, c], [c, 0, c], [0, c, 0]], or Style 3 [[c, 0, c], [0, c, 0], [c, c, c]], where c refers to the color of object B.",
            "Based on the identified style of object B, the color of object A is changed: if object B has Style 1, the color of object A becomes {color('change_1')}; if Style 2, it becomes {color('change_2')}; and if Style 3, it becomes {color('change_3')}.",
            "Once the color of object A has been changed remove object B."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Define task variables
        taskvars = {}
        
        # Set grid size (between 11 and 30)
        taskvars['grid_size'] = random.randint(11, 20)
        
        # Choose 5 different colors for object_a, object_b, and the 3 change colors
        colors = random.sample(range(1, 10), 5)
        taskvars['object_a'] = colors[0]
        taskvars['object_b'] = colors[1]
        taskvars['change_1'] = colors[2]
        taskvars['change_2'] = colors[3]
        taskvars['change_3'] = colors[4]
        
        # Create training examples with each style represented
        train_examples = []
        
        # Ensure we have at least one example of each style
        styles_to_use = [0, 1, 2]  # 0: Style 1, 1: Style 2, 2: Style 3
        
        # Create examples with each style
        for style in styles_to_use:
            gridvars = {'style': style}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Add 2 more random examples
        for _ in range(2):
            gridvars = {'style': random.choice(styles_to_use)}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_gridvars = {'style': random.choice(styles_to_use)}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        object_a_color = taskvars['object_a']
        object_b_color = taskvars['object_b']
        style = gridvars['style']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Define the three styles for object B
        style_1 = np.array([[0, object_b_color, 0], 
                           [object_b_color, object_b_color, object_b_color], 
                           [0, object_b_color, 0]])
        
        style_2 = np.array([[object_b_color, object_b_color, object_b_color], 
                           [object_b_color, 0, object_b_color], 
                           [0, object_b_color, 0]])
        
        style_3 = np.array([[object_b_color, 0, object_b_color], 
                           [0, object_b_color, 0], 
                           [object_b_color, object_b_color, object_b_color]])
        
        styles = [style_1, style_2, style_3]
        object_b = styles[style]
        
        # Create object A (must be larger than object B)
        # Object B has 5 cells in style 1 & 3, and 6 cells in style 2
        min_object_a_size = 10  # Ensure object A is clearly larger than B
        
        def create_object_a():
            # The height and width of object A should be randomized but reasonable
            height = random.randint(3, min(7, grid_size//2))
            width = random.randint(3, min(7, grid_size//2))
            obj = create_object(height, width, object_a_color, Contiguity.EIGHT)
            return obj
        
        def valid_object_a(obj):
            # Ensure the object has enough cells
            return np.sum(obj != 0) >= min_object_a_size
        
        object_a = retry(create_object_a, valid_object_a)
        
        # Place object B at a random valid position
        b_height, b_width = object_b.shape
        
        def valid_placement(pos_a, pos_b, obj_a, obj_b_shape):
            a_r, a_c = pos_a
            b_r, b_c = pos_b
            a_h, a_w = obj_a.shape
            b_h, b_w = obj_b_shape
            
            # Check if object B would be within grid bounds
            if b_r + b_h > grid_size or b_c + b_w > grid_size:
                return False
                
            # Check if objects would have empty rows/columns between them
            # Get bounding boxes
            a_top, a_bottom = a_r, a_r + a_h - 1
            a_left, a_right = a_c, a_c + a_w - 1
            b_top, b_bottom = b_r, b_r + b_h - 1
            b_left, b_right = b_c, b_c + b_w - 1
            
            # Check for separation: objects must not share rows or columns
            # Either B must be completely below A, or A must be completely below B
            vertical_separation = (b_top > a_bottom + 1) or (a_top > b_bottom + 1)
            # Either B must be completely to the right of A, or A must be completely to the right of B
            horizontal_separation = (b_left > a_right + 1) or (a_left > b_right + 1)
            
            return vertical_separation or horizontal_separation
        
        # Try to place both objects
        max_attempts = 100
        for _ in range(max_attempts):
            # Place object A randomly
            a_row = random.randint(0, grid_size - object_a.shape[0])
            a_col = random.randint(0, grid_size - object_a.shape[1])
            
            # Try different placements for object B
            for attempt in range(50):
                b_row = random.randint(0, grid_size - b_height)
                b_col = random.randint(0, grid_size - b_width)
                
                if valid_placement((a_row, a_col), (b_row, b_col), object_a, object_b.shape):
                    # Place both objects
                    for r in range(object_a.shape[0]):
                        for c in range(object_a.shape[1]):
                            if object_a[r, c] != 0:
                                grid[a_row + r, a_col + c] = object_a[r, c]
                    
                    for r in range(b_height):
                        for c in range(b_width):
                            if object_b[r, c] != 0:
                                grid[b_row + r, b_col + c] = object_b[r, c]
                    
                    return grid
        
        # If we can't find a valid placement, create a larger grid temporarily
        backup_size = grid_size + 10
        backup_grid = np.zeros((backup_size, backup_size), dtype=int)
        
        # Place object A in the top-left quadrant
        a_row, a_col = 2, 2
        for r in range(object_a.shape[0]):
            for c in range(object_a.shape[1]):
                if object_a[r, c] != 0:
                    backup_grid[a_row + r, a_col + c] = object_a[r, c]
        
        # Place object B in the bottom-right quadrant
        b_row, b_col = grid_size - b_height - 2, grid_size - b_width - 2
        for r in range(b_height):
            for c in range(b_width):
                if object_b[r, c] != 0:
                    backup_grid[b_row + r, b_col + c] = object_b[r, c]
        
        # Crop back to original size
        return backup_grid[:grid_size, :grid_size]
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        object_a_color = taskvars['object_a']
        object_b_color = taskvars['object_b']
        change_1 = taskvars['change_1']
        change_2 = taskvars['change_2']
        change_3 = taskvars['change_3']
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=True)
        
        # Find object A and B
        object_a = None
        object_b = None
        
        for obj in objects:
            if object_a_color in obj.colors:
                object_a = obj
            elif object_b_color in obj.colors:
                object_b = obj
        
        if object_b is None:
            return output_grid  # Return unchanged if we can't find object B
        
        # Convert object B to array format to identify its style
        b_array = np.zeros((3, 3), dtype=int)
        
        # Normalize the coordinates to fit in a 3x3 grid
        b_coords = list(object_b.coords)
        min_row = min(r for r, _, in b_coords)
        min_col = min(c for _, c in b_coords)
        
        for r, c in b_coords:
            norm_r = r - min_row
            norm_c = c - min_col
            if 0 <= norm_r < 3 and 0 <= norm_c < 3:
                b_array[norm_r, norm_c] = object_b_color
        
        # Define the patterns to match
        style_1 = np.array([[0, object_b_color, 0], 
                           [object_b_color, object_b_color, object_b_color], 
                           [0, object_b_color, 0]])
        
        style_2 = np.array([[object_b_color, object_b_color, object_b_color], 
                           [object_b_color, 0, object_b_color], 
                           [0, object_b_color, 0]])
        
        style_3 = np.array([[object_b_color, 0, object_b_color], 
                           [0, object_b_color, 0], 
                           [object_b_color, object_b_color, object_b_color]])
        
        # Determine which style object B has
        new_color = None
        if np.array_equal(b_array, style_1):
            new_color = change_1
        elif np.array_equal(b_array, style_2):
            new_color = change_2
        elif np.array_equal(b_array, style_3):
            new_color = change_3
        
        # Change the color of object A based on the style of object B
        if new_color is not None:
            for r, c in object_a.coords:
                output_grid[r, c] = new_color
        
        # Remove object B
        for r, c in object_b.coords:
            output_grid[r, c] = 0
            
        return output_grid

