from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import Contiguity, create_object, enforce_object_height, enforce_object_width, retry
import numpy as np
import random

class TaskRCdwdHBGotnBYezKj6t6amGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
           "All input grids are of size {vars['n']}x{vars['n']}.",
           "Each input grid contains a 4-way connected object of size {vars['n']-2}x{vars['n']-2}.",
           "The first and the last, rows and columns are empty (0).",
           "The object is either {color('object_color1')} or {color('object_color2')}."
        ]
        reasoning_chain = [ 
           "To construct the output grid, copy the input grid and apply the following transformation.",
           "The left half of the object retains its original color.",
           "The right half of the object is transformed to a new color, based on the color of the input grid: {color('object_color1')} becomes {color('object_color3')} and {color('object_color2')} becomes {color('object_color4')}."
        ]
        super().__init__(observation_chain, reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        n = taskvars["n"]
        chosen_color = gridvars["chosen_color"]
        grid = np.zeros((n, n), dtype=int)
        
        # Calculate dimensions for the left-side object
        height = n - 2  # Leave space at top and bottom
        width = (n // 2) - 1  # Use left half minus border
        
        # Define a function to generate a valid object
        def generate_valid_object():
            # Create base object
            obj = create_object(
                height=height,
                width=width,
                color_palette=chosen_color,
                contiguity=Contiguity.FOUR,
                background=0
            )
            
            # Verify it meets our requirements
            if obj.shape != (height, width):
                return None
            if not find_connected_objects(obj, background=0):
                return None
            return obj
        
        # Generate object with retries
        left_object = None
        for _ in range(500):  # Max attempts
            candidate = generate_valid_object()
            if candidate is not None:
                left_object = candidate
                break
        
        # Fallback if we couldn't generate a valid object
        if left_object is None:
            left_object = np.ones((height, width), dtype=int) * chosen_color
        
        # Place the left object in the grid with 1-cell border
        grid[1:n-1, 1:width+1] = left_object
        
        # Mirror the left object to the right side (flip horizontally)
        right_start = n - width - 1
        grid[1:n-1, right_start:n-1] = np.fliplr(left_object)
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars=None) -> np.ndarray:
        n = grid.shape[0]
        mid_col = n // 2
        output = grid.copy()

        # Transform any non-zero pixel in the right half
        for r in range(n):
            for c in range(mid_col, n):
                if output[r, c] != 0:
                    if output[r, c] == taskvars['object_color1']:
                        output[r, c] = taskvars['object_color3']
                    elif output[r, c] == taskvars['object_color2']:
                        output[r, c] = taskvars['object_color4']

        return output

    def create_grids(self):
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        obj_col1, obj_col2, obj_col3, obj_col4 = all_colors[:4]

        taskvars = {
            "object_color1": obj_col1,
            "object_color2": obj_col2,
            "object_color3": obj_col3,
            "object_color4": obj_col4,
            "n": random.choice([8, 10, 12])  # Using more reliable grid sizes
        }

        nr_train = random.randint(3, 5)
        
        # Generate training examples
        train_examples = []
        colors_used = set()
        
        for _ in range(nr_train):
            # Alternate between colors to ensure both are represented
            if len(colors_used) < 2:
                chosen_color = taskvars['object_color1'] if taskvars['object_color1'] not in colors_used else taskvars['object_color2']
            else:
                chosen_color = random.choice([taskvars['object_color1'], taskvars['object_color2']])
            
            colors_used.add(chosen_color)
            train_examples.append({
                'input': (input_grid := self.create_input(taskvars, {"chosen_color": chosen_color})),
                'output': self.transform_input(input_grid, taskvars)
            })
        
        # Generate one test example with random color
        test_examples = [{
            'input': (input_grid := self.create_input(taskvars, {"chosen_color": random.choice([taskvars['object_color1'], taskvars['object_color2']])})),
            'output': self.transform_input(input_grid, taskvars)
        }]

        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }