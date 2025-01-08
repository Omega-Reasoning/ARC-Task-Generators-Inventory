from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import retry
import numpy as np
import random

class TaskRCdwdHBGotnBYezKj6t6amGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
           "All input grids are of size {vars['n']}x{vars['n']}.",
           "Each input grid contains a square frame of size {vars['n']-2}x{vars['n']-2}, which is one cell wide and contains empty (0) cells both inside and outside the frame.",
           "It is placed centrally in the grid, such that the first and the last, rows and columns are empty (0).",
           "The square frame is either {color('object_color1')} or {color('object_color2')} in color."
        ]
        reasoning_chain = [ 
           "To construct the output grid, copy the input grid and apply the following transformation.",
           "The left half of the frame retains its original color.",
           "The right half of the frame is transformed to a new color, based on the color of the input grid: {color('object_color1')} becomes {color('object_color3')} and {color('object_color2')} becomes {color('object_color4')}."
        ]
        super().__init__(observation_chain, reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        n = taskvars["n"]
        chosen_color = gridvars["chosen_color"]
        grid = np.zeros((n, n), dtype=int)

        for c in range(1, n - 1):
            grid[1, c] = chosen_color
            grid[n - 2, c] = chosen_color
        for r in range(1, n - 1):
            grid[r, 1] = chosen_color
            grid[r, n - 2] = chosen_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars=None) -> np.ndarray:
                
        n = grid.shape[0]
        mid_col = n // 2
        output = grid.copy()

        # Transform the right half of the square frame
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                # Horizontal top and bottom borders
                if (r == 1 or r == n - 2) and c >= mid_col:
                    if output[r, c] == taskvars['object_color1']:
                        output[r, c] = taskvars['object_color3']
                    elif output[r, c] == taskvars['object_color2']:
                        output[r, c] = taskvars['object_color4']
                
                # Vertical right border
                if c == n - 2:
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
            "n": 2 * random.randint(3, 9)
        }

        nr_train = random.randint(3, 5)
        
        # Generate n-1 training examples randomly
        train_examples = []
        colors_used = set()
        
        for _ in range(nr_train - 1):
            chosen_color = random.choice([taskvars['object_color1'], taskvars['object_color2']])
            colors_used.add(chosen_color)
            train_examples.append({
                'input': (input_grid := self.create_input(taskvars, {"chosen_color": chosen_color})),
                'output': self.transform_input(input_grid, taskvars)
            })
        
        # For the last example, use the unused color if necessary
        if len(colors_used) == 1:
            unused_color = taskvars['object_color1'] if taskvars['object_color2'] in colors_used else taskvars['object_color2']
            chosen_color = unused_color
        else:
            chosen_color = random.choice([taskvars['object_color1'], taskvars['object_color2']])
        
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