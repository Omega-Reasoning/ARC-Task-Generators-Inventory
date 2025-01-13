from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import Contiguity, create_object, enforce_object_height, enforce_object_width, retry
import numpy as np
import random

class TaskfQBfDBScsibpM2ASDDAHZEGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain only a single 4-way connected object, which is a 2x2 block, with all remaining cells being empty (0).",
            "The color of this block can only be either {color('object_color1')} or {color('object_color2')}.",
            "The position of the block may vary between examples."
        ]

        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid.",
            "Change the color of the object according to the color of the object in the input grid.",
            "If the object in the input matrix is {color('object_color1')}, change it to {color('new_color1')}.",
            "If the object in the input matrix is {color('object_color2')}, change it to {color('new_color2')}."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        n = taskvars["n"]
        chosen_color = gridvars["chosen_color"]
        grid = np.zeros((n, n), dtype=int)
        
        # Place the 2x2 block randomly in the grid
        start_row = random.randint(0, n-2)
        start_col = random.randint(0, n-2)
        grid[start_row:start_row+2, start_col:start_col+2] = chosen_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output = grid.copy()
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        new_color1 = taskvars['new_color1']
        new_color2 = taskvars['new_color2']

        # Transform the color of the 2x2 block
        if object_color1 in grid:
            output[output == object_color1] = new_color1
        elif object_color2 in grid:
            output[output == object_color2] = new_color2

        return output

    def create_grids(self):
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        obj_col1, obj_col2, new_col1, new_col2 = all_colors[:4]

        taskvars = {
            "object_color1": obj_col1,
            "object_color2": obj_col2,
            "new_color1": new_col1,
            "new_color2": new_col2,
            "n": random.randint(3, 9)
        }

        nr_train = random.randint(3, 5)
        train_examples = []
        colors_used = set()
        
        for _ in range(nr_train - 1):
            chosen_color = random.choice([taskvars['object_color1'], taskvars['object_color2']])
            colors_used.add(chosen_color)
            input_grid = self.create_input(taskvars, {"chosen_color": chosen_color})
            train_examples.append({
                'input': input_grid,
                'output': self.transform_input(input_grid, taskvars)
            })
        
        if len(colors_used) == 1:
            unused_color = taskvars['object_color1'] if taskvars['object_color2'] in colors_used else taskvars['object_color2']
            chosen_color = unused_color
        else:
            chosen_color = random.choice([taskvars['object_color1'], taskvars['object_color2']])
        
        input_grid = self.create_input(taskvars, {"chosen_color": chosen_color})
        train_examples.append({
            'input': input_grid,
            'output': self.transform_input(input_grid, taskvars)
        })
        
        test_examples = []
        for color in [taskvars['object_color1'], taskvars['object_color2']]:
            input_grid = self.create_input(taskvars, {"chosen_color": color})
            test_examples.append({
                'input': input_grid,
                'output': self.transform_input(input_grid, taskvars)
            })

        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }