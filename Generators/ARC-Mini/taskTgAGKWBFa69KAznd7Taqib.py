from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, Contiguity
import numpy as np
import random

class TaskTgAGKWBFa69KAznd7TaqibGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain colored objects, where an object is a 4-way connected group of cells, of the same color.",
            "The colors of these objects are {color('object_color1')}, {color('object_color2')}, {color('object_color3')} and {color('object_color4')}.",
            "The remaining cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling the empty (0) cells with {color('fill_color')} color."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        # Choose a more reasonable grid size to avoid placement issues
        grid_size = random.randint(10, 30), random.randint(10, 30)
        grid = np.zeros(grid_size, dtype=int)

        colors = [
            taskvars['object_color1'],
            taskvars['object_color2'],
            taskvars['object_color3'],
            taskvars['object_color4']
        ]

        # Limit the number of placement attempts to avoid infinite loops
        max_attempts = 100
        
        for color in colors:
            # Make sure objects aren't too large for the grid
            max_obj_size = min(grid.shape) // 3
            obj_height = random.randint(1, min(5, max_obj_size))
            obj_width = random.randint(1, min(5, max_obj_size))
            
            obj = create_object(obj_height, obj_width, color_palette=color, contiguity=Contiguity.FOUR)

            # Limit the number of placement attempts for each object
            attempts = 0
            placed = False
            
            while not placed and attempts < max_attempts:
                attempts += 1
                row = random.randint(0, grid.shape[0] - obj.shape[0])
                col = random.randint(0, grid.shape[1] - obj.shape[1])
                
                # Check if the region is empty
                if np.all(grid[row:row+obj.shape[0], col:col+obj.shape[1]] == 0):
                    grid[row:row+obj.shape[0], col:col+obj.shape[1]] = obj
                    placed = True
            
            # If we couldn't place the object after max attempts, just continue
            # This ensures we don't get stuck in an infinite loop
            if not placed:
                continue

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()
        output_grid[output_grid == 0] = taskvars['fill_color']
        return output_grid

    def create_grids(self) -> tuple:
        taskvars = {
            'object_color1': random.randint(1, 9),
            'object_color2': random.randint(1, 9),
            'object_color3': random.randint(1, 9),
            'object_color4': random.randint(1, 9),
            'fill_color': random.randint(1, 9)
        }

        # Ensure colors are unique
        while len(set(taskvars.values())) < len(taskvars):
            taskvars = {
                'object_color1': random.randint(1, 9),
                'object_color2': random.randint(1, 9),
                'object_color3': random.randint(1, 9),
                'object_color4': random.randint(1, 9),
                'fill_color': random.randint(1, 9)
            }

        num_train = random.randint(3, 4)
        train_grids = [
            {
                'input': (input_grid := self.create_input(taskvars, {})),
                'output': self.transform_input(input_grid, taskvars)
            } for _ in range(num_train)
        ]

        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)

        train_test_data = {
            'train': train_grids,
            'test': [{'input': test_input, 'output': test_output}]
        }

        return taskvars, train_test_data

