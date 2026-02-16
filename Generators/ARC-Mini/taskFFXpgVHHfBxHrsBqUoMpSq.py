from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import create_object, Contiguity
import numpy as np
import random

class TaskFFXpgVHHfBxHrsBqUoMpSqGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains some {color('object_color1')} and {color('object_color2')} cells, with the remaining cells being empty (0)."
        ]

        reasoning_chain = [
            "To construct the output grid, copy the input grid and change the color of {color('object_color1')} cells to {color('object_color2')} and {color('object_color2')} cells to {color('object_color1')}."
        ]
        
        super().__init__(observation_chain, reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        size = random.randint(8, 30)
        grid = np.zeros((size, size), dtype=int)
        
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']

        def place_object(color):
            obj_height = random.randint(2, size // 2)
            obj_width = random.randint(2, size // 2)
            obj = create_object(obj_height, obj_width, color, contiguity=Contiguity.EIGHT)
            
            r_offset = random.randint(0, size - obj_height)
            c_offset = random.randint(0, size - obj_width)
            grid[r_offset:r_offset + obj_height, c_offset:c_offset + obj_width] = obj
        
        place_object(object_color1)
        place_object(object_color2)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']

        transformed_grid = grid.copy()
        transformed_grid[grid == object_color1] = object_color2
        transformed_grid[grid == object_color2] = object_color1

        return transformed_grid

    def create_grids(self) -> tuple:
        object_color1 = random.randint(1, 9)
        object_color2 = random.randint(1, 9)
        while object_color2 == object_color1:
            object_color2 = random.randint(1, 9)

        taskvars = {
            'object_color1': object_color1,
            'object_color2': object_color2
        }

        num_train = random.randint(3, 6)
        num_test = 1

        train_test_data = self.create_grids_default(num_train, num_test, taskvars)

        return taskvars, train_test_data

