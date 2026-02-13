from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class TaskNT5FZfntNtBNwfxaB5AjYYGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain multiple {color('object_color')} objects, 4-way connected cells of the same color, along with several single {color('object_color')} cells.",
            "The remaining cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling the empty cells with {color('fill_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        height, width = random.randint(5, 30), random.randint(5, 30)
        object_color = taskvars['object_color']
        grid = np.zeros((height, width), dtype=int)
        
        # Create multiple connected objects
        num_objects = random.randint(2, 6)
        for _ in range(num_objects):
            obj_height, obj_width = random.randint(2, 5), random.randint(2, 5)
            obj = create_object(obj_height, obj_width, object_color, Contiguity.FOUR)
            r, c = random.randint(0, height - obj.shape[0]), random.randint(0, width - obj.shape[1])
            
            obj_indices = np.where(obj != 0)
            for i in range(len(obj_indices[0])):
                grid[r + obj_indices[0][i], c + obj_indices[1][i]] = object_color
        
        # Add some scattered single object_color cells
        num_singles = random.randint(2, 5)
        for _ in range(num_singles):
            r, c = random.randint(0, height - 1), random.randint(0, width - 1)
            grid[r, c] = object_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        fill_color = taskvars['fill_color']
        transformed_grid = grid.copy()
        transformed_grid[grid == 0] = fill_color
        return transformed_grid
    
    def create_grids(self):
        # Select object and fill colors ensuring they are different
        object_color, fill_color = random.sample(range(1, 10), 2)
        taskvars = {'object_color': object_color, 'fill_color': fill_color}
        
        num_train = random.randint(3, 6)
        train_data = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}


