from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects
import numpy as np
import random

class Task4Vs2j37CoGVtM4QEgV7ig9Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains exactly two objects, where an object is a 4-way connected group of cells, of the same color.",
            "The two objects are of colors {color('object_color1')} and {color('object_color2')}.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and changing the color of both objects to {color('fill_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        size = random.randint(5, 30)
        grid = np.zeros((size, size), dtype=int)
        
        # Randomly select object colors and ensure they are different
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        
        # Place the first object
        object1 = create_object(
            random.randint(2, size//2), 
            random.randint(2, size//2), 
            object_color1, 
            contiguity=Contiguity.FOUR
        )
        
        r1, c1 = random.randint(0, size - object1.shape[0]), random.randint(0, size - object1.shape[1])
        grid[r1:r1 + object1.shape[0], c1:c1 + object1.shape[1]] += object1
        
        # Place the second object ensuring separation
        while True:
            object2 = create_object(
                random.randint(2, size//2), 
                random.randint(2, size//2), 
                object_color2, 
                contiguity=Contiguity.FOUR
            )
            r2, c2 = random.randint(0, size - object2.shape[0]), random.randint(0, size - object2.shape[1])
            
            temp_grid = grid.copy()
            temp_grid[r2:r2 + object2.shape[0], c2:c2 + object2.shape[1]] += object2
            
            if len(find_connected_objects(temp_grid, diagonal_connectivity=False).objects) == 2:
                grid = temp_grid
                break
        
        return grid

    def transform_input(self, grid, taskvars):
        fill_color = taskvars['fill_color']
        output_grid = grid.copy()
        
        objects = find_connected_objects(grid, diagonal_connectivity=False)
        for obj in objects:
            for r, c, _ in obj.cells:
                output_grid[r, c] = fill_color
        
        return output_grid

    def create_grids(self):
        # Randomize task variables
        object_color1 = random.randint(1, 9)
        object_color2 = random.choice([x for x in range(1, 10) if x != object_color1])
        fill_color = random.choice([x for x in range(1, 10) if x not in [object_color1, object_color2]])

        taskvars = {
            'object_color1': object_color1,
            'object_color2': object_color2,
            'fill_color': fill_color
        }
        
        # Generate train and test grids
        grids = self.create_grids_default(nr_train_examples=random.randint(3, 6), nr_test_examples=1, taskvars=taskvars)
        return taskvars, grids
