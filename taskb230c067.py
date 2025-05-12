from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, retry

class Taskb230c067Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size 10x10.", 
            "The grid consists of three objects of {{color(\"object_color\")}} color, where two of the objects are exactly the same while the other one object is different"
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "Only the different object gets a new color, namely {{color(\"fill_color\")}} color. But similar objects retain their original color from the input grid."
        ]
        
        taskvars_definitions = {
            "object_color": "The color of the objects in the input grid (between 1 and 9)",
            "fill_color": "The color to fill the different object with (between 1 and 9, different from object_color)"
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, grid_size=10, object_color=1):
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create base object (will be duplicated)
        base_obj = create_object(
            height=random.randint(2, 3),
            width=random.randint(2, 3),
            color_palette=object_color,
            contiguity=Contiguity.FOUR
        )
        
        # Create different object (ensure it's different)
        while True:
            diff_obj = create_object(
                height=random.randint(2, 3),
                width=random.randint(2, 3),
                color_palette=object_color,
                contiguity=Contiguity.FOUR
            )
            if not np.array_equal(base_obj, diff_obj):
                break
        
        # Place objects with spacing
        positions = []
        for i in range(3):  # Need 3 positions
            while True:
                r = random.randint(0, grid_size - 4)
                c = random.randint(0, grid_size - 4)
                valid = True
                
                # Check overlap with existing positions
                for pr, pc in positions:
                    if abs(r - pr) < 4 and abs(c - pc) < 4:
                        valid = False
                        break
                
                if valid:
                    positions.append((r, c))
                    break
        
        # Place two identical objects and one different
        objects = [base_obj, base_obj.copy(), diff_obj]
        random.shuffle(objects)  # Randomize placement order
        
        for (r, c), obj in zip(positions, objects):
            h, w = obj.shape
            grid[r:r+h, c:c+w] = obj
        
        return grid

    def transform_input(self, input_grid, fill_color):
        output_grid = input_grid.copy()
        objects = find_connected_objects(input_grid, diagonal_connectivity=False)
        
        # Find the different object by comparing shapes
        object_arrays = []
        for obj in objects:
            arr = obj.to_array()
            arr = arr[~np.all(arr == 0, axis=1)]  # Remove empty rows
            arr = arr[:, ~np.all(arr == 0, axis=0)]  # Remove empty columns
            object_arrays.append((arr, obj))
        
        # Find the object that doesn't match any other
        for i, (arr1, obj1) in enumerate(object_arrays):
            matches = 0
            for j, (arr2, _) in enumerate(object_arrays):
                if i != j and np.array_equal(arr1, arr2):
                    matches += 1
            
            if matches == 0:  # This is our different object
                for r, c, _ in obj1:
                    output_grid[r, c] = fill_color
                break
        
        return output_grid
    
    def create_grids(self):
        # Choose random colors between 1-9
        object_color = random.randint(1, 9)  # Any color for objects
        fill_color = random.randint(1, 9)    # Any color for filling
        # Make sure colors are different
        while fill_color == object_color:
            fill_color = random.randint(1, 9)
            
        gridvars = {
            "object_color": object_color,
            "fill_color": fill_color
        }
        
        # Generate training pairs with these colors
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            input_grid = self.create_input(grid_size=10, object_color=gridvars["object_color"])
            output_grid = self.transform_input(input_grid, fill_color=gridvars["fill_color"])
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair using same colors
        test_input = self.create_input(grid_size=10, object_color=gridvars["object_color"])
        test_output = self.transform_input(test_input, fill_color=gridvars["fill_color"])
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)
