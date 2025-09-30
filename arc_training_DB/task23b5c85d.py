from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects

class ARCTask23b5c85dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "There are multiple 4-way connected objects which are either a square or a rectangle of varying sizes, each of these objects have different colors(between 1-9).",
            "Each of these objects have different areas,i.e,number of cells of the same color.",
            "The objects with smaller area overlap with the objects with larger area in some examples.",
            "The remaining cells of the input grid are empty(0)."
        ]
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "First, identify all the 4-way objects in the input grid of different colors.",
            "Determine the object with the least area amongst the objects, identify the subgrid which contains this object.",
            "The above subgrid is the output grid."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        num_objects = random.randint(2, 5)
        colors = random.sample(range(1, 10), num_objects)
        
        def generate():
            input_grid = np.zeros((rows, cols), dtype=int)
            areas = {}  # Changed to dict to store area for each color
            
            # First generate all shapes and their dimensions
            shapes = []
            for color in colors:
                while True:
                    shape_type = random.choice(['square', 'rectangle'])
                    if shape_type == 'square':
                        max_size = min(rows, cols)
                        size = random.randint(1, max_size)
                        w = h = size
                    else:
                        w = random.randint(1, cols)
                        h = random.randint(1, rows)
                        while w == h:
                            if random.choice([True, False]):
                                w = random.randint(1, cols)
                            else:
                                h = random.randint(1, rows)
                    area = w * h
                    if area not in areas.values():
                        areas[color] = area
                        shapes.append((color, w, h, area))
                        break
            
            # Sort shapes by area (largest first)
            shapes.sort(key=lambda x: x[3], reverse=True)
            
            # Place shapes in order (largest to smallest)
            for color, w, h, area in shapes:
                attempts = 100
                while attempts > 0:
                    x = random.randint(0, rows - h)
                    y = random.randint(0, cols - w)
                    
                    # Check if placement is valid
                    region = input_grid[x:x+h, y:y+w]
                    overlap_colors = set(region.flatten()) - {0}
                    
                    # Only allow placement if:
                    # 1. No overlap (empty space)
                    # 2. Or overlapping with larger area objects only
                    valid = True
                    for overlap_color in overlap_colors:
                        if areas[overlap_color] <= area:  # If overlapping with same or smaller area
                            valid = False
                            break
                    
                    if valid:
                        input_grid[x:x+h, y:y+w] = color
                        break
                    attempts -= 1
                
                if attempts == 0:  # If couldn't place after max attempts
                    return None
                    
            return input_grid
        
        input_grid = retry(generate, lambda grid: grid is not None, max_attempts=100)
        return input_grid
    
    def transform_input(self, grid, taskvars):
        print("grid: ", grid)
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        #print("objects: ", objects)
        if not objects:
            return grid.copy()
        min_area_obj = min(objects, key=lambda obj: len(obj))
        row_slice, col_slice = min_area_obj.bounding_box
        output_grid = grid[row_slice, col_slice].copy()
        return output_grid
    
    def create_grids(self):
        rows = random.randint(10, 30)
        cols = random.randint(10, 30)
        taskvars = {'rows': rows, 'cols': cols}
        nr_train = random.randint(3, 4)
        nr_test = 1
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        return taskvars, train_test_data
