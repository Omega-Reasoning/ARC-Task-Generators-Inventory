from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, BorderBehavior

class ARCTask8efcae92Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid with dimension {vars['rows']} X {vars['rows']}.",
            "There are three to four, 4-way objects present in the input grid of which are either square or rectangle of color {color('in_color')}.",
            "Few cells in the 4-way connected objects are filled with color {color('cell_color')}.",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "First identify all the 4-way connected objects which are not monochromatic.",
            "Count the number of cells of color {color('cell_color')} in all the objects, extract the subgrid containing the object with highest number of these cells"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict, TrainTestData]:
        # Initialize task variables
        # Generate task variables
        rows = random.randint(20, 30)
        
        # Choose distinct colors for the objects and the cells to count
        color_options = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        in_color = random.choice(color_options)
        color_options.remove(in_color)
        cell_color = random.choice(color_options)
        
        taskvars = {
            'rows': rows,
            'in_color': in_color,
            'cell_color': cell_color
        }
        
        # Generate 3-4 training examples and 1 test example
        n_train = random.randint(3, 4)
        train_examples = []
        
        for _ in range(n_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_grid = self.create_input(taskvars, {})
        test_output = self.transform_input(test_grid, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_grid, 'output': test_output}]
        }
    
    def create_input(self, taskvars, gridvars) -> np.ndarray:
        rows = taskvars['rows']
        in_color = taskvars['in_color']
        cell_color = taskvars['cell_color']
        
        # Create empty grid
        grid = np.zeros((rows, rows), dtype=int)
        
        # Decide how many objects to create (3 or 4)
        num_objects = random.randint(3, 4)
        
        # Parameters for object generation
        min_size = 3
        max_size = 8
        
        # Keep track of occupied areas to avoid overlap
        occupied = set()
        
        objects_created = 0
        max_attempts = 100
        
        # Create objects with different counts of special colored cells
        cell_color_counts = []
        
        for i in range(num_objects):
            attempts = 0
            while attempts < max_attempts and objects_created < num_objects:
                # Generate random object size
                obj_height = random.randint(min_size, max_size)
                obj_width = random.randint(min_size, max_size)
                
                # Generate random position
                top = random.randint(0, rows - obj_height - 1)
                left = random.randint(0, rows - obj_width - 1)
                
                # Check if this area overlaps with existing objects or is adjacent
                area = {(r, c) for r in range(top-1, top+obj_height+1) 
                               for c in range(left-1, left+obj_width+1)}
                
                if not area.intersection(occupied):
                    # Create the object
                    obj = np.full((obj_height, obj_width), in_color)
                    
                    # Determine how many cells to color with cell_color
                    # Make sure each object has a different count
                    num_special_cells = random.randint(1, obj_height * obj_width // 3)
                    while num_special_cells in cell_color_counts:
                        num_special_cells = random.randint(1, obj_height * obj_width // 3)
                    
                    cell_color_counts.append(num_special_cells)
                    
                    # Randomly place special colored cells
                    coords = [(r, c) for r in range(obj_height) for c in range(obj_width)]
                    special_coords = random.sample(coords, num_special_cells)
                    
                    for r, c in special_coords:
                        obj[r, c] = cell_color
                    
                    # Place object in grid
                    grid[top:top+obj_height, left:left+obj_width] = obj
                    
                    # Update occupied area
                    for r in range(top, top+obj_height):
                        for c in range(left, left+obj_width):
                            occupied.add((r, c))
                    
                    objects_created += 1
                    break
                
                attempts += 1
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
        
        # Filter to get non-monochromatic objects
        non_mono_objects = [obj for obj in objects.objects if not obj.is_monochromatic]
        
        if not non_mono_objects:
            # Fallback if no non-monochromatic objects found
            return np.zeros((1, 1), dtype=int)
        
        # Count cells of the special color in each object
        cell_color = taskvars['cell_color']
        counts = []
        
        for obj in non_mono_objects:
            special_cells = sum(1 for _, _, color in obj.cells if color == cell_color)
            counts.append((obj, special_cells))
        
        # Find object with highest count
        best_obj, _ = max(counts, key=lambda x: x[1])
        
        # Extract the subgrid containing this object
        r_min, c_min = float('inf'), float('inf')
        r_max, c_max = -1, -1
        
        for r, c, _ in best_obj.cells:
            r_min = min(r_min, r)
            c_min = min(c_min, c)
            r_max = max(r_max, r)
            c_max = max(c_max, c)
        
        # Extract subgrid
        output = grid[r_min:r_max+1, c_min:c_max+1]
        return output