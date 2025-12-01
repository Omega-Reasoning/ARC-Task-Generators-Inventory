from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, GridObjects, find_connected_objects
from input_library import create_object, retry, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task2013d3e2Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} × {vars['n']}.",
            "They contain a single 4-fold symmetric object, which is a rotation of a square of size {vars['m']} × {vars['m']}.",
            "The position of the object varies in each input grid.",
            "The object may consist of a random number of colors, with at least two colors.",
            "The cells in the object have 8-connectivity."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is of size {vars['m']} × {vars['m']}.",
            "The {vars['quadrant_name']} quadrant of the object is extracted and presented as the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        m = taskvars['m']
        colors = gridvars['colors']
        offset_r = gridvars['offset_r']
        offset_c = gridvars['offset_c']
        
        # Create base quadrant (top-left) - this will be quadrant 1
        # The quadrant size is m × m (no contiguity requirement here)
        quadrant = retry(
            lambda: create_object(m, m, colors, Contiguity.NONE, background=0),
            lambda obj: len(np.unique(obj[obj != 0])) >= 2  # at least 2 colors
        )
        
        # Create 4-fold rotationally symmetric object (size will be 2m × 2m)
        full_object = np.zeros((2*m, 2*m), dtype=int)
        
        # Place quadrant 1 (top-left)
        full_object[:m, :m] = quadrant
        
        # Create quadrant 2 (top-right) - 90° rotation
        full_object[:m, m:] = np.rot90(quadrant, k=-1)
        
        # Create quadrant 3 (bottom-right) - 180° rotation
        full_object[m:, m:] = np.rot90(quadrant, k=2)
        
        # Create quadrant 4 (bottom-left) - 270° rotation
        full_object[m:, :m] = np.rot90(quadrant, k=1)
        
        # Verify that the full object has 8-connectivity
        # Find connected components with 8-connectivity
        from scipy.ndimage import label
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled_array, num_features = label(full_object != 0, structure=structure)
        
        # If not connected as a single object, retry
        if num_features != 1:
            # This shouldn't happen often, but we handle it by returning empty
            # The retry in create_grids will handle regeneration
            pass
        
        # Place in larger grid
        grid = np.zeros((n, n), dtype=int)
        grid[offset_r:offset_r+2*m, offset_c:offset_c+2*m] = full_object
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        m = taskvars['m']
        quadrant_id = taskvars['quadrant']
        
        # Find all non-zero cells
        non_zero = np.argwhere(grid != 0)
        if len(non_zero) == 0:
            return np.zeros((m, m), dtype=int)
        
        # Get bounding box
        min_r, min_c = non_zero.min(axis=0)
        max_r, max_c = non_zero.max(axis=0)
        
        # Extract the full object
        full_object = grid[min_r:max_r+1, min_c:max_c+1]
        
        # Ensure it's the right size (pad if necessary)
        if full_object.shape[0] < 2*m or full_object.shape[1] < 2*m:
            padded = np.zeros((2*m, 2*m), dtype=int)
            padded[:full_object.shape[0], :full_object.shape[1]] = full_object
            full_object = padded
        
        # Extract the appropriate quadrant
        if quadrant_id == 1:  # top-left
            quadrant = full_object[:m, :m]
        elif quadrant_id == 2:  # top-right
            quadrant = full_object[:m, m:2*m]
        elif quadrant_id == 3:  # bottom-right
            quadrant = full_object[m:2*m, m:2*m]
        else:  # quadrant_id == 4, bottom-left
            quadrant = full_object[m:2*m, :m]
        
        return quadrant
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Task variables
        n = random.randint(10, 30)  # Grid size
        m = random.randint(2, 7)  # Quadrant size (full object will be 2m × 2m)
        
        # Ensure the full object fits in the grid
        while 2*m > n - 2:
            m -= 1
        
        # Choose which quadrant to extract (1-4)
        quadrant = random.randint(1, 4)
        quadrant_names = ["top-left", "top-right", "bottom-right", "bottom-left"]
        quadrant_name = quadrant_names[quadrant - 1]
        
        taskvars = {
            'n': n,
            'm': m,
            'quadrant': quadrant,
            'quadrant_name': quadrant_name
        }
        
        # Generate training examples with varying colors
        num_train = random.randint(3, 6)
        train_pairs = []
        
        all_colors = list(range(1, 10))
        
        for _ in range(num_train):
            # Each training example gets different colors (2-5 colors)
            num_colors = random.randint(2, 5)
            colors = random.sample(all_colors, num_colors)
            
            # Retry until we get a grid with 8-connected object
            def create_valid_grid():
                gridvars = {
                    'colors': colors,
                    'offset_r': random.randint(0, n - 2*m),
                    'offset_c': random.randint(0, n - 2*m)
                }
                return self.create_input(taskvars, gridvars)
            
            def is_8_connected(grid):
                from scipy.ndimage import label
                structure = np.ones((3, 3), dtype=int)
                labeled_array, num_features = label(grid != 0, structure=structure)
                return num_features == 1
            
            input_grid = retry(create_valid_grid, is_8_connected)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example with different colors
        num_colors_test = random.randint(2, 5)
        colors_test = random.sample(all_colors, num_colors_test)
        
        def create_valid_test_grid():
            gridvars_test = {
                'colors': colors_test,
                'offset_r': random.randint(0, n - 2*m),
                'offset_c': random.randint(0, n - 2*m)
            }
            return self.create_input(taskvars, gridvars_test)
        
        def is_8_connected(grid):
            from scipy.ndimage import label
            structure = np.ones((3, 3), dtype=int)
            labeled_array, num_features = label(grid != 0, structure=structure)
            return num_features == 1
        
        test_input = retry(create_valid_test_grid, is_8_connected)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_pairs, 'test': test_pairs}

