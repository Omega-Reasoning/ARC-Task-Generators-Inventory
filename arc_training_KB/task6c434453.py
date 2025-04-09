from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple
from input_library import Contiguity, create_object, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task6c434453Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain several {color('object_color')} objects, each made of 4-way connected cells of the same color, while all remaining cells are empty (0).",
            "One or two objects have a cross shape, which is defined as [[0,c,0], [c,c,c], [0,c,0]], for a color c.",
            "Some are one-cell wide 3x3 square frames with an empty (0) interior.",
            "The remaining objects should have simple shapes, such as a 1x2 rectangle or a 2x2 square block.",
            "All objects are completely separated from one another."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying all {color('object_color')} one-cell wide 3x3 square frames.",
            "Once identified, remove these frames and in their exact same 3x3 subgrid locations, place {color('newobject_color')} cross shapes.",
            "The cross shape is defined as: [[0,c,0], [c,c,c], [0,c,0]], for a color c.",
            "All other cells remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        grid_size = random.randint(10, 20)
        object_color = random.randint(1, 9)
        
        # Ensure newobject_color is different from object_color
        colors = list(range(1, 10))
        colors.remove(object_color)
        newobject_color = random.choice(colors)
        
        taskvars = {
            'grid_size': grid_size,
            'object_color': object_color,
            'newobject_color': newobject_color
        }
        
        # Create 3-4 train examples and 1 test example
        num_train_examples = random.randint(3, 4)
        
        train_examples = []
        
        # Track how many train examples have 1 or 2 cross shapes
        one_cross_created = False
        two_cross_created = False
        
        for i in range(num_train_examples):
            # Determine whether this grid should have 1 or 2 crosses
            if not one_cross_created:
                num_crosses = 1
                one_cross_created = True
            elif not two_cross_created:
                num_crosses = 2
                two_cross_created = True
            else:
                num_crosses = random.choice([1, 2])
            
            # Calculate number of frames based on grid size
            num_frames = grid_size // 5
            
            # Create grid variables for this specific example
            gridvars = {
                'num_crosses': num_crosses,
                'num_frames': num_frames,
                'num_other_objects': random.randint(2, 4)
            }
            
            # Create input grid and transform it
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with randomly chosen parameters
        test_gridvars = {
            'num_crosses': random.choice([1, 2]),
            'num_frames': grid_size // 5,
            'num_other_objects': random.randint(2, 4)
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        num_crosses = gridvars['num_crosses']
        num_frames = gridvars['num_frames']
        num_other_objects = gridvars['num_other_objects']
        
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Define object shapes
        cross_shape = np.array([
            [0, object_color, 0],
            [object_color, object_color, object_color],
            [0, object_color, 0]
        ])
        
        frame_shape = np.array([
            [object_color, object_color, object_color],
            [object_color, 0, object_color],
            [object_color, object_color, object_color]
        ])
        
        # Define possible simple shapes
        simple_shapes = [
            np.array([[object_color, object_color]]),  # 1x2 rectangle
            np.array([[object_color], [object_color]]),  # 2x1 rectangle
            np.array([[object_color, object_color], [object_color, object_color]])  # 2x2 square
        ]
        
        # Place objects ensuring they don't overlap
        def place_object(shape, attempts=100):
            for _ in range(attempts):
                # Find a random position where shape fits
                max_row = grid.shape[0] - shape.shape[0]
                max_col = grid.shape[1] - shape.shape[1]
                
                if max_row <= 0 or max_col <= 0:
                    continue
                
                row = random.randint(0, max_row)
                col = random.randint(0, max_col)
                
                # Check if the area and its border is clear
                region_height = shape.shape[0] + 2
                region_width = shape.shape[1] + 2
                
                # Ensure we stay within grid bounds for checking
                if (row > 0 and col > 0 and 
                    row + shape.shape[0] < grid.shape[0] and 
                    col + shape.shape[1] < grid.shape[1]):
                    
                    # Check a region that includes a 1-cell border around the shape
                    region_row_start = max(0, row - 1)
                    region_row_end = min(grid.shape[0], row + shape.shape[0] + 1)
                    region_col_start = max(0, col - 1)
                    region_col_end = min(grid.shape[1], col + shape.shape[1] + 1)
                    
                    region = grid[region_row_start:region_row_end, region_col_start:region_col_end]
                    
                    if np.all(region == 0):  # If all cells are empty
                        # Place the shape
                        grid[row:row+shape.shape[0], col:col+shape.shape[1]] = shape
                        return True
            
            return False
        
        # Place cross shapes
        for _ in range(num_crosses):
            if not place_object(cross_shape):
                # If we can't place crosses, retry with a new grid
                return self.create_input(taskvars, gridvars)
        
        # Place frame shapes
        for _ in range(num_frames):
            if not place_object(frame_shape):
                # If we can't place frames, retry with a new grid
                return self.create_input(taskvars, gridvars)
        
        # Place other simple shapes
        for _ in range(num_other_objects):
            shape = random.choice(simple_shapes)
            if not place_object(shape):
                # It's okay if we can't place all simple shapes
                break
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        object_color = taskvars['object_color']
        newobject_color = taskvars['newobject_color']
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # Define the cross shape
        cross = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])
        
        # Define the frame shape pattern
        frame = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        
        # Identify and transform 3x3 frames to crosses
        for obj in objects:
            if obj.has_color(object_color) and obj.height == 3 and obj.width == 3:
                obj_array = obj.to_array()
                
                # Check if the object is a 3x3 frame
                is_frame = True
                for r in range(3):
                    for c in range(3):
                        if (frame[r, c] == 1 and obj_array[r, c] != object_color) or \
                           (frame[r, c] == 0 and obj_array[r, c] != 0):
                            is_frame = False
                            break
                
                if is_frame:
                    # Get the position where the frame starts
                    r_start, c_start = obj.bounding_box[0].start, obj.bounding_box[1].start
                    
                    # Remove the frame
                    obj.cut(output_grid)
                    
                    # Place a cross in the same position
                    for r in range(3):
                        for c in range(3):
                            if cross[r, c] == 1:
                                output_grid[r_start + r, c_start + c] = newobject_color
        
        return output_grid

