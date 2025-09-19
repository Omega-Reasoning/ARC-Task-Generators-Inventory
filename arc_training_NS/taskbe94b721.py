from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import create_object, retry, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskbe94b721(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} × {vars['m']}.",
            "Each grid contains a random number of objects.",
            "Every object has a unique color and a unique size.",
            "An object occupies k contiguous cells with 4-connectivity, where k ∈ [3, 8].",
            "Objects do not touch, they share no edges or corners."
        ]
        
        transformation_reasoning_chain = [
            "The object with the largest size (most cells) is identified from the input grid.",
            "The output grid is constructed so that it encloses only this object, with no extra rows or columns beyond what is required."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n, m = taskvars['n'], taskvars['m']
        grid = np.zeros((n, m), dtype=int)
        
        # Generate 2-5 objects with unique colors and sizes
        num_objects = random.randint(2, 5)
        used_colors = {0}  # background is already used
        used_sizes = set()
        
        # Generate unique sizes for each object
        available_sizes = list(range(3, 9))  # sizes 3-8
        random.shuffle(available_sizes)
        object_sizes = available_sizes[:num_objects]
        
        placed_objects = []
        
        for target_size in object_sizes:
            # Find available color
            available_colors = [c for c in range(1, 10) if c not in used_colors]
            if not available_colors:
                break
            color = random.choice(available_colors)
            used_colors.add(color)
            
            # Try to place object with exact target size
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                # Generate object of roughly the right size
                obj_height = random.randint(2, min(6, n-2))
                obj_width = random.randint(2, min(6, m-2))
                
                # Create object and check its size
                obj_grid = create_object(obj_height, obj_width, color, Contiguity.FOUR, background=0)
                actual_size = np.sum(obj_grid != 0)
                
                # Accept if size is exactly what we want
                if actual_size == target_size:
                    # Try to place it in the grid without overlapping
                    placement_attempts = 50
                    for _ in range(placement_attempts):
                        # Random position
                        start_row = random.randint(1, n - obj_height - 1)
                        start_col = random.randint(1, m - obj_width - 1)
                        
                        # Check if placement area is clear and has buffer space
                        buffer = 1
                        check_area = grid[max(0, start_row-buffer):min(n, start_row+obj_height+buffer),
                                        max(0, start_col-buffer):min(m, start_col+obj_width+buffer)]
                        
                        if np.all(check_area == 0):
                            # Place the object
                            for r in range(obj_height):
                                for c in range(obj_width):
                                    if obj_grid[r, c] != 0:
                                        grid[start_row + r, start_col + c] = obj_grid[r, c]
                            placed = True
                            placed_objects.append((color, target_size))
                            break
                    
                    if placed:
                        break
            
            if not placed and len(placed_objects) >= 2:
                # If we can't place this object but have at least 2, continue
                continue
        
        # Ensure we have at least 2 objects with different sizes
        if len(placed_objects) < 2:
            raise ValueError("Could not place minimum required objects")
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        if len(objects) == 0:
            return np.array([[0]])
        
        # Find the largest object
        largest_object = max(objects.objects, key=lambda obj: len(obj))
        
        # Get bounding box of the largest object
        bounding_box = largest_object.bounding_box
        
        # Extract the region containing only the largest object
        extracted_region = grid[bounding_box[0], bounding_box[1]]
        
        # Create output grid with only the largest object
        output = np.zeros_like(extracted_region)
        
        # Copy only the largest object to the output
        for r, c, color in largest_object.cells:
            # Convert to local coordinates within the bounding box
            local_r = r - bounding_box[0].start
            local_c = c - bounding_box[1].start
            output[local_r, local_c] = color
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'n': random.randint(8, 30),  # Grid height
            'm': random.randint(8, 30),  # Grid width
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Try to create a valid input/output pair
            input_grid = retry(
                lambda: self.create_input(taskvars, {}),
                lambda g: np.sum(g != 0) >= 6  # At least 6 non-background cells
            )
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = retry(
            lambda: self.create_input(taskvars, {}),
            lambda g: np.sum(g != 0) >= 6
        )
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

