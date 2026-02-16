from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects
from Framework.input_library import create_object, retry, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskbe94b721(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} × {vars['m']}.",
            "Each grid contains a random number of objects.",
            "Every object has a unique color and a unique size.",
            "An object occupies k contiguous cells with 4-connectivity, where k ≥ 3.",
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
        
        # Generate 2-4 objects
        num_objects = random.randint(2, 5)
        
        # STEP 1: Pre-determine unique sizes BEFORE creating objects
        min_size = 3
        max_size = min(20, (n * m) // (num_objects * 3))  # Conservative max size
        
        target_sizes = []
        used_sizes = set()
        
        # Generate unique target sizes
        max_size_attempts = 100
        for _ in range(max_size_attempts):
            size = random.randint(min_size, max_size)
            if size not in used_sizes:
                target_sizes.append(size)
                used_sizes.add(size)
                if len(target_sizes) >= num_objects:
                    break
        
        # Ensure we have enough unique sizes
        if len(target_sizes) < 2:
            raise ValueError("Could not generate enough unique sizes")
        
        # Adjust num_objects if needed
        num_objects = len(target_sizes)
        
        # Sort sizes so we place smaller objects first (easier to fit)
        target_sizes.sort()
        
        # STEP 2: Create and place objects with target sizes
        used_colors = {0}
        placed_objects = []
        
        for obj_idx, target_size in enumerate(target_sizes):
            # Find available color
            available_colors = [c for c in range(1, 10) if c not in used_colors]
            if not available_colors:
                break
            color = random.choice(available_colors)
            used_colors.add(color)
            
            # Try to create and place object with approximately the target size
            max_attempts = 200
            placed = False
            
            for attempt in range(max_attempts):
                # Estimate dimensions based on target size
                # Try different aspect ratios for variety
                aspect_ratio = random.uniform(0.6, 1.8)
                estimated_height = max(2, int(np.sqrt(target_size / aspect_ratio)))
                estimated_width = max(2, int(np.sqrt(target_size * aspect_ratio)))
                
                # Clamp to reasonable bounds
                obj_height = min(n // 2, estimated_height)
                obj_width = min(m // 2, estimated_width)
                
                # Create object
                try:
                    obj_grid = create_object(obj_height, obj_width, color, Contiguity.FOUR, background=0)
                    actual_size = np.sum(obj_grid != 0)
                except:
                    continue
                
                # Check if size is acceptable
                if actual_size < 3:
                    continue
                
                # Check if this actual size conflicts with other placed objects
                if any(actual_size == size for _, size in placed_objects):
                    continue
                
                # Try to place it in the grid
                placement_attempts = 100
                for _ in range(placement_attempts):
                    if obj_grid.shape[0] >= n - 2 or obj_grid.shape[1] >= m - 2:
                        break
                    
                    # Random position with margin
                    max_start_row = n - obj_grid.shape[0] - 2
                    max_start_col = m - obj_grid.shape[1] - 2
                    
                    if max_start_row < 1 or max_start_col < 1:
                        break
                    
                    start_row = random.randint(1, max_start_row)
                    start_col = random.randint(1, max_start_col)
                    
                    # Check if placement area is clear with buffer
                    buffer = 1
                    check_row_start = max(0, start_row - buffer)
                    check_row_end = min(n, start_row + obj_grid.shape[0] + buffer)
                    check_col_start = max(0, start_col - buffer)
                    check_col_end = min(m, start_col + obj_grid.shape[1] + buffer)
                    
                    check_area = grid[check_row_start:check_row_end, check_col_start:check_col_end]
                    
                    if np.all(check_area == 0):
                        # Place the object
                        for r in range(obj_grid.shape[0]):
                            for c in range(obj_grid.shape[1]):
                                if obj_grid[r, c] != 0:
                                    grid[start_row + r, start_col + c] = obj_grid[r, c]
                        
                        placed = True
                        placed_objects.append((color, actual_size))
                        break
                
                if placed:
                    break
            
            if not placed:
                # If we have at least 2 objects already, we can stop
                if len(placed_objects) >= 2:
                    break
                else:
                    raise ValueError("Could not place minimum required objects")
        
        # Final check: ensure we have at least 2 objects with unique sizes
        if len(placed_objects) < 2:
            raise ValueError("Could not place minimum required objects")
        
        # Verify all placed objects have unique sizes
        actual_sizes = [size for _, size in placed_objects]
        if len(set(actual_sizes)) != len(actual_sizes):
            raise ValueError("Objects do not have unique sizes")
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        if len(objects) == 0:
            return np.array([[0]])
        
        # Find the largest object by number of cells
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
        # Generate task variables - use larger grids for better success
        taskvars = {
            'n': random.randint(10, 30),  
            'm': random.randint(10, 30),
        }
        
        # Generate training examples
        num_train = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train):
            # Try to create a valid input/output pair
            input_grid = retry(
                lambda: self.create_input(taskvars, {}),
                lambda g: np.sum(g != 0) >= 6,  
            )
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = retry(
            lambda: self.create_input(taskvars, {}),
            lambda g: np.sum(g != 0) >= 6,
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