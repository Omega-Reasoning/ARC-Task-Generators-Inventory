from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, retry

class Taskb230c067Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size 10x10.", 
            "The grid consists of three objects of {color('object_color')}, where two of the objects are exactly the same while the other one object is different"
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The two identical objects get a new color, namely {color('same_color')}.",
            "The different object gets another new color, namely {color('different_color')}.",
            "All objects change from their original {color('object_color')} to their respective new colors."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def objects_are_identical(self, obj1_array, obj2_array):
        """Check if two object arrays are identical in shape"""
        # Normalize both arrays by removing empty rows/columns
        def normalize_array(arr):
            if arr.size == 0:
                return arr
            non_zero_rows = np.any(arr != 0, axis=1)
            non_zero_cols = np.any(arr != 0, axis=0)
            if not np.any(non_zero_rows) or not np.any(non_zero_cols):
                return np.array([[]])
            return arr[non_zero_rows][:, non_zero_cols]
        
        norm_obj1 = normalize_array(obj1_array)
        norm_obj2 = normalize_array(obj2_array)
        
        return norm_obj1.shape == norm_obj2.shape and np.array_equal(norm_obj1, norm_obj2)
    
    def create_different_object(self, base_obj, object_color):
        """Create an object that is guaranteed to be different from base_obj"""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Try creating a completely new object
            diff_obj = create_object(
                height=random.randint(2, 4),
                width=random.randint(2, 4),
                color_palette=object_color,
                contiguity=Contiguity.FOUR
            )
            
            if not self.objects_are_identical(base_obj, diff_obj):
                return diff_obj
        
        # If we can't create a different object, modify the base object
        diff_obj = base_obj.copy()
        
        # Strategy 1: Add a cell
        if diff_obj.shape[0] < 4 and diff_obj.shape[1] < 4:
            # Try to extend the object
            new_obj = np.zeros((diff_obj.shape[0] + 1, diff_obj.shape[1] + 1), dtype=int)
            new_obj[:diff_obj.shape[0], :diff_obj.shape[1]] = diff_obj
            new_obj[-1, -1] = object_color  # Add a corner cell
            if not self.objects_are_identical(base_obj, new_obj):
                return new_obj
        
        # Strategy 2: Remove a cell (if object has more than 1 cell)
        non_zero_positions = np.where(diff_obj != 0)
        if len(non_zero_positions[0]) > 1:
            idx = random.randint(0, len(non_zero_positions[0]) - 1)
            r, c = non_zero_positions[0][idx], non_zero_positions[1][idx]
            diff_obj[r, c] = 0
            if not self.objects_are_identical(base_obj, diff_obj):
                return diff_obj
        
        # Strategy 3: Add a cell to existing structure
        zero_positions = np.where(diff_obj == 0)
        if len(zero_positions[0]) > 0:
            idx = random.randint(0, len(zero_positions[0]) - 1)
            r, c = zero_positions[0][idx], zero_positions[1][idx]
            diff_obj[r, c] = object_color
            if not self.objects_are_identical(base_obj, diff_obj):
                return diff_obj
        
        # Strategy 4: Create a simple different shape as last resort
        if base_obj.shape == (2, 2):
            # Make a 3x1 line
            return np.array([[object_color], [object_color], [object_color]])
        else:
            # Make a 2x2 square
            return np.array([[object_color, object_color], [object_color, object_color]])
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a 10x10 grid with 3 objects: 2 identical and 1 different."""
        object_color = taskvars["object_color"]
        grid_size = 10
        
        max_grid_attempts = 20
        for grid_attempt in range(max_grid_attempts):
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Create base object (will be duplicated)
            base_obj = create_object(
                height=random.randint(2, 3),
                width=random.randint(2, 3),
                color_palette=object_color,
                contiguity=Contiguity.FOUR
            )
            
            # Create different object
            diff_obj = self.create_different_object(base_obj, object_color)
            
            # Prepare objects: 2 identical + 1 different
            objects_to_place = [base_obj, base_obj.copy(), diff_obj]
            random.shuffle(objects_to_place)
            
            # Find non-overlapping positions
            positions = []
            max_placement_attempts = 100
            
            for i, obj in enumerate(objects_to_place):
                placed = False
                for attempt in range(max_placement_attempts):
                    max_r = grid_size - obj.shape[0]
                    max_c = grid_size - obj.shape[1]
                    
                    if max_r < 0 or max_c < 0:
                        break  # Object too big
                    
                    r = random.randint(0, max_r)
                    c = random.randint(0, max_c)
                    
                    # Check if this position overlaps with existing objects
                    overlap = False
                    for pr, pc, pobj in positions:
                        # Check if bounding boxes overlap
                        if not (r >= pr + pobj.shape[0] or r + obj.shape[0] <= pr or
                                c >= pc + pobj.shape[1] or c + obj.shape[1] <= pc):
                            overlap = True
                            break
                    
                    if not overlap:
                        positions.append((r, c, obj))
                        placed = True
                        break
                
                if not placed:
                    break  # Couldn't place this object, try new grid
            
            # If we successfully placed all 3 objects
            if len(positions) == 3:
                # Place objects on grid
                for r, c, obj in positions:
                    h, w = obj.shape
                    grid[r:r+h, c:c+w] = obj
                
                # Verify we have exactly 3 connected objects
                found_objects = find_connected_objects(grid, diagonal_connectivity=False)
                
                if len(found_objects) == 3:
                    # Verify we have exactly 2 similar and 1 different
                    object_arrays = []
                    for obj in found_objects:
                        arr = obj.to_array()
                        object_arrays.append(arr)
                    
                    # Count identical pairs
                    identical_pairs = 0
                    unique_objects = 0
                    
                    for i in range(len(object_arrays)):
                        matches = 0
                        for j in range(len(object_arrays)):
                            if i != j and self.objects_are_identical(object_arrays[i], object_arrays[j]):
                                matches += 1
                        
                        if matches == 1:  # Has exactly one match (part of a pair)
                            identical_pairs += 1
                        elif matches == 0:  # Has no matches (unique)
                            unique_objects += 1
                    
                    # We should have exactly 2 objects in the identical pair and 1 unique
                    if identical_pairs == 2 and unique_objects == 1:
                        return grid
        
        # If we couldn't create a valid grid, create a simple fallback
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Simple guaranteed different objects
        obj1 = np.array([[object_color, object_color], [object_color, 0]])  # L-shape
        obj2 = np.array([[object_color, object_color], [object_color, 0]])  # Same L-shape
        obj3 = np.array([[object_color], [object_color], [object_color]])    # Line
        
        # Place them with safe spacing
        grid[1:3, 1:3] = obj1
        grid[1:3, 5:7] = obj2  
        grid[5:8, 1:2] = obj3
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input by coloring identical and different objects with different colors."""
        same_color = taskvars["same_color"]
        different_color = taskvars["different_color"]
        output_grid = np.zeros_like(grid)  # Start with empty grid
        
        objects = find_connected_objects(grid, diagonal_connectivity=False)
        
        if len(objects) != 3:
            return grid.copy()  # Should have exactly 3 objects
        
        # Find the different object and the identical objects by comparing shapes
        object_arrays = []
        for obj in objects:
            arr = obj.to_array()
            object_arrays.append((arr, obj))
        
        # Classify objects as identical or different
        identical_objects = []
        different_objects = []
        
        for i, (arr1, obj1) in enumerate(object_arrays):
            matches = 0
            for j, (arr2, _) in enumerate(object_arrays):
                if i != j and self.objects_are_identical(arr1, arr2):
                    matches += 1
            
            if matches == 1:  # This object has exactly one match (part of identical pair)
                identical_objects.append(obj1)
            elif matches == 0:  # This object has no matches (unique/different)
                different_objects.append(obj1)
        
        # Color the identical objects with same_color
        for obj in identical_objects:
            for r, c, _ in obj:
                output_grid[r, c] = same_color
        
        # Color the different object with different_color
        for obj in different_objects:
            for r, c, _ in obj:
                output_grid[r, c] = different_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Choose 3 different random colors between 1-9
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        object_color = available_colors[0]
        same_color = available_colors[1]
        different_color = available_colors[2]
            
        taskvars = {
            "object_color": object_color,
            "same_color": same_color,
            "different_color": different_color
        }
        
        # Generate 3-5 training examples with same task variables
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with same task variables
        test_gridvars = {}
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

# Test code
if __name__ == "__main__":
    generator = Taskb230c067Generator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)