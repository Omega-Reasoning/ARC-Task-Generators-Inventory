from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects

class Task60b61512Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "They contain two {color('object_color')} colored objects along with empty (0) cells.",
            "Each {color('object_color')} object is made of 8-way connected cells and has a distinct shape.",
            "Every object is confined within a 3x3 subgrid and is completely surrounded by empty (0) cells.",
            "Each row and column within this 3x3 subgrid must contain at least one {color('object_color')} cell."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the two {color('object_color')} colored objects.",
            "Each object is confined within its own 3x3 subgrid.",
            "Once both objects and their respective 3x3 subgrids are identified, fill all empty (0) cells within their respective 3x3 subgrids using {color('fill_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(10, 30),  # Increased minimum size to ensure enough space
            'object_color': random.randint(1, 9),
            'fill_color': random.randint(1, 9)
        }
        
        # Ensure object_color and fill_color are different
        while taskvars['fill_color'] == taskvars['object_color']:
            taskvars['fill_color'] = random.randint(1, 9)
        
        # Generate 3-6 training examples and one test example
        num_train_examples = random.randint(3, 6)
        
        return taskvars, self.create_grids_default(num_train_examples, 1, taskvars)
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Generate a valid object using create_object and making sure it's 8-way connected
        def generate_valid_object():
            # Create an object with specified color
            obj = np.zeros((3, 3), dtype=int)
            
            # Random starting point
            r, c = random.randint(0, 2), random.randint(0, 2)
            obj[r, c] = object_color
            
            # List of cells we've filled and their neighbors
            filled = [(r, c)]
            candidates = []
            
            # Add neighbors of the first cell
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 3 and 0 <= nc < 3:
                        candidates.append((nr, nc))
            
            # Add 2-6 more cells to create a connected object
            num_to_add = random.randint(2, 6)
            for _ in range(num_to_add):
                if not candidates:
                    break
                
                # Pick a random candidate cell
                idx = random.randrange(len(candidates))
                nr, nc = candidates[idx]
                candidates.pop(idx)
                
                # Fill it
                obj[nr, nc] = object_color
                filled.append((nr, nc))
                
                # Add its neighbors to candidates
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nnr, nnc = nr + dr, nc + dc
                        if 0 <= nnr < 3 and 0 <= nnc < 3 and obj[nnr, nnc] == 0 and (nnr, nnc) not in candidates:
                            candidates.append((nnr, nnc))
            
            # Ensure each row and column has at least one colored cell
            for i in range(3):
                if np.all(obj[i, :] == 0):  # Empty row
                    obj[i, random.randint(0, 2)] = object_color
                if np.all(obj[:, i] == 0):  # Empty column
                    obj[random.randint(0, 2), i] = object_color
            
            # Check if it's 8-way connected
            from scipy.ndimage import label
            binary = np.where(obj != 0, 1, 0)
            labeled, num = label(binary, structure=np.ones((3, 3)))
            
            return obj if num == 1 else None
        
        # Create two distinct objects
        def is_distinct(obj2, obj1):
            if obj2 is None:
                return False
                
            # Check all rotations and reflections
            obj1_binary = np.where(obj1 != 0, 1, 0)
            obj2_binary = np.where(obj2 != 0, 1, 0)
            
            for k in range(4):  # 4 rotations
                rotated = np.rot90(obj2_binary, k=k)
                if np.array_equal(rotated, obj1_binary):
                    return False
                    
                flipped = np.flip(rotated, axis=0)
                if np.array_equal(flipped, obj1_binary):
                    return False
                    
                flipped = np.flip(rotated, axis=1)
                if np.array_equal(flipped, obj1_binary):
                    return False
            
            return True
        
        # Create first object
        obj1 = None
        while obj1 is None:
            obj1 = generate_valid_object()
            
        # Create second distinct object
        obj2 = None
        attempts = 0
        while obj2 is None or not is_distinct(obj2, obj1):
            obj2 = generate_valid_object()
            attempts += 1
            if attempts > 100:
                # Fallback if we can't find a distinct object
                obj2 = np.zeros((3, 3), dtype=int)
                if obj1[0, 0] == 0:  # Different pattern than obj1
                    obj2[0, 0] = object_color
                obj2[1, 1] = object_color
                obj2[2, 2] = object_color
                break
        
        # Place objects with enough separation
        # Calculate minimum required distance for non-8-connectivity
        min_distance = 3  # Minimum distance between subgrids to avoid 8-connectivity
        
        # Find positions for the first object
        r1 = random.randint(1, grid_size - 5)
        c1 = random.randint(1, grid_size - 5)
        
        # Place first object
        grid[r1:r1+3, c1:c1+3] = obj1
        
        # Find position for second object ensuring no 8-way connectivity
        attempts = 0
        while attempts < 100:
            r2 = random.randint(1, grid_size - 5)
            c2 = random.randint(1, grid_size - 5)
            
            # Check if there's enough distance between the subgrids
            if abs(r2 - r1) > min_distance or abs(c2 - c1) > min_distance:
                # Place second object
                grid[r2:r2+3, c2:c2+3] = obj2
                break
                
            attempts += 1
        
        # If we couldn't find a good position, place at opposite corners
        if attempts >= 100:
            r2 = grid_size - 5 if r1 < grid_size // 2 else 1
            c2 = grid_size - 5 if c1 < grid_size // 2 else 1
            grid[r2:r2+3, c2:c2+3] = obj2
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        object_color = taskvars['object_color']
        fill_color = taskvars['fill_color']
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find all connected colored objects
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        colored_objects = objects.filter(lambda obj: object_color in obj.colors)
        
        # Process each object
        for obj in colored_objects:
            # Find the 3x3 subgrid containing the object
            min_row = min(r for r, _, _ in obj.cells)
            min_col = min(c for _, c, _ in obj.cells)
            max_row = max(r for r, _, _ in obj.cells)
            max_col = max(c for _, c, _ in obj.cells)
            
            # Fill all empty cells within the 3x3 subgrid with fill_color
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    if output_grid[r, c] == 0:
                        output_grid[r, c] = fill_color
        
        return output_grid
