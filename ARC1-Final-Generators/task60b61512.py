from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects

class Task60b61512Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids vary in size.",
            "They contain several {color('object_color')} colored objects along with empty (0) cells.",
            "Each {color('object_color')} object is made of 8-way connected cells and has a distinct shape.",
            "Objects are separated from one another by empty (0) cells."
        ]
        
        transformation_reasoning_chain = [
           "The output grid is constructed by copying the input grid and identifying all {color('object_color')} colored objects, which may differ in number across grids.",
    "For each object, determine its tight axis-aligned bounding box.",
    "Fill all empty (0) cells inside the bounding box of each object using {color('fill_color')} color."
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
            # Create an object with a random rectangular bounding box (not limited to 3x3)
            # Height and width chosen so object is reasonably small but variable
            max_dim = max(2, min(8, grid_size // 3))
            h = random.randint(2, max_dim)
            w = random.randint(2, max_dim)

            obj = np.zeros((h, w), dtype=int)

            # Random starting point inside object's local coords
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
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
                    if 0 <= nr < h and 0 <= nc < w:
                        candidates.append((nr, nc))

            # Add 2-6 more cells to create a connected object
            num_to_add = random.randint(2, min(6, h * w - 1))
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
                        if 0 <= nnr < h and 0 <= nnc < w and obj[nnr, nnc] == 0 and (nnr, nnc) not in candidates:
                            candidates.append((nnr, nnc))

            # Ensure each row and column has at least one colored cell
            for i in range(h):
                if np.all(obj[i, :] == 0):  # Empty row
                    obj[i, random.randint(0, w - 1)] = object_color
            for j in range(w):
                if np.all(obj[:, j] == 0):  # Empty column
                    obj[random.randint(0, h - 1), j] = object_color

            # Check if it's 8-way connected
            from scipy.ndimage import label
            binary = np.where(obj != 0, 1, 0)
            labeled, num = label(binary, structure=np.ones((3, 3)))

            # Reject shapes that are perfect filled rectangles (including squares).
            # Compute tight bounding box of the filled cells and ensure it's not completely filled.
            if num == 1:
                ys, xs = np.where(binary == 1)
                if ys.size == 0:
                    return None
                min_r, max_r = ys.min(), ys.max()
                min_c, max_c = xs.min(), xs.max()
                sub = binary[min_r:max_r+1, min_c:max_c+1]
                if np.all(sub == 1):
                    # object is a perfect rectangle/square — reject
                    return None

            return obj if num == 1 else None

        # Distinctness check against any previously created object
        def is_distinct_against_all(candidate, existing_list):
            if candidate is None:
                return False

            cand_bin = np.where(candidate != 0, 1, 0)
            for ex in existing_list:
                ex_bin = np.where(ex != 0, 1, 0)
                for k in range(4):
                    rotated = np.rot90(cand_bin, k=k)
                    if np.array_equal(rotated, ex_bin):
                        return False
                    if np.array_equal(np.flip(rotated, axis=0), ex_bin):
                        return False
                    if np.array_equal(np.flip(rotated, axis=1), ex_bin):
                        return False
            return True

        # Decide how many objects to place in this grid (varies across grids)
        max_objects = min(6, max(2, grid_size // 3))
        num_objects = random.randint(2, max_objects)

        objects = []
        # Generate distinct object shapes
        for _ in range(num_objects):
            attempts = 0
            obj = None
            while (obj is None) or (not is_distinct_against_all(obj, objects)):
                obj = generate_valid_object()
                attempts += 1
                if attempts > 200:
                    # Fallback: create a simple but different pattern
                    fallback = np.zeros((3, 3), dtype=int)
                    # try to add a few anchor cells that are unlikely to match previous ones
                    fallback[0, 0] = object_color
                    fallback[1, 1] = object_color
                    fallback[2, 0] = object_color
                    obj = fallback
                    break
            objects.append(obj)

        # Place objects with enough separation
        # minimum separation between object origins (adapt to object size later)
        base_min_distance = 3
        placed_positions = []

        for obj in objects:
            h, w = obj.shape
            min_distance = max(base_min_distance, max(h, w))

            placed = False
            attempts = 0
            while not placed and attempts < 300:
                # pick a top-left start so the object fits
                r = random.randint(0, grid_size - h)
                c = random.randint(0, grid_size - w)

                # Ensure region is empty
                region_empty = np.all(grid[r:r+h, c:c+w] == 0)
                # Ensure distance from previously placed objects
                far_enough = all(max(abs(r - pr), abs(c - pc)) > min_distance for pr, pc in placed_positions)

                if region_empty and far_enough:
                    grid[r:r+h, c:c+w] = obj
                    placed_positions.append((r, c))
                    placed = True

                attempts += 1

            if not placed:
                # Try a systematic scan for a fitting location
                found = False
                for rr in range(0, grid_size - h + 1):
                    if found:
                        break
                    for cc in range(0, grid_size - w + 1):
                        if np.all(grid[rr:rr+h, cc:cc+w] == 0) and all(max(abs(rr - pr), abs(cc - pc)) > min_distance for pr, pc in placed_positions):
                            grid[rr:rr+h, cc:cc+w] = obj
                            placed_positions.append((rr, cc))
                            found = True
                            break

                if not found:
                    # As a last resort, place in the first empty 3x3 area without regard to distance
                    for rr in range(1, grid_size - 4):
                        if found:
                            break
                            for cc in range(0, grid_size - w + 1):
                                if np.all(grid[rr:rr+h, cc:cc+w] == 0):
                                    grid[rr:rr+h, cc:cc+w] = obj
                                    placed_positions.append((rr, cc))
                                    found = True
                                    break

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
