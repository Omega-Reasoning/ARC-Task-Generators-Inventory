from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task63613498Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain several colored (1–9) objects, where each object is made of 4-way connected cells of the same color.",
            "Each object is sized to fit within a 3x3 subgrid.",
            "There is one vertical line starting from (0, 3) to (3, 3), and one horizontal line starting from (3, 0) to (3, 3), both lines are made of {color('frame_color')} color.",
            "These two lines form a 3x3 subgrid in the top-left corner that always contains one colored object made of 4-way connected cells.",
            "One of the objects outside this 3x3 subgrid bounded by {color('frame_color')} lines has the exact same shape as the object inside the top-left 3x3 subgrid.",
            "All the objects outside the {color('frame_color')} boundary, must be completely separated from each other."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the object inside the top-left 3x3 subgrid bounded by the {color('frame_color')} lines.",
            "Once this object is identified, scan the grid to find the second object that has the exact same shape as the object inside the 3x3 subgrid.",
            "Once found, change the color of the second object to {color('new_color')}.",
            "All other cells remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Define task variables
        grid_size = random.randint(9, 20)  # Keeping slightly smaller than max for performance
        
        # Now new_color and frame_color are the same
        frame_color = random.randint(1, 9)
        new_color = frame_color  # Using same color
        
        taskvars = {
            'grid_size': grid_size,
            'frame_color': frame_color,
            'new_color': new_color
        }
        
        # Create 4 training and 1 test example
        num_train = 4
        num_test = 1
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        for _ in range(num_test):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_examples.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        frame_color = taskvars['frame_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create frame (vertical and horizontal lines)
        grid[0:4, 3] = frame_color
        grid[3, 0:4] = frame_color
        
        # Generate an object for the 3x3 top-left corner
        def create_corner_object():
            # Randomly decide object size (3, 4, or 5 cells)
            object_size = random.choice([3, 4, 5])
            
            # Random color that's not frame_color
            color = random.choice([c for c in range(1, 10) if c != frame_color])
            
            # Create an object that fits within 3x3 space
            obj_matrix = np.zeros((3, 3), dtype=int)
            
            # Start with one random cell
            r, c = random.randint(0, 2), random.randint(0, 2)
            obj_matrix[r, c] = color
            filled_cells = 1
            
            # Set of potential neighbors to add
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            neighbors = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    neighbors.append((nr, nc))
            
            # Keep adding cells until we reach desired size
            while filled_cells < object_size and neighbors:
                # Pick a random neighbor
                idx = random.randrange(len(neighbors))
                nr, nc = neighbors.pop(idx)
                
                # Fill it
                obj_matrix[nr, nc] = color
                filled_cells += 1
                
                # Add its neighbors to the list
                for dr, dc in directions:
                    new_r, new_c = nr + dr, nc + dc
                    if (0 <= new_r < 3 and 0 <= new_c < 3 and 
                        obj_matrix[new_r, new_c] == 0 and 
                        (new_r, new_c) not in neighbors):
                        neighbors.append((new_r, new_c))
            
            return obj_matrix, color
        
        # Create corner object
        corner_obj_matrix, corner_obj_color = create_corner_object()
        
        # Place it in the top-left 3x3 region (keeping frame intact)
        grid[0:3, 0:3] = corner_obj_matrix
        
        # Create a list of all available colors (except frame_color and corner_obj_color)
        available_colors = [c for c in range(1, 10) if c != frame_color and c != corner_obj_color]
        
        # Calculate how many objects to add based on grid size
        # Let's use a formula that scales with grid size but limit it to ensure we have enough space
        num_objects = max(4, min(grid_size // 3, (grid_size * grid_size) // 30))
        
        # We'll need one object with the same shape as the corner object
        # Let's pick a random color for it
        matching_obj_color = random.choice(available_colors)
        available_colors.remove(matching_obj_color)
        
        # Create placeholder for objects to place
        objects_to_place = []
        
        # Add the matching object to our list first - ensuring we have exactly one match
        matching_obj = corner_obj_matrix.copy()
        objects_to_place.append((matching_obj, matching_obj_color))
        
        # Now create other objects with shapes that differ from the corner object
        for _ in range(num_objects - 1):  # -1 because we already have one object
            if not available_colors:
                # If we run out of colors, reuse them
                available_colors = [c for c in range(1, 10) if c != frame_color]
            
            # Pick a color
            color = random.choice(available_colors)
            available_colors.remove(color)
            
            # Create a random object of size 3, 4, or 5
            obj_size = random.choice([3, 4, 5])
            
            def create_distinct_object():
                # Create a random object
                obj = create_object(3, 3, color, Contiguity.FOUR, 0)
                
                # Check that it has the right size
                if np.sum(obj > 0) != obj_size:
                    return None
                    
                # Check that it has a different shape than the corner object
                obj_coords = set()
                for r in range(3):
                    for c in range(3):
                        if obj[r, c] > 0:
                            obj_coords.add((r, c))
                
                corner_coords = set()
                for r in range(3):
                    for c in range(3):
                        if corner_obj_matrix[r, c] > 0:
                            corner_coords.add((r, c))
                            
                # Convert to normalized coordinates for shape comparison
                if obj_coords:
                    min_r = min(r for r, _ in obj_coords)
                    min_c = min(c for _, c in obj_coords)
                    obj_shape = {(r - min_r, c - min_c) for r, c in obj_coords}
                    
                    min_r = min(r for r, _ in corner_coords)
                    min_c = min(c for _, c in corner_coords)
                    corner_shape = {(r - min_r, c - min_c) for r, c in corner_coords}
                    
                    # Return None if shapes match (we want different shapes)
                    if obj_shape == corner_shape:
                        return None
                        
                return obj
            
            # Try to create a distinct object multiple times
            distinct_obj = None
            for _ in range(100):  # Limit attempts to avoid infinite loop
                distinct_obj = create_distinct_object()
                if distinct_obj is not None:
                    break
                    
            # If we couldn't create a distinct object, create a simple one
            if distinct_obj is None:
                distinct_obj = np.zeros((3, 3), dtype=int)
                # Make a simple L shape that's unlikely to match the corner object
                distinct_obj[0:2, 0] = color
                distinct_obj[0, 0:2] = color
            
            objects_to_place.append((distinct_obj, color))
        
        # Function to check if a position is valid for an object
        def is_valid_position(obj, pos_r, pos_c):
            obj_rows, obj_cols = obj.shape
            
            # Check if out of bounds
            if pos_r + obj_rows > grid_size or pos_c + obj_cols > grid_size:
                return False
                
            # Check if it overlaps with frame area
            if pos_r < 4 and pos_c < 4:
                return False
                
            # Check for collision with existing objects or being adjacent to them
            for r in range(obj_rows):
                for c in range(obj_cols):
                    if obj[r, c] > 0:
                        grid_r, grid_c = pos_r + r, pos_c + c
                        
                        # Check direct collision
                        if grid[grid_r, grid_c] > 0:
                            return False
                            
                        # Check all 8 surrounding cells to ensure separation
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = grid_r + dr, grid_c + dc
                                if (0 <= nr < grid_size and 0 <= nc < grid_size and 
                                    grid[nr, nc] > 0 and not (nr == grid_r and nc == grid_c)):
                                    # Found a non-zero cell adjacent or diagonal
                                    return False
            
            return True
        
        # Place all objects with proper separation
        for obj, color in objects_to_place:
            # Try to find a valid position multiple times
            placed = False
            for attempt in range(200):  # Increased attempts
                pos_r = random.randint(4, grid_size - obj.shape[0])  # Start after frame area
                pos_c = random.randint(4, grid_size - obj.shape[1])  # Start after frame area
                
                if is_valid_position(obj, pos_r, pos_c):
                    # Place the object with its color
                    for r in range(obj.shape[0]):
                        for c in range(obj.shape[1]):
                            if obj[r, c] > 0:
                                grid[pos_r + r, pos_c + c] = color
                    placed = True
                    break
            
            # If we couldn't place this object, reduce the total number of objects to place
            if not placed:
                continue
            
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        frame_color = taskvars['frame_color']
        new_color = taskvars['new_color']  # Now equal to frame_color
        
        # Find all objects in the grid using 4-way connectivity
        objects = find_connected_objects(grid, diagonal_connectivity=False)
        
        # Identify the object inside the 3x3 top-left region
        reference_object = None
        for obj in objects:
            # Check if all cells are within the 3x3 top-left corner
            if all(r < 3 and c < 3 for r, c, _ in obj.cells):
                reference_object = obj
                break
        
        if reference_object is None:
            # This shouldn't happen with our generation logic
            return output_grid
            
        # Get the shape of the reference object (as a set of relative coordinates)
        ref_min_r = min(r for r, _, _ in reference_object.cells)
        ref_min_c = min(c for _, c, _ in reference_object.cells)
        reference_shape = {(r - ref_min_r, c - ref_min_c) for r, c, _ in reference_object.cells}
        
        # Find the matching object outside the frame
        matching_found = False
        for obj in objects:
            # Skip objects inside frame or with frame color
            if any(r < 3 and c < 3 for r, c, _ in obj.cells) or frame_color in obj.colors:
                continue
                
            # Check if shape matches
            obj_min_r = min(r for r, _, _ in obj.cells)
            obj_min_c = min(c for _, c, _ in obj.cells)
            obj_shape = {(r - obj_min_r, c - obj_min_c) for r, c, _ in obj.cells}
            
            if reference_shape == obj_shape:
                # Change the color of this matching object
                for r, c, _ in obj.cells:
                    output_grid[r, c] = new_color
                matching_found = True
                break
        
        # Our generator should guarantee a match, but just in case
        if not matching_found:
            print("Warning: No matching object found!")
                
        return output_grid

