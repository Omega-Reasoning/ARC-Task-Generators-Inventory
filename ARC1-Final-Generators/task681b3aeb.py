from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task681b3aebGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
        
            "They contain exactly two objects, where each object is made of 4-way connected cells of the same color, and all remaining cells are empty (0).",
        
            "The objects are formed by first creating an imaginary 3x3 grid completely filled with exactly two different colored objects.",
    
            "Once this imaginary grid is created, both colored objects are randomly placed within the actual grid.",
            "The shapes and sizes of both objects should be different from each other but strictly such that, when connected, they form a completely colored 3x3 grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size 3x3.",
          
            "They are constructed by identifying the two colored objects in the input grids.",
           
            "Once identified, both objects are placed into a 3x3 output grid and are connected such that they form a completely filled grid with no empty (0) cells remaining.",
            "The positioning should ensure that both objects fit together perfectly and are completely added to the output grid, with no portion truncated."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Set a random grid size between 7 and 30
        grid_size = random.randint(7, 30)
        taskvars = {'grid_size': grid_size}
        
        # Generate 3-4 training examples
        train_count = random.randint(3, 4)
        
        train_data = []
        for _ in range(train_count):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate 1 test example
        test_input_grid = self.create_input(taskvars, {})
        test_output_grid = self.transform_input(test_input_grid, taskvars)
        test_data = [{
            'input': test_input_grid,
            'output': test_output_grid
        }]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        
        # Create an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Choose two random distinct colors from 1-9
        colors = random.sample(range(1, 10), 2)
        
        # Create a 3x3 template with two differently shaped objects
        # Templates where objects only fit together in one unique way
        templates = [
            # Template 1: L-shape and complementary shape
            np.array([
                [1, 2, 2],
                [1, 2, 2],
                [1, 1, 2]
            ]),
            # Template 2: Zigzag and complementary shape
            np.array([
                [1, 1, 2],
                [2, 1, 2],
                [2, 1, 1]
            ]),
            # Template 3: Corner and complementary shape
            np.array([
                [1, 1, 2],
                [1, 2, 2],
                [2, 2, 2]
            ]),
            # Template 4: T-shape and complementary shape
            np.array([
                [1, 1, 1],
                [2, 1, 2],
                [2, 2, 2]
            ]),
            # Template 5: U-shape and complementary shape
            np.array([
                [1, 2, 1],
                [1, 2, 1],
                [1, 1, 1]
            ]),
            # Template 6: S-shape and complementary shape
            np.array([
                [1, 1, 2],
                [2, 1, 1],
                [2, 2, 2]
            ]),
            # Template 7: Plus-sign fragment and complementary shape
            np.array([
                [2, 1, 2],
                [1, 1, 1],
                [2, 1, 2]
            ]),
        ]
        
        # Create the template grid
        template = random.choice(templates)
        rotations = random.randint(0, 3)
        template = np.rot90(template, k=rotations)
        if random.choice([True, False]):
            template = 3 - template  # Swaps 1 and 2
        
        # Extract the two objects from the template
        object1_cells = {(r, c) for r in range(3) for c in range(3) if template[r, c] == 1}
        object2_cells = {(r, c) for r in range(3) for c in range(3) if template[r, c] == 2}
        
        # Function to check if cells are connected
        def is_connected(cells):
            if not cells:
                return False
            
            # Start with first cell
            visited = set()
            to_visit = [next(iter(cells))]
            
            # BFS to find all connected cells
            while to_visit:
                r, c = to_visit.pop(0)
                if (r, c) in visited:
                    continue
                
                visited.add((r, c))
                
                # Check neighbors (4-way connectivity)
                neighbors = [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]
                for nr, nc in neighbors:
                    if (nr, nc) in cells and (nr, nc) not in visited:
                        to_visit.append((nr, nc))
            
            # If all cells were visited, the component is connected
            return len(visited) == len(cells)
        
        # Function to check if two sets of cells are 4-way adjacent to each other
        def are_4way_adjacent(cells1, cells2):
            for r1, c1 in cells1:
                for r2, c2 in cells2:
                    # Check if cells are 4-way adjacent (sharing an edge)
                    if (abs(r1 - r2) == 1 and c1 == c2) or (r1 == r2 and abs(c1 - c2) == 1):
                        return True
            return False
        
        # Verify that both objects form connected components
        if not is_connected(object1_cells) or not is_connected(object2_cells):
            return self.create_input(taskvars, gridvars)
        
        # Place the objects at random positions in the larger grid
        max_attempts = 100  # Increased for better chances of success
        for attempt in range(max_attempts):
            # Try random positions
            r_offset1 = random.randint(0, grid_size - 3)
            c_offset1 = random.randint(0, grid_size - 3)
            
            # Place second object far enough from first to ensure no 4-way adjacency
            valid_position_found = False
            for inner_attempt in range(max_attempts):
                r_offset2 = random.randint(0, grid_size - 3)
                c_offset2 = random.randint(0, grid_size - 3)
                
                # Calculate new positions for both objects
                obj1_new_cells = {(r + r_offset1, c + c_offset1) for r, c in object1_cells}
                obj2_new_cells = {(r + r_offset2, c + c_offset2) for r, c in object2_cells}
                
                # Check if there's no overlap and no 4-way adjacency
                if (not obj1_new_cells.intersection(obj2_new_cells) and 
                    not are_4way_adjacent(obj1_new_cells, obj2_new_cells)):
                    valid_position_found = True
                    break
            
            if not valid_position_found:
                continue
                
            # Place the objects in the grid
            temp_grid = np.zeros_like(grid)
            for r, c in obj1_new_cells:
                temp_grid[r, c] = colors[0]
            for r, c in obj2_new_cells:
                temp_grid[r, c] = colors[1]
            
            # Double check: verify we have exactly 2 objects with 4-way connectivity
            objects = find_connected_objects(temp_grid, diagonal_connectivity=False, background=0)
            if len(objects) == 2:
                # Also verify the objects don't touch each other (redundant but safe)
                obj1_coords = objects[0].coords
                obj2_coords = objects[1].coords
                if not are_4way_adjacent(obj1_coords, obj2_coords):
                    return temp_grid
        
        # If we couldn't place objects without overlap or adjacency, try again
        return self.create_input(taskvars, gridvars)
    
    def transform_input(self, grid, taskvars):
        # Create the output 3x3 grid
        output_grid = np.zeros((3, 3), dtype=int)
        
        # Find the two colored objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # Ensure we found exactly two objects
        if len(objects) != 2:
            raise ValueError(f"Expected 2 objects but found {len(objects)}")
        
        # Extract the objects as arrays
        object_arrays = []
        for obj in objects:
            # Get the object color
            color = list(obj.colors)[0]
            
            # Create an array representation
            array = obj.to_array()
            
            # Replace non-zero values with the object's color
            colored_array = np.zeros_like(array)
            colored_array[array > 0] = color
            
            object_arrays.append((colored_array, color))
        
        # Try to place the objects to form a complete 3x3 grid
        # We need to find the correct positions and rotations
        success = False
        
        # Try different arrangements of the two objects
        for rot1 in range(4):  # Try 4 rotations for first object
            for rot2 in range(4):  # Try 4 rotations for second object
                # Rotate the objects
                obj1_rotated = np.rot90(object_arrays[0][0], k=rot1)
                obj2_rotated = np.rot90(object_arrays[1][0], k=rot2)
                
                color1 = object_arrays[0][1]
                color2 = object_arrays[1][1]
                
                # Try all possible positions for placing obj1 and obj2
                for pos1_r in range(-obj1_rotated.shape[0]+1, 3):
                    for pos1_c in range(-obj1_rotated.shape[1]+1, 3):
                        for pos2_r in range(-obj2_rotated.shape[0]+1, 3):
                            for pos2_c in range(-obj2_rotated.shape[1]+1, 3):
                                # Create a test output grid
                                test_grid = np.zeros((3, 3), dtype=int)
                                
                                # Try to place first object
                                invalid_placement = False
                                for r in range(obj1_rotated.shape[0]):
                                    for c in range(obj1_rotated.shape[1]):
                                        if obj1_rotated[r, c] > 0:
                                            new_r, new_c = pos1_r + r, pos1_c + c
                                            if 0 <= new_r < 3 and 0 <= new_c < 3:
                                                if test_grid[new_r, new_c] == 0:
                                                    test_grid[new_r, new_c] = color1
                                                else:
                                                    invalid_placement = True
                                                    break
                                            else:
                                                invalid_placement = True
                                                break
                                        if invalid_placement:
                                            break
                                    if invalid_placement:
                                        break
                                
                                if invalid_placement:
                                    continue
                                
                                # Try to place second object
                                for r in range(obj2_rotated.shape[0]):
                                    for c in range(obj2_rotated.shape[1]):
                                        if obj2_rotated[r, c] > 0:
                                            new_r, new_c = pos2_r + r, pos2_c + c
                                            if 0 <= new_r < 3 and 0 <= new_c < 3:
                                                if test_grid[new_r, new_c] == 0:
                                                    test_grid[new_r, new_c] = color2
                                                else:
                                                    invalid_placement = True
                                                    break
                                            else:
                                                invalid_placement = True
                                                break
                                        if invalid_placement:
                                            break
                                    if invalid_placement:
                                        break
                                
                                if invalid_placement:
                                    continue
                                
                                # Check if the grid is completely filled
                                if np.all(test_grid > 0):
                                    output_grid = test_grid
                                    success = True
                                    break
                            
                            if success:
                                break
                        if success:
                            break
                    if success:
                        break
                if success:
                    break
            if success:
                break
        
        if not success:
            # If we couldn't find a valid arrangement, something is wrong with our objects
            # This shouldn't happen given our input generation, but let's handle it
            raise ValueError("Could not arrange objects to fill a 3x3 grid")
            
        return output_grid

