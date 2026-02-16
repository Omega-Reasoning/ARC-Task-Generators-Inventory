from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity, retry
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects

class Task5117e062Generator(ARCTaskGenerator):
    def __init__(self):
        self.input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a random number of colored (1-9) objects, each made of 4-way connected, same-colored (1-9) cells, with all objects being differently colored.",
            "The objects are shaped and sized to fit within a 3x3 subgrid, ensuring each object has a distinct shape.",
            "One of these objects contains exactly one {color('cell_color')} cell within it.",
            "Each colored object is completely separated from the others."
        ]
        
        self.transformation_reasoning_chain = [
            "Output grids are of size 3x3.",
            "They are constructed by identifying the colored object that contains a {color('cell_color')} cell.",
            "Once identified, this object is copied to the output grid while maintaining its original shape and color.",
            "Then, change the color of the {color('cell_color')} cell to match the color of the object."
        ]
        
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        
        taskvars = {
            'rows': random.randint(10, 30),
            'cols': random.randint(10, 30),
            'cell_color': random.randint(1, 9)
        }
        
        # Generate training and test examples
        num_train_examples = random.randint(3, 4)
        
        train_examples = []
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        for _ in range(1):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_examples.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_object_shape(self, color):
        """Helper method to create a valid 3x3 shape with the specified color
        that has at least one cell in each row and column"""
        
        def generate_shape():
            # Start with empty 3x3 grid
            shape = np.zeros((3, 3), dtype=int)
            
            # Ensure we have at least one cell in each row and column
            for row in range(3):
                col = random.randint(0, 2)
                shape[row, col] = color
            
            # Check if any columns are empty and fill them
            for col in range(3):
                if np.all(shape[:, col] == 0):
                    row = random.randint(0, 2)
                    shape[row, col] = color
            
            # Add more cells to ensure connectivity if needed
            while True:
                # Find all cells adjacent to colored cells
                candidates = []
                for i in range(3):
                    for j in range(3):
                        if shape[i, j] == color:
                            # Check 4 neighbors
                            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < 3 and 0 <= nj < 3 and shape[ni, nj] == 0:
                                    candidates.append((ni, nj))
                
                # Check if all cells are connected
                objects = find_connected_objects(shape, diagonal_connectivity=False, background=0)
                colored_cells = np.sum(shape != 0)
                
                if len(objects.objects) <= 1 and colored_cells >= 3:
                    # We have a connected object with at least 3 cells
                    break
                
                # Add more cells to connect components or reach minimum size
                if candidates:
                    r, c = random.choice(candidates)
                    shape[r, c] = color
                else:
                    # Can't add more cells, this shape isn't working
                    return np.zeros((3, 3), dtype=int)
            
            return shape
        
        def is_valid_shape(shape):
            # Check if shape has at least one cell in each row and column
            for row in range(3):
                if np.all(shape[row, :] == 0):
                    return False
                    
            for col in range(3):
                if np.all(shape[:, col] == 0):
                    return False
            
            # Check 4-way connectivity
            objects = find_connected_objects(shape, diagonal_connectivity=False, background=0)
            colored_cells = np.sum(shape != 0)
            
            # Must have between 3-5 cells, be connected, and cover all rows and columns
            return colored_cells >= 3 and colored_cells <= 5 and len(objects.objects) == 1
        
        return retry(generate_shape, is_valid_shape, max_attempts=50)
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        cell_color = taskvars['cell_color']
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Determine number of objects (random between 2 and 7)
        num_objects = random.randint(2, 7)
        
        # Generate available colors (different from cell_color)
        available_colors = [c for c in range(1, 10) if c != cell_color]
        random.shuffle(available_colors)
        object_colors = available_colors[:num_objects]
        
        # Select which object will contain the special cell (randomly)
        special_object_idx = random.randint(0, num_objects - 1)
        
        # Generate the shapes for all objects first
        all_shapes = []
        for i in range(num_objects):
            # Use unique shapes
            shape_seed = random.randint(0, 100000)  # Use a different seed for each shape
            random.seed(shape_seed + i)
            
            color = object_colors[i]
            shape = self.create_object_shape(color)
            
            # If this is the special object, add the special cell
            if i == special_object_idx:
                # Find a cell to replace with special color
                nonzero_positions = list(zip(*np.where(shape == color)))
                if nonzero_positions:
                    r, c = random.choice(nonzero_positions)
                    shape[r, c] = cell_color
            
            all_shapes.append(shape)
        
        # Reset random seed to maintain diversity
        random.seed()
            
        # Try multiple times to place all objects
        max_attempts = 20
        success = False
        
        for attempt in range(max_attempts):
            # Reset the grid
            grid = np.zeros((rows, cols), dtype=int)
            placed_objects = []
            all_placed = True
            
            # Shuffle object placement order
            indices = list(range(num_objects))
            random.shuffle(indices)
            
            # Try to place each object
            for idx in indices:
                shape = all_shapes[idx]
                
                # Ensure minimum separation between objects
                min_distance = 2
                max_placement_attempts = 50
                
                for placement_attempt in range(max_placement_attempts):
                    # Generate random position
                    r_pos = random.randint(0, rows - 3)
                    c_pos = random.randint(0, cols - 3)
                    
                    # Check if position is valid (doesn't overlap with existing objects)
                    valid_position = True
                    
                    # Check all positions where the shape would be placed
                    for r_offset in range(3):
                        for c_offset in range(3):
                            if shape[r_offset, c_offset] == 0:
                                continue  # Skip empty cells
                            
                            # Check if there's anything nearby
                            for dr in range(-min_distance, min_distance + 1):
                                for dc in range(-min_distance, min_distance + 1):
                                    check_r = r_pos + r_offset + dr
                                    check_c = c_pos + c_offset + dc
                                    
                                    # Bounds check
                                    if 0 <= check_r < rows and 0 <= check_c < cols:
                                        # If position near our object is already filled
                                        if grid[check_r, check_c] != 0 and (abs(dr) > 0 or abs(dc) > 0):
                                            valid_position = False
                                            break
                                if not valid_position:
                                    break
                        if not valid_position:
                            break
                    
                    if valid_position:
                        # Place the object
                        for r_offset in range(3):
                            for c_offset in range(3):
                                if shape[r_offset, c_offset] != 0:
                                    grid[r_pos + r_offset, c_pos + c_offset] = shape[r_offset, c_offset]
                        
                        placed_objects.append((r_pos, c_pos))
                        break
                
                # If couldn't place an object after max attempts
                if placement_attempt == max_placement_attempts - 1:
                    all_placed = False
                    break
            
            # If all objects placed successfully
            if all_placed:
                success = True
                break
        
        # If couldn't place all objects after multiple attempts, create a simpler grid
        if not success:
            # Fall back to a simpler approach - just place objects with more padding
            grid = np.zeros((rows, cols), dtype=int)
            
            # Place objects in a grid pattern
            spacing = max(5, min(rows // 3, cols // 3))
            positions = []
            
            for r in range(0, rows - 3, spacing):
                for c in range(0, cols - 3, spacing):
                    positions.append((r, c))
            
            # Shuffle positions and use as many as needed
            random.shuffle(positions)
            
            for i in range(min(num_objects, len(positions))):
                r_pos, c_pos = positions[i]
                shape = all_shapes[i]
                
                # Place the object
                for r_offset in range(3):
                    for c_offset in range(3):
                        if shape[r_offset, c_offset] != 0:
                            grid[r_pos + r_offset, c_pos + c_offset] = shape[r_offset, c_offset]
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        cell_color = taskvars['cell_color']
        
        # Create a 3x3 output grid
        output_grid = np.zeros((3, 3), dtype=int)
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
        
        # Find the object containing the special cell
        special_object = None
        for obj in objects:
            obj_colors = {color for _, _, color in obj.cells}
            if cell_color in obj_colors:
                special_object = obj
                break
        
        if special_object is None:
            # This should not happen with correctly generated inputs
            return output_grid
        
        # Get the object's color (any color except cell_color)
        obj_colors = {color for _, _, color in special_object.cells if color != cell_color}
        if not obj_colors:
            # This should not happen with correctly generated inputs
            return output_grid
            
        obj_color = next(iter(obj_colors))
        
        # Extract the object's coordinates
        coords = [(r, c, color) for r, c, color in special_object.cells]
        
        # Calculate bounding box
        r_coords = [r for r, _, _ in coords]
        c_coords = [c for _, c, _ in coords]
        min_r, max_r = min(r_coords), max(r_coords)
        min_c, max_c = min(c_coords), max(c_coords)
        
        # Create a normalized version of the object in the 3x3 output grid
        for r, c, color in coords:
            normalized_r = r - min_r
            normalized_c = c - min_c
            
            # Ensure we stay within the 3x3 boundaries
            if 0 <= normalized_r < 3 and 0 <= normalized_c < 3:
                # If this is the special cell, change its color to match the object
                if color == cell_color:
                    output_grid[normalized_r, normalized_c] = obj_color
                else:
                    output_grid[normalized_r, normalized_c] = color
        
        return output_grid

