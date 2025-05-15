from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
import numpy as np
import random

class Taskb9b7f026Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.", 
            "The grid consists of exactly square or rectangular blocks of different colors (between 1 to 9).", 
            "They are spaced uniformly.",
            "There surely exists exactly one hollow block either in one of the squares or rectangles."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is always 1x1.",
            "We have to identify the shape which has that hollow block and replace the grid color with that particular shape color."
        ]
        
        taskvars_definitions = {}
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self):
        # Generate a grid with random shape blocks
        grid_size = random.randint(12, 20)  # Increased minimum size
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine number of shapes (3-6 to ensure they fit)
        num_shapes = random.randint(3, 6)
        
        # Randomly choose colors for the shapes (1-9)
        colors = random.sample(range(1, 10), num_shapes)
        
        # Shape sizes and spacing
        min_shape_size = 3  # Minimum 3x3 to ensure we can place a hollow
        spacing = 1
        
        # Calculate available space per shape
        shapes_per_row = int(np.ceil(np.sqrt(num_shapes)))
        available_space = (grid_size - (shapes_per_row + 1) * spacing) // shapes_per_row
        
        # Ensure max_shape_size is larger than min_shape_size
        max_shape_size = max(min_shape_size + 1, min(5, available_space))
        
        if max_shape_size <= min_shape_size:
            # If we can't fit shapes of minimum size, try again with larger grid
            return self.create_input()
        
        # Decide which shape will have the hollow
        hollow_shape_index = random.randint(0, num_shapes - 1)
        
        # Plan shape positions in a grid-like arrangement
        positions = []
        shape_counter = 0
        
        # Create shapes in a grid-like arrangement
        for i in range(shapes_per_row):
            for j in range(shapes_per_row):
                if shape_counter >= num_shapes:
                    break
                
                # Determine shape size
                shape_height = random.randint(min_shape_size, max_shape_size)
                shape_width = random.randint(min_shape_size, max_shape_size)
                
                # Calculate position with proper spacing
                row = i * (max_shape_size + spacing) + spacing
                col = j * (max_shape_size + spacing) + spacing
                
                # Ensure we don't go out of bounds
                if row + shape_height >= grid_size or col + shape_width >= grid_size:
                    continue
                
                positions.append((row, col, shape_height, shape_width))
                shape_counter += 1
                
                if shape_counter >= num_shapes:
                    break
        
        # If we couldn't place enough shapes, adjust num_shapes
        num_shapes = len(positions)
        if num_shapes == 0:
            # If we couldn't place any shapes, retry with smaller sizes
            return self.create_input()
        
        # Ensure hollow_shape_index is valid
        hollow_shape_index = random.randint(0, num_shapes - 1)
        
        # Create the shapes in the grid
        for i, (row, col, height, width) in enumerate(positions):
            color = colors[i % len(colors)]
            
            # Create solid shape
            for r in range(row, row + height):
                for c in range(col, col + width):
                    grid[r, c] = color
            
            # Add hollow block if this is the chosen shape
            if i == hollow_shape_index:
                # Pick a random cell in the interior (not on the border)
                if height >= 3 and width >= 3:  # Ensure shape is large enough
                    hollow_row = random.randint(row + 1, row + height - 2)
                    hollow_col = random.randint(col + 1, col + width - 2)
                    grid[hollow_row, hollow_col] = 0
        
        # Double-check if a hollow was created
        has_hollow = False
        for r in range(1, grid_size - 1):
            for c in range(1, grid_size - 1):
                if grid[r, c] == 0 and grid[r-1, c] != 0 and grid[r+1, c] != 0 and grid[r, c-1] != 0 and grid[r, c+1] != 0:
                    has_hollow = True
                    break
            if has_hollow:
                break
        
        if not has_hollow:
            # If no hollow was created, try again
            return self.create_input()
        
        return grid
    
    def transform_input(self, input_grid):
        # Find connected objects in the grid
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        rows, cols = input_grid.shape
        
        hollow_color = None
        # Examine each object to find the one with a hollow
        for obj in objects.objects:
            # Get the object's color - use next() since we know objects are monochromatic
            color = next(iter(obj.colors))
            if color == 0:
                continue
                
            # Get object's bounding box
            min_r = min(r for r, c, _ in obj.cells)
            max_r = max(r for r, c, _ in obj.cells)
            min_c = min(c for r, c, _ in obj.cells)
            max_c = max(c for r, c, _ in obj.cells)
            
            # Check interior cells for hollow
            found_hollow = False
            for r in range(min_r + 1, max_r):
                for c in range(min_c + 1, max_c):
                    if input_grid[r, c] == 0:  # Potential hollow
                        # Check if surrounded by this object's color
                        neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                        is_hollow = True
                        for nr, nc in neighbors:
                            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or input_grid[nr, nc] != color:
                                is_hollow = False
                                break
                        
                        if is_hollow:
                            hollow_color = color
                            found_hollow = True
                            break
                if found_hollow:
                    break
            
            if found_hollow:
                break
        
        if hollow_color is None:
            raise ValueError("No shape with hollow found in the input grid")
        
        # Create 1x1 output grid with the hollow shape's color
        output_grid = np.array([[hollow_color]], dtype=int)
        return output_grid
    
    def create_grids(self):
        # Generate 3-5 train examples and 1 test example
        num_train = random.randint(3, 5)
        
        train_pairs = []
        for _ in range(num_train):
            input_grid = self.create_input()
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        test_input = self.create_input()
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]   
        
        gridvars = {}
        return gridvars, TrainTestData(train = train_pairs, test = test_pairs)
