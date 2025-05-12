from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import Contiguity, retry, create_object, random_cell_coloring

class ScatteredCellsBoundaryGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can be of different sizes.",
            "The grid consists of many scattered cells of red color. Most of them are properly scattered and away from each other. But very few of them have only 1 cell adjacent to them."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The scattered cells without adjacent cells are displayed as it is.",
            "The scattered cells with 1 adjacent cell have a boundary forming around it of blue color.",
            "If the 2 cells are occurring on the edges of the grid, form a boundary of 3 sides."
        ]
        
        taskvars_definitions = {
            "object_color": "The color of the scattered cells (1-9)",
            "bound_color": "The color of the boundary (1-9, different from object_color)"
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars):
        object_color = taskvars["object_color"]
        
        # Generate a random grid size between 5 and 20
        height = random.randint(5, 20)
        width = random.randint(5, 20)
        
        # Create an empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Calculate number of adjacent pairs based on grid size (reduced ratio)
        grid_area = height * width
        min_pairs = 1  # Always at least 1 pair
        max_pairs = max(2, grid_area // 40)  # Significantly reduced ratio for pairs
        num_pairs = random.randint(min_pairs, max_pairs)
        
        placed_cells = 0
        pairs_placed = 0
        max_attempts = 100
        
        # First, place the required number of adjacent pairs
        while pairs_placed < num_pairs and max_attempts > 0:
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            
            # Check if this location and its surroundings are empty
            if grid[r, c] == 0:
                valid_surroundings = True
                # Check 5x5 neighborhood to ensure pairs are well-spaced
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < height and 0 <= nc < width and 
                            grid[nr, nc] != 0):
                            valid_surroundings = False
                            break
                    if not valid_surroundings:
                        break
                
                if valid_surroundings:
                    # Try to place second cell of the pair
                    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                    random.shuffle(directions)
                    
                    for dr, dc in directions:
                        new_r, new_c = r + dr, c + dc
                        if (0 <= new_r < height and 0 <= new_c < width and 
                            grid[new_r, new_c] == 0):
                            # Place the adjacent pair
                            grid[r, c] = object_color
                            grid[new_r, new_c] = object_color
                            placed_cells += 2
                            pairs_placed += 1
                            break
            
            max_attempts -= 1

        # Calculate remaining scattered cells (increased ratio)
        min_cells = max(8, height * width // 8)  # Increased minimum cells
        max_cells = max(min_cells + 2, height * width // 6)  # Increased maximum cells
        num_scattered_cells = random.randint(min_cells, max_cells)
        
        # Place remaining cells with spacing (no adjacency)
        attempts = 0
        while placed_cells < num_scattered_cells and attempts < 100:
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            
            # Check 3x3 neighborhood to ensure proper spacing
            valid = True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width and 
                        grid[nr, nc] != 0):
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid and grid[r, c] == 0:
                grid[r, c] = object_color
                placed_cells += 1
            
            attempts += 1
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Extract color variables
        object_color = taskvars["object_color"]
        bound_color = taskvars["bound_color"]
        
        # Create a copy of the input grid for the output
        output_grid = grid.copy()
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Filter for objects of the specified color
        colored_objects = objects.with_color(object_color)
        
        height, width = grid.shape
        
        # Process each connected object
        for obj in colored_objects:
            # We only add boundaries to pairs of adjacent cells
            if len(obj) == 2:
                coords = list(obj.coords)
                
                # Create a boundary around the entire pair
                boundary_positions = set()
                
                # For each cell in the pair
                for r, c in coords:
                    # Check all 8 surrounding positions
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip the cell itself
                            
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                # Only add boundary if this position isn't part of the object
                                if (nr, nc) not in coords and output_grid[nr, nc] == 0:
                                    boundary_positions.add((nr, nc))
                
                # Add the boundary in the output grid
                for r, c in boundary_positions:
                    output_grid[r, c] = bound_color
        
        return output_grid
    
    def create_grids(self):
        # Randomly choose colors
        color_options = list(range(1, 10))
        object_color = random.choice(color_options)
        
        # Make sure bound_color is different from object_color
        remaining_colors = [c for c in color_options if c != object_color]
        bound_color = random.choice(remaining_colors)
        
        taskvars = {
            "object_color": object_color,
            "bound_color": bound_color
        }
        
        # Determine number of training examples
        num_train = random.randint(3, 5)
        
        # Generate training pairs
        train_pairs = []
        for i in range(num_train):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input, taskvars)  # Added taskvars parameter
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
