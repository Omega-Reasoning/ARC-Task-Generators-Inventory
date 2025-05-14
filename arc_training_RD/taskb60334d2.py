from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, BorderBehavior, CollisionBehavior, GridObject, GridObjects
import numpy as np
import random

class Taskb60334d2Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size MXM.",
            "The grid consists of exactly 4 cells of {{color(\"object_color\")}} color.",
            "They are scattered in such a way that it can form a boundary around each cell.",
            "Boundaries never overlap with existing cells or other boundaries."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The original cells become empty (color 0) in the output.",
            "The boundaries have specific colors based on their position with center cell:",
            "- If the boundary forming cells are 4-way connected to main cell, then fill it with {{color(\"object_color\")}} color",
            "- If the boundary forming cells are 8-way connected to the main cell then fill it with {{color(\"fill_color\")}} color."
        ]
        
        taskvars_definitions = {
            "object_color": "The color of the main cells that form the objects",
            "fill_color_4way": "The color used for the 4-way connected boundary cells",
            "fill_color_8way": "The color used for the 8-way connected boundary cells",
            "grid_size": "The size of the grid (MxM)"
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, grid_size=None, object_color=None, existing_grid=None):
        if grid_size is None:
            grid_size = random.randint(7, 15)  # Randomizing grid size between 7 and 15
        
        if object_color is None:
            object_color = random.randint(1, 9)  # Randomizing object color between 1 and 9
        
        # Initialize grid if not provided
        if existing_grid is None:
            grid = np.zeros((grid_size, grid_size), dtype=int)
        else:
            grid = existing_grid.copy()
        
        # We need to place 4 random cells with enough spacing between them
        def place_cells_with_spacing():
            # Reset grid to ensure we're starting fresh
            grid.fill(0)
            
            # Try to place 4 cells with enough spacing
            placed_cells = []
            for _ in range(4):
                for attempt in range(100):  # Attempt 100 times to place a cell
                    r = random.randint(1, grid_size - 2)  # Avoid edges to allow for boundary
                    c = random.randint(1, grid_size - 2)
                    
                    # Check if this position is far enough from existing cells
                    if all(abs(r - cr) > 2 or abs(c - cc) > 2 for cr, cc in placed_cells):
                        grid[r, c] = object_color
                        placed_cells.append((r, c))
                        break
                else:
                    # If we couldn't place a cell after 100 attempts, start over
                    return False
            
            # Check if we've successfully placed all 4 cells
            return len(placed_cells) == 4
        
        # Try to place cells with spacing until successful
        success = retry(place_cells_with_spacing, lambda x: x, max_attempts=100)
        
        if not success:
            raise ValueError("Failed to generate a valid input grid with proper spacing")
        
        return grid

    def transform_input(self, input_grid, vars=None):
        # Create empty output grid (all zeros)
        output_grid = np.zeros_like(input_grid)
        
        if vars is None:
            # If vars not provided, infer from the input grid
            object_color = None
            for color in range(1, 10):  # Find the color used for objects
                if np.sum(input_grid == color) == 4:  # There should be exactly 4 cells
                    object_color = color
                    break
            
            if object_color is None:
                raise ValueError("Could not determine object color from input grid")
            
            # Choose colors for boundaries that are different from object_color
            possible_colors = [c for c in range(1, 10) if c != object_color]
            fill_color_4way = possible_colors[0]
            fill_color_8way = possible_colors[1]
        else:
            object_color = vars["object_color"]
            fill_color_4way = vars["object_color"]
            fill_color_8way = vars["fill_color_8way"]
        
        # Find the 4 object cells from input grid
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=0)
        object_cells = [(r, c) for obj in objects.objects 
                        for r, c, col in obj.cells if col == object_color]
        
        # For each object cell, create its boundary
        for r, c in object_cells:
            # Main cell remains empty (0)
            output_grid[r, c] = 0
            
            # Check 4-way connections (orthogonal neighbors)
            four_way = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            
            # Check 8-way connections (diagonal neighbors)
            eight_way = [(r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)]
            
            # Add boundary cells with appropriate colors
            for nr, nc in four_way:
                if (0 <= nr < input_grid.shape[0] and 0 <= nc < input_grid.shape[1] and 
                    output_grid[nr, nc] == 0):  # Only if cell is empty
                    output_grid[nr, nc] = fill_color_4way
            
            for nr, nc in eight_way:
                if (0 <= nr < input_grid.shape[0] and 0 <= nc < input_grid.shape[1] and 
                    output_grid[nr, nc] == 0):  # Only if cell is empty
                    output_grid[nr, nc] = fill_color_8way
        
        return output_grid
    
    def create_grids(self):
        # Choose random parameters for the task
        grid_size = random.randint(7, 15)
        
        # Pick distinct colors for objects and boundaries
        colors = random.sample(range(1, 10), 3)
        object_color = colors[0]
        fill_color_4way = colors[1]
        fill_color_8way = colors[2]
        
        # Store the variables for the task
        vars_dict = {
            "object_color": object_color,
            "fill_color_4way": fill_color_4way,
            "fill_color_8way": fill_color_8way,
            "grid_size": f"{grid_size}x{grid_size}"
        }
        
        # Generate train and test pairs
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train_pairs):
            input_grid = self.create_input(grid_size=grid_size, object_color=object_color)
            output_grid = self.transform_input(input_grid, vars=vars_dict)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(grid_size=grid_size, object_color=object_color)
        test_output = self.transform_input(test_input, vars=vars_dict)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return vars_dict, TrainTestData(train=train_pairs, test=test_pairs)