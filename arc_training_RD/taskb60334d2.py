from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, BorderBehavior, CollisionBehavior, GridObject, GridObjects
import numpy as np
import random

class Taskb60334d2Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids and can have size {vars['grid_size']} x {vars['grid_size']}.",
            "The grid consists of exactly 4 cells of {color('object_color')}.",
            "They are scattered in such a way that it can form a boundary around each cell.",
            "Boundaries never overlap with existing cells or other boundaries."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The original cells become empty (color 0) in the output.",
            "The boundaries have specific colors based on their position with center cell:",
            "If the boundary forming cells are 4-way connected to main cell, then fill it with {color('fill_color_4way')}",
            "If the boundary forming cells are 8-way connected to the main cell then fill it with {color('object_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def color_name(self, color: int) -> str:
        color_map = {
            0: "black",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "cyan",
            9: "brown"
        }
        return color_map.get(color, f"color_{color}")
    
    def create_input(self, taskvars):
        object_color = taskvars["object_color"]
        grid_size = taskvars["grid_size"]  # Now an integer from taskvars
        
        # Initialize grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
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

    def transform_input(self, input_grid, taskvars):
        object_color = taskvars["object_color"]
        fill_color_4way = taskvars["fill_color_4way"]
        # 8-way connected cells use the same color as object_color
        fill_color_8way = object_color
        
        # Create empty output grid (all zeros)
        output_grid = np.zeros_like(input_grid)
        
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
                    output_grid[nr, nc] = fill_color_8way  # This is object_color
        
        return output_grid
    
    def create_grids(self):
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        # Choose random grid size
        grid_size = random.randint(7, 15)
        
        # Pick distinct colors for objects and 4-way boundary
        # We only need 2 distinct colors since 8-way uses object_color
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Store the variables for the task
        taskvars = {
            "object_color": available_colors[0],
            "fill_color_4way": available_colors[1],
            "grid_size": grid_size,  # <-- integer for logic
        }
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Replace color and grid_size placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('object_color')}", color_fmt('object_color'))
                 .replace("{color('fill_color_4way')}", color_fmt('fill_color_4way'))
                 .replace("{vars['grid_size']} x {vars['grid_size']}", f"{taskvars['grid_size']} x {taskvars['grid_size']}")
            for chain in self.input_reasoning_chain
        ]
        
        self.transformation_reasoning_chain = [
            chain.replace("{color('object_color')}", color_fmt('object_color'))
                 .replace("{color('fill_color_4way')}", color_fmt('fill_color_4way'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate train pairs
        for _ in range(num_train_pairs):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input.copy(), taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)