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
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a grid with exactly 4 scattered cells of object_color."""
        object_color = taskvars["object_color"]
        grid_size = taskvars["grid_size"]
        
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
                    r = random.randint(2, grid_size - 3)  # Increased margin from edges
                    c = random.randint(2, grid_size - 3)
                    
                    # Check if this position is far enough from existing cells
                    # Increased minimum distance to ensure boundary blocks don't touch
                    min_distance = 4  # This ensures at least 1 empty cell between boundary blocks
                    if all(abs(r - cr) >= min_distance or abs(c - cc) >= min_distance for cr, cc in placed_cells):
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

    def transform_input(self, grid, taskvars):
        """Transform input by creating boundaries around object cells."""
        object_color = taskvars["object_color"]
        fill_color_4way = taskvars["fill_color_4way"]
        # 8-way connected cells use the same color as object_color
        fill_color_8way = object_color
        
        # Create empty output grid (all zeros)
        output_grid = np.zeros_like(grid)
        
        # Find the 4 object cells from input grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
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
                if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and 
                    output_grid[nr, nc] == 0):  # Only if cell is empty
                    output_grid[nr, nc] = fill_color_4way
            
            for nr, nc in eight_way:
                if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and 
                    output_grid[nr, nc] == 0):  # Only if cell is empty
                    output_grid[nr, nc] = fill_color_8way  # This is object_color
        
        return output_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Choose random grid size - increased minimum to accommodate spacing
        grid_size = random.randint(10, 15)  # Increased from 7-15 to 10-15
        
        # Pick distinct colors for objects and 4-way boundary
        # We only need 2 distinct colors since 8-way uses object_color
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Store the variables for the task
        taskvars = {
            "object_color": available_colors[0],
            "fill_color_4way": available_colors[1],
            "grid_size": grid_size,
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
    generator = Taskb60334d2Generator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)