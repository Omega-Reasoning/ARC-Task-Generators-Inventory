from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Tuple
from transformation_library import GridObject, find_connected_objects
from input_library import create_object, enforce_object_height, enforce_object_width, retry

class Taska5313dffGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid with various dimensions.",
            "The objects which are 4-way connected are either square or rectangle.",
            "The object must be in red color.",
            "If the object is a square, then it must contain one cell of color red exactly in the center.",
            "The object can have extended lines not forming an enclosed shape."
        ]
        
        transformation_reasoning_chain = [
            "The output grid must have the same size as the input grid.",
            "The object remains the same in the output grid.",
            "The only difference in the output grid is the color of the enclosure ring of the object is changed to blue.",
            "The enclosure of the ring is not touching the red cell in the center of the object.",
            "The color of the enclosure ring or enclosure is changed to blue only when the object is enclosed by red cells and not by the grid edges."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with red squares or rectangles."""
        rows = gridvars['grid_size']
        
        # Create empty grid
        grid = np.zeros((rows, rows), dtype=int)
        
        # Define red color
        red = 1
        
        # Randomly decide if we'll create a square or rectangle
        is_square = random.choice([True, False])
        
        if is_square:
            # Create a square with odd dimensions to have a clear center
            size = random.choice([3, 5, 7])
            if size > rows:
                size = rows
            
            # Create square object
            square = np.zeros((size, size), dtype=int)
            
            # Fill the border with red
            square[0, :] = red
            square[-1, :] = red
            square[:, 0] = red
            square[:, -1] = red
            
            # Add center cell if size is odd
            if size % 2 == 1:
                center = size // 2
                square[center, center] = red
            
            # Randomly position the square in the grid
            max_row = rows - size
            max_col = rows - size
            if max_row > 0 and max_col > 0:
                start_row = random.randint(0, max_row)
                start_col = random.randint(0, max_col)
            else:
                start_row = 0
                start_col = 0
            
            # Place the square in the grid
            grid[start_row:start_row+size, start_col:start_col+size] = square
            
        else:
            # Create a rectangle (not square)
            height = random.randint(2, rows-1)
            width = random.randint(2, rows-1)
            
            # Ensure it's not a square
            while height == width:
                height = random.randint(2, rows-1)
                width = random.randint(2, rows-1)
            
            # Create rectangle object
            rect = np.zeros((height, width), dtype=int)
            
            # Fill the border with red
            rect[0, :] = red
            rect[-1, :] = red
            rect[:, 0] = red
            rect[:, -1] = red
            
            # Randomly position the rectangle in the grid
            max_row = rows - height
            max_col = rows - width
            if max_row > 0 and max_col > 0:
                start_row = random.randint(0, max_row)
                start_col = random.randint(0, max_col)
            else:
                start_row = 0
                start_col = 0
            
            # Place the rectangle in the grid
            grid[start_row:start_row+height, start_col:start_col+width] = rect
        
        # Sometimes add extended lines (50% chance)
        if random.random() < 0.5:
            objects = find_connected_objects(grid)
            if len(objects) > 0:
                obj = objects[0]  # Get the main object
                
                # Choose a random border cell to extend from
                border_cells = []
                for r, c, color in obj.cells:
                    # Check if it's a border cell (has at least one adjacent background cell)
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < rows and grid[nr, nc] == 0:
                            border_cells.append((r, c))
                            break
                
                if border_cells:
                    start_r, start_c = random.choice(border_cells)
                    
                    # Decide line direction (horizontal or vertical)
                    if random.choice([True, False]):
                        # Horizontal line
                        length = random.randint(1, rows - start_c - 1)
                        grid[start_r, start_c:start_c+length] = red
                    else:
                        # Vertical line
                        length = random.randint(1, rows - start_r - 1)
                        grid[start_r:start_r+length, start_c] = red
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by changing enclosed red border cells to blue."""
        output_grid = grid.copy()
        rows = output_grid.shape[0]
        red = 1
        blue = 2
        
        # Find all red objects
        objects = find_connected_objects(output_grid)
        red_objects = objects.with_color(red)
        
        for obj in red_objects:
            # Check if the object is fully enclosed by red cells (not touching grid edges)
            min_row, max_row = obj.bounding_box[0].start, obj.bounding_box[0].stop - 1
            min_col, max_col = obj.bounding_box[1].start, obj.bounding_box[1].stop - 1
            
            # Object is enclosed if it doesn't touch grid edges
            enclosed = (min_row > 0 and max_row < rows - 1 and 
                       min_col > 0 and max_col < rows - 1)
            
            if enclosed:
                # Get all border cells of the object's bounding box
                border_cells = set()
                
                # Top and bottom borders
                for c in range(min_col, max_col + 1):
                    border_cells.add((min_row, c))
                    border_cells.add((max_row, c))
                
                # Left and right borders (excluding corners already added)
                for r in range(min_row + 1, max_row):
                    border_cells.add((r, min_col))
                    border_cells.add((r, max_col))
                
                # Change color of border cells that are red to blue
                for r, c in border_cells:
                    if output_grid[r, c] == red:
                        output_grid[r, c] = blue
        
        return output_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # No task-level variables needed for this task
        taskvars = {}
        
        # Create 3-6 train examples (randomly chosen)
        num_train_examples = random.randint(3, 6)
        train_examples = []
        
        # Generate different grid sizes
        grid_sizes = [random.choice([5, 7, 9, 11, 13, 15]) for _ in range(num_train_examples + 1)]
        
        for i in range(num_train_examples):
            gridvars = {'grid_size': grid_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': grid_sizes[-1]}
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