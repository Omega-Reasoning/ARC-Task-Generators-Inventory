from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity
from Framework.transformation_library import find_connected_objects, BorderBehavior

class Task952a094cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid with dimension {vars['rows']} X {vars['rows']}.",
            "There is exactly one 4-way connected object which is either a square or rectangle with only perimeter of color in_color(between 1-9), the remaining cells of the object are empty(0), which forms an empty rectangle or square.",
            "Four random cells are placed inside the object of different colors(between 1-9) such that they are at corners of the empty square or rectangle."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "The 4-way connected object remains the same in the output grid.",
            "The colored cell on the bottom-left corner of the empty rectangle or square is placed on the top-right corner outside the object.",
            "The colored cell on the bottom-right corner of the empty rectangle or square is placed on the top-left corner outside the object.",
            "The colored cell on the top-left corner of the empty rectangle or square is placed on the bottom-right corner outside the object.",
            "The colored cell on the top-right corner of the empty rectangle or square is placed on the bottom-left corner outside the object."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Define task variables
        rows = random.randint(10, 30)  # Grid size between 10 and 20 (up to 30 allowed)
        
        # Create task variables dictionary
        taskvars = {
            'rows': rows,
        }
        
        # Generate 3-4 training examples
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        for _ in range(num_train_examples):
            input_color = random.randint(1, 9)
            gridvars = {
                'input_color': input_color
                }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate 1 test example
        input_color = random.randint(1, 9)
        gridvars = {
            'input_color': input_color
        }
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        input_color = gridvars['input_color']
        
        # Create empty grid
        grid = np.zeros((rows, rows), dtype=int)
        
        # Determine rectangle size and position with adequate margins
        min_rect_size = 4  # Minimum size to fit corner points
        max_rect_size = rows - 4  # Leave at least 2 cells margin on each side
        
        rect_height = random.randint(min_rect_size, max_rect_size)
        rect_width = random.randint(min_rect_size, max_rect_size)
        
        # Ensure there's at least 1 cell offset from grid edges
        max_start_row = rows - rect_height - 1
        max_start_col = rows - rect_width - 1
        
        start_row = random.randint(1, max_start_row)
        start_col = random.randint(1, max_start_col)
        
        end_row = start_row + rect_height - 1
        end_col = start_col + rect_width - 1
        
        # Draw the perimeter of the rectangle with input_color
        # Top and bottom edges
        grid[start_row, start_col:end_col+1] = input_color
        grid[end_row, start_col:end_col+1] = input_color
        
        # Left and right edges (excluding corners which are already drawn)
        grid[start_row+1:end_row, start_col] = input_color
        grid[start_row+1:end_row, end_col] = input_color
        
        # Generate 4 different colors for corners (different from input_color)
        available_colors = [c for c in range(1, 10) if c != input_color]
        corner_colors = random.sample(available_colors, 4)
        
        # Place corner points inside the rectangle
        # Top-left corner
        grid[start_row+1, start_col+1] = corner_colors[0]
        
        # Top-right corner
        grid[start_row+1, end_col-1] = corner_colors[1]
        
        # Bottom-left corner
        grid[end_row-1, start_col+1] = corner_colors[2]
        
        # Bottom-right corner
        grid[end_row-1, end_col-1] = corner_colors[3]
        
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Find the rectangle object
        objects = find_connected_objects(grid, diagonal_connectivity=False, monochromatic=False)
        rectangle = objects[0]  # Should be only one object
        
        # Get the bounding box of the rectangle
        box = rectangle.bounding_box
        start_row, end_row = box[0].start, box[0].stop - 1
        start_col, end_col = box[1].start, box[1].stop - 1
        
        # Find the corner points inside the rectangle
        top_left_corner = None
        top_right_corner = None
        bottom_left_corner = None
        bottom_right_corner = None
        
        # Scan the grid to find the colored corner points
        for r in range(start_row+1, end_row):
            for c in range(start_col+1, end_col):
                color = grid[r, c]
                if color == 0:  # Skip empty cells
                    continue
                    
                # Determine which corner this is
                if r < (start_row + end_row) / 2:  # Top half
                    if c < (start_col + end_col) / 2:  # Left half
                        top_left_corner = (r, c, color)
                    else:  # Right half
                        top_right_corner = (r, c, color)
                else:  # Bottom half
                    if c < (start_col + end_col) / 2:  # Left half
                        bottom_left_corner = (r, c, color)
                    else:  # Right half
                        bottom_right_corner = (r, c, color)
        
        # Clear the corner points from inside
        if top_left_corner:
            output_grid[top_left_corner[0], top_left_corner[1]] = 0
        if top_right_corner:
            output_grid[top_right_corner[0], top_right_corner[1]] = 0
        if bottom_left_corner:
            output_grid[bottom_left_corner[0], bottom_left_corner[1]] = 0
        if bottom_right_corner:
            output_grid[bottom_right_corner[0], bottom_right_corner[1]] = 0
        
        # Place the corners outside the rectangle according to the transformation rule
        # bottom-left -> top-right outside
        if bottom_left_corner:
            output_grid[start_row-1, end_col+1] = bottom_left_corner[2]
            
        # bottom-right -> top-left outside
        if bottom_right_corner:
            output_grid[start_row-1, start_col-1] = bottom_right_corner[2]
        
        # top-left -> bottom-right outside
        if top_left_corner:
            output_grid[end_row+1, end_col+1] = top_left_corner[2]
            
        # top-right -> bottom-left outside
        if top_right_corner:
            output_grid[end_row+1, start_col-1] = top_right_corner[2]
        
        return output_grid