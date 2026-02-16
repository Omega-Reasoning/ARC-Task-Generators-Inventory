from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity
from Framework.transformation_library import find_connected_objects

class Task444801d8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They consist of atleast one {color('object_color')} rectangular frame, each fully separated from the others.",
            "All {color('object_color')} rectangular frames have a fixed length of five and a varying width, always greater than two.",
            "Each frame has an empty(0) cell exactly in the middle of its top edge and a single-colored cell positioned directly below the missing cell.",
            "This single-colored cell is always a different color from the frame and varies across examples.",
            "Each {color('object_color')} rectangular frame must have all cells above its top edge completely empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all {color('object_color')} rectangular frames and the single-colored cell contained within them.",
            "Once identified, fill the empty (0) cell directly above the single-colored cell and all empty (0) interior cells of each frame with the color of the single-colored cell.",
            "Additionally, fill all empty cells in the row above the frame that are exactly above the top edge with the same color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Define task variables
        taskvars = {
            'grid_size': random.randint(11, 30),
            'object_color': random.randint(1, 9)
        }
        
        # Generate 3-4 training examples and 1 test example
        num_train_examples = random.randint(3, 4)
        
        train_data = []
        for _ in range(num_train_examples):
            # Create input grid with randomized frames
            gridvars = {
                'num_frames': random.randint(1, 3),
                'frame_colors': [color for color in range(1, 10) if color != taskvars['object_color']]
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with different random parameters
        gridvars = {
            'num_frames': random.randint(1, 3),
            'frame_colors': [color for color in range(1, 10) if color != taskvars['object_color']]
        }
        
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_data = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        num_frames = gridvars['num_frames']
        frame_colors = gridvars['frame_colors']
        
        # Initialize an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Track placed frames to avoid overlap
        placed_frames = []
        
        # Create the specified number of frames
        for _ in range(num_frames):
            max_attempts = 100
            for attempt in range(max_attempts):
                # Random frame dimensions (fixed length of 5, width > 2)
                frame_length = 5
                frame_width = random.randint(3, min(8, grid_size - 4))
                
                # Random position for the frame
                # Ensure frame is not at the very edge and has space above it
                max_row = grid_size - frame_width - 2
                max_col = grid_size - frame_length - 2
                
                if max_row < 4 or max_col < 4:  # Not enough space
                    continue
                    
                top_row = random.randint(3, max_row)
                left_col = random.randint(2, max_col)
                
                # Define frame boundaries
                frame_rows = (top_row, top_row + frame_width - 1)
                frame_cols = (left_col, left_col + frame_length - 1)
                
                # Check for overlap with existing frames (including buffer space)
                overlap = False
                for existing_frame in placed_frames:
                    existing_rows, existing_cols = existing_frame
                    
                    # Add buffer around existing frame
                    if (frame_rows[0] - 1 <= existing_rows[1] + 1 and 
                        frame_rows[1] + 1 >= existing_rows[0] - 1 and
                        frame_cols[0] - 1 <= existing_cols[1] + 1 and
                        frame_cols[1] + 1 >= existing_cols[0] - 1):
                        overlap = True
                        break
                
                if overlap:
                    continue
                
                # Check if area above frame is empty
                if np.any(grid[0:top_row, left_col:left_col+frame_length] != 0):
                    continue
                
                # We've found a valid position, create the frame
                
                # Choose a random color for the cell inside the frame (different from frame color)
                inner_color = random.choice([c for c in frame_colors if c != object_color])
                
                # Draw the frame
                for r in range(top_row, top_row + frame_width):
                    for c in range(left_col, left_col + frame_length):
                        # Only draw the perimeter of the frame
                        if (r == top_row or r == top_row + frame_width - 1 or 
                            c == left_col or c == left_col + frame_length - 1):
                            grid[r, c] = object_color
                
                # Calculate middle of top edge and leave it empty
                middle_col = left_col + frame_length // 2
                grid[top_row, middle_col] = 0  # Empty cell in the middle of top edge
                
                # Place the single-colored cell below the empty cell
                grid[top_row + 1, middle_col] = inner_color
                
                # Add this frame to our list of placed frames
                placed_frames.append((frame_rows, frame_cols))
                break
                
            if attempt == max_attempts - 1:
                # If we couldn't place all frames, just continue with what we have
                break
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid for the output
        output_grid = grid.copy()
        object_color = taskvars['object_color']
        
        # Find all connected objects with the frame color
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        frame_objects = objects.with_color(object_color)
        
        # Process each frame
        for frame_obj in frame_objects:
            # Get bounding box of the frame
            box = frame_obj.bounding_box
            top_row = box[0].start
            bottom_row = box[0].stop - 1
            left_col = box[1].start
            right_col = box[1].stop - 1
            
            # Find the middle column of the frame
            middle_col = (left_col + right_col) // 2
            
            # If this is a valid frame (has empty cell at top middle)
            if grid[top_row, middle_col] == 0:
                # Get the color of the cell directly below the empty cell
                if top_row + 1 < grid.shape[0]:
                    fill_color = grid[top_row + 1, middle_col]
                    
                    # Only proceed if we found a non-zero, non-frame color cell
                    if fill_color != 0 and fill_color != object_color:
                        # Fill the interior of the frame with this color
                        for r in range(top_row + 1, bottom_row):
                            for c in range(left_col + 1, right_col):
                                if output_grid[r, c] == 0:
                                    output_grid[r, c] = fill_color
                        
                        # Fill the empty cell at the top middle
                        output_grid[top_row, middle_col] = fill_color
                        
                        # Fill all empty cells in the row above the frame that are above the frame
                        if top_row > 0:
                            for c in range(left_col, right_col + 1):
                                if output_grid[top_row - 1, c] == 0:
                                    output_grid[top_row - 1, c] = fill_color
        
        return output_grid

