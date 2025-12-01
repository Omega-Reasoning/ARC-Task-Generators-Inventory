from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task6b9890afGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "They contain exactly two colored objects: the first is a one-cell wide {color('frame_color')} square frame, and the second is of a different color and shape.",
            "The {color('frame_color')} square frame is of size (m+2)x(m+2), where m is a multiple of 3 and its value varies across examples.",
            "The second object has a different shape in each input grid and is made of 8-way connected cells, sized so that it fits within a 3x3 grid.",
            "The two objects are completely separated, and the second object can never appear inside the {color('frame_color')} square frame."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by identifying the two colored objects: the first is a one-cell wide {color('frame_color')} square frame, and the second is of a different color and shape confined within a 3x3 subgrid.",
            "Once the {color('frame_color')} square frame has been identified, subtract 2 from its side length and divide the result by 3 — this value is referred to as the scale number.",
            "The output grid is initialized with the same dimensions as the {color('frame_color')} square frame. The frame is then directly copied from the input and placed into the output grid.",
            "Then, identify the 3x3 subgrid from the input grid that contains the second object.",
            "Expand each of its cells by the scale number in both the horizontal and vertical directions to match the size of the frame, and place the scaled object inside the frame in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        """Create an input grid with a frame and a separate small object."""
        frame_color = taskvars['frame_color']
        object_color = gridvars['object_color']
        m = gridvars['m']  # Frame size (m+2)x(m+2) where m is a multiple of 3
        
        # Create a small object that fits within a 3x3 grid
        def generate_small_object():
            # Create a random connected object
            obj = create_object(3, 3, object_color, Contiguity.EIGHT, background=0)
            
            # Ensure object has at least one cell in each row and column
            has_cells_in_rows = np.any(obj != 0, axis=1)
            has_cells_in_cols = np.any(obj != 0, axis=0)
            
            if np.all(has_cells_in_rows) and np.all(has_cells_in_cols):
                return obj
            return None
        
        small_object = retry(
            generate_small_object,
            lambda x: x is not None,
            max_attempts=100
        )
        
        # Calculate frame dimensions
        frame_size = m + 2  # External size of the frame
        
        # Create grid with enough space for all elements
        grid_size = frame_size + 10  # More space to ensure frame isn't at the border
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine frame placement with padding from borders
        frame_padding = random.randint(2, 4)
        frame_start_row = frame_padding
        frame_start_col = frame_padding
        
        # Create frame with padding from borders
        for i in range(frame_size):
            for j in range(frame_size):
                r, c = frame_start_row + i, frame_start_col + j
                if (i == 0 or i == frame_size-1 or j == 0 or j == frame_size-1):
                    grid[r, c] = frame_color
        
        # Calculate frame boundaries
        frame_end_row = frame_start_row + frame_size - 1
        frame_end_col = frame_start_col + frame_size - 1
        
        # Find valid positions for the small object (not overlapping with frame and with buffer)
        positions = []
        buffer = 1  # Buffer space between objects
        for i in range(grid_size - 3):
            for j in range(grid_size - 3):
                # Check if position is valid (not overlapping with frame + buffer)
                valid = True
                for di in range(3 + 2*buffer):
                    for dj in range(3 + 2*buffer):
                        r, c = i - buffer + di, j - buffer + dj
                        if (frame_start_row <= r <= frame_end_row and 
                            frame_start_col <= c <= frame_end_col):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    positions.append((i, j))
        
        # Place small object at random valid position
        if positions:
            pos_i, pos_j = random.choice(positions)
            h, w = small_object.shape
            grid[pos_i:pos_i+h, pos_j:pos_j+w] = small_object
        
        # Find furthest extent of both objects
        max_row = max(frame_end_row + 1, pos_i + h)
        max_col = max(frame_end_col + 1, pos_j + w)
        
        # Trim grid to appropriate size (with small padding)
        padding = random.randint(1, 3)
        max_row = min(max_row + padding, grid_size)
        max_col = min(max_col + padding, grid_size)
        grid = grid[:max_row, :max_col]
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform the input grid by scaling the small object."""
        frame_color = taskvars['frame_color']
        
        # Find the frame and the small object
        objects = find_connected_objects(grid, diagonal_connectivity=True)
        
        # Get the frame (largest object with frame_color)
        frame_objects = objects.with_color(frame_color)
        frame = frame_objects.sort_by_size(reverse=True)[0]
        
        # Get the second object (not the frame)
        non_frame_objects = objects.filter(lambda obj: frame_color not in obj.colors)
        small_object = non_frame_objects[0]
        
        # Determine the frame size and calculate the scale number
        frame_bbox = frame.bounding_box
        frame_size = frame_bbox[0].stop - frame_bbox[0].start
        m = frame_size - 2  # Subtract 2 to get m
        scale = m // 3
        
        # Create output grid with same size as the frame
        output_grid = np.zeros((frame_size, frame_size), dtype=int)
        
        # Extract the frame as a subgrid from the input grid
        frame_rows = frame_bbox[0]
        frame_cols = frame_bbox[1]
        
        # Copy the frame to output grid
        for r in range(frame_rows.start, frame_rows.stop):
            for c in range(frame_cols.start, frame_cols.stop):
                if grid[r, c] == frame_color:
                    output_grid[r - frame_rows.start, c - frame_cols.start] = frame_color
        
        # Extract the small object array and its color
        small_obj_array = small_object.to_array()
        small_obj_color = list(small_object.colors)[0]
        
        # Scale the small object
        scaled_obj = np.zeros((m, m), dtype=int)
        for r in range(small_obj_array.shape[0]):
            for c in range(small_obj_array.shape[1]):
                if small_obj_array[r, c] != 0:
                    # Expand this cell by scale in each direction
                    for dr in range(scale):
                        for dc in range(scale):
                            scaled_obj[r*scale + dr, c*scale + dc] = small_obj_color
        
        # Place scaled object inside the frame
        output_grid[1:1+m, 1:1+m] = scaled_obj
        
        return output_grid

    def create_grids(self):
        """Create training and test grids."""
        # Define task variables
        frame_color = random.randint(1, 9)
        
        taskvars = {
            'frame_color': frame_color
        }
        
        # Create training examples
        num_train_examples = random.randint(3, 4)
        train_examples = []
        used_colors = set([frame_color])
        used_m_values = set()
        
        for _ in range(num_train_examples):
            # Select a different color for the object each time
            object_color = random.choice([c for c in range(1, 10) if c != frame_color and c not in used_colors])
            used_colors.add(object_color)
            
            # Choose m as a multiple of 3 (between 3 and 18)
            possible_m_values = [m for m in [3, 6, 9, 12, 15, 18] if m not in used_m_values]
            if not possible_m_values:  # If we've used all values, allow reuse
                possible_m_values = [3, 6, 9, 12, 15, 18]
            
            m = random.choice(possible_m_values)
            used_m_values.add(m)
            
            gridvars = {
                'object_color': object_color,
                'm': m
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        object_color = random.choice([c for c in range(1, 10) if c != frame_color])
        m = random.choice([3, 6, 9, 12, 15, 18])
        
        gridvars = {
            'object_color': object_color,
            'm': m
        }
        
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

