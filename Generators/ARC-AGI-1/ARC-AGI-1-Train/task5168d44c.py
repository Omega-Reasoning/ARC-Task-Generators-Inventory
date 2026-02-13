from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, retry
from transformation_library import find_connected_objects, BorderBehavior

class Task5168d44cGenerator(ARCTaskGenerator):
    def __init__(self):
        # Initialize the input reasoning chain from the task specification
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a 3x3 {color('object_color')} frame and a {color('cell_color')} dashed line, where the dashed line alternates between {color('cell_color')} and empty (0) cells.",
            "The dashed line always starts with a {color('cell_color')} cell and can either be vertical or horizontal but is never positioned in the first or last row/column.",
            "The {color('object_color')} frame is placed so that its single empty (0) interior cell aligns with a {color('cell_color')} cell on the dashed line.",
            "This ensures that the {color('object_color')} frame surrounds a {color('cell_color')} cell.",
            "The {color('object_color')} frame is positioned so that at least one {color('cell_color')} cell appears before it and at least two {color('cell_color')} cells appear after it in the dashed line."
        ]
        
        # Initialize the transformation reasoning chain from the task specification with the modified shift amount
        transformation_reasoning_chain = [
            "Output grids are constructed by identifying the {color('cell_color')} dashed line and the one-cell wide {color('object_color')} frame surrounding a {color('cell_color')} cell.",
            "Once identified, shift the {color('object_color')} frame two cells to the right if the {color('cell_color')} dashed line is horizontal, or two cells downward if the dashed line is vertical."
        ]
        
        # Call the parent class constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'rows': random.randint(9, 30),  # Minimum 9 to ensure space for 3x3 frame
            'cols': random.randint(9, 30),  # Minimum 9 to ensure space for 3x3 frame
            'cell_color': random.randint(1, 9),
            'object_color': random.randint(1, 9)
        }
        
        # Ensure cell_color and object_color are different
        while taskvars['cell_color'] == taskvars['object_color']:
            taskvars['object_color'] = random.randint(1, 9)
        
        # Create 3-4 train examples and 1 test example
        num_train_examples = random.randint(3, 4)
        
        train_examples = []
        for _ in range(num_train_examples):
            # For each example, create new gridvars to randomize placement
            gridvars = {
                'is_horizontal': random.choice([True, False]),
                'line_position': None,  # Will be set in create_input
                'frame_position': None  # Will be set in create_input
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {
            'is_horizontal': random.choice([True, False]),
            'line_position': None,
            'frame_position': None
        }
        
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
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        cell_color = taskvars['cell_color']
        object_color = taskvars['object_color']
        is_horizontal = gridvars['is_horizontal']
        
        # Initialize an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Create a dashed line
        if is_horizontal:
            # Horizontal line - select a row not at the edges, with enough padding for the frame
            line_row = random.randint(2, rows - 3)
            
            # Fill the line with alternating colors
            for c in range(0, cols, 2):
                grid[line_row, c] = cell_color
            
            gridvars['line_position'] = line_row
        else:
            # Vertical line - select a column not at the edges, with enough padding for the frame
            line_col = random.randint(2, cols - 3)
            
            # Fill the line with alternating colors
            for r in range(0, rows, 2):
                grid[r, line_col] = cell_color
            
            gridvars['line_position'] = line_col
        
        # Find the colored cells in the dashed line
        colored_positions = []
        if is_horizontal:
            for c in range(0, cols, 2):
                if c < cols:
                    colored_positions.append((gridvars['line_position'], c))
        else:
            for r in range(0, rows, 2):
                if r < rows:
                    colored_positions.append((r, gridvars['line_position']))
        
        # Choose a position for the frame that:
        # 1. Allows complete 3x3 frame
        # 2. Has at least 2 more colored cells after it
        # 3. Has room for 2-cell shift
        # 4. Is in the first half of the line
        
        valid_positions = []
        
        for idx, (r, c) in enumerate(colored_positions):
            # Check if there's room for the complete 3x3 frame
            if r-1 < 0 or r+1 >= rows or c-1 < 0 or c+1 >= cols:
                continue
                
            # Check if there are at least 2 more colored cells after this one
            if idx >= len(colored_positions) - 2:
                continue
                
            # Check if there's room for the 2-cell shift
            if is_horizontal and c+2 >= cols - 1:
                continue
            if not is_horizontal and r+2 >= rows - 1:
                continue
                
            # Check if the position is in the first half of the line
            if idx >= len(colored_positions) // 2:
                continue
                
            valid_positions.append((idx, r, c))
        
        # If we have valid positions, choose one randomly
        if valid_positions:
            frame_pos_idx, frame_r, frame_c = random.choice(valid_positions)
            
            # Store the frame position
            gridvars['frame_position'] = (frame_r, frame_c)
            
            # Draw the complete 3x3 frame
            frame_coords = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        # Skip the center (it should remain with the dashed line color)
                        continue
                    new_r, new_c = frame_r + dr, frame_c + dc
                    frame_coords.append((new_r, new_c))
                    grid[new_r, new_c] = object_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        cell_color = taskvars['cell_color']
        object_color = taskvars['object_color']
        rows, cols = grid.shape
        
        # Find the dashed line
        is_horizontal = False
        line_position = -1
        
        # Check for horizontal dashed lines with the specific pattern
        for r in range(1, rows - 1):
            cells = grid[r, :]
            colored_indices = np.where(cells == cell_color)[0]
            if len(colored_indices) >= 2 and all(i % 2 == 0 for i in colored_indices):
                is_horizontal = True
                line_position = r
                break
        
        # If not horizontal, check for vertical with the specific pattern
        if not is_horizontal:
            for c in range(1, cols - 1):
                cells = grid[:, c]
                colored_indices = np.where(cells == cell_color)[0]
                if len(colored_indices) >= 2 and all(i % 2 == 0 for i in colored_indices):
                    is_horizontal = False
                    line_position = c
                    break
        
        # Find the frame
        # First, detect 3x3 regions where the center is a colored cell from the dashed line
        # and the surrounding 8 cells form a complete frame
        frame_center = None
        frame_cells = []
        
        if is_horizontal:
            for c in range(1, cols - 1):
                r = line_position
                # Check if this is a colored cell in the dashed line
                if grid[r, c] == cell_color:
                    # Check if it's surrounded by a complete frame
                    surrounding_cells = []
                    frame_valid = True
                    
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            if not (0 <= r+dr < rows and 0 <= c+dc < cols):
                                frame_valid = False
                                break
                            if grid[r+dr, c+dc] != object_color:
                                frame_valid = False
                                break
                            surrounding_cells.append((r+dr, c+dc))
                    
                    if frame_valid and len(surrounding_cells) == 8:
                        frame_center = (r, c)
                        frame_cells = surrounding_cells
                        break
        else:
            for r in range(1, rows - 1):
                c = line_position
                # Check if this is a colored cell in the dashed line
                if grid[r, c] == cell_color:
                    # Check if it's surrounded by a complete frame
                    surrounding_cells = []
                    frame_valid = True
                    
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            if not (0 <= r+dr < rows and 0 <= c+dc < cols):
                                frame_valid = False
                                break
                            if grid[r+dr, c+dc] != object_color:
                                frame_valid = False
                                break
                            surrounding_cells.append((r+dr, c+dc))
                    
                    if frame_valid and len(surrounding_cells) == 8:
                        frame_center = (r, c)
                        frame_cells = surrounding_cells
                        break
        
        # If we found a valid frame, shift it
        if frame_center and frame_cells:
            # Remove the original frame
            for r, c in frame_cells:
                output_grid[r, c] = 0
            
            # Determine the shift amount
            shift_r = 2 if not is_horizontal else 0
            shift_c = 2 if is_horizontal else 0
            
            # Draw the shifted frame
            for r, c in frame_cells:
                new_r, new_c = r + shift_r, c + shift_c
                if 0 <= new_r < rows and 0 <= new_c < cols:
                    output_grid[new_r, new_c] = object_color
        
        return output_grid

