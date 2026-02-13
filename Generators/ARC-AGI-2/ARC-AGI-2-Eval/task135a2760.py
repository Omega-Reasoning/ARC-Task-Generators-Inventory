from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects

class Task135a2760Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have {vars['cols']} columns, and the number of rows follows the form (m × n + 1), where m can be 4, 5, 6, or 7, and n is between 1 and 5.",
            "Each grid contains a completely filled background with same-colored cells, along with n {color('frame')} one-cell wide rectangular frames. Each one-cell wide rectangular frame starts at position (1, 1) and extends horizontally to position (1, {vars['cols'] - 2}). All rectangular frames are separated from each other by exactly one row of background color, and the last frame ends in the second-last row of the grid, with the last row being filled with background color.",
            "The width of each frame is (m - 1), matching the height allocation within the total row count defined by the formula (m × n + 1). Within each {color('frame')} one-cell wide rectangular frame, there is a specific arrangement of cells made up of the background color and one additional color t. This additional color t is different for each frame.",
            "To construct the specific arrangement within each rectangular frame, create a cyclic or repeated pattern having any of the following forms:",
            "Pattern 1 — Works for m = 4 with any number of columns. There is a horizontally alternating pattern enclosed within a 3×11 one-cell wide rectangular frame. The structure is: Row 1: [{color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}], Row 2: [{color('frame')}, t, background color, t, background color, t, background color, t, background color, t, {color('frame')}], Row 3: [{color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}, {color('frame')}]",
            "Pattern 2 — Works for m = 5, 7. There is a 2×2 checker-like tiling pattern enclosed within a {color('frame')} one-cell wide rectangular frame. The structure is: Row 1: [{color('frame')}, ..., {color('frame')}], Row 2: [{color('frame')}, t, t, background color, t, t, background color, ..., {color('frame')}], Row 3: [{color('frame')}, t, t, background color, t, t, background color, ..., {color('frame')}], Row 4: [{color('frame')}, ..., {color('frame')}]",
            "Pattern 3 — Works for m > 3. There is a checkerboard-like alternating pattern enclosed within a {color('frame')} one-cell wide rectangular frame. The structure is:\\n Row 1: {color('frame')}, ..., {color('frame')}\\n Row 2: {color('frame')}, t, background color, t, background color, t, background color, ..., {color('frame')}\\n Row 3: {color('frame')}, background color, t, background color, t, background color, t, ..., {color('frame')}, Row 4: {color('frame')}, ..., {color('frame')}",
            "Pattern 4 — Works for any m > 3. There is a horizontal stripe-based alternating pattern enclosed within a {color('frame')} one-cell wide rectangular frame. The structure is:\\n Row 1: {color('frame')}, ..., {color('frame')}\\n Row 2: {color('frame')}, t, background color, t, background color, t, background color, ..., {color('frame')}\\n Row 3: {color('frame')}, t, background color, t, background color, t, background color, ..., {color('frame')}, Row 4: {color('frame')}, ..., {color('frame')}",
            "Pattern 5 — Works for m = 5 and any m > 5. There is a sequence of connected U-shaped structures enclosed within a {color('frame')} one-cell wide rectangular frame. The structure is:\\n Row 1: {color('frame')}, {color('frame')}, {color('frame')}, ..., {color('frame')}, Row 2: {color('frame')}, t, background color, t, t, t, background color, t, t, background color, ..., {color('frame')}, Row 3: {color('frame')}, t, t, t, background color, t, t, t, background color, t, t, t, ..., {color('frame')}, Row 4: {color('frame')}, {color('frame')}, {color('frame')}, ..., {color('frame')}",
            "After the patterns are created, add either one or two background-colored cells that overlap color t cells, or one or two color t cells that overlap background cells.",
            "This results in {color('frame')} frames, each having a width of (m - 1) and a length of {vars['cols'] - 2}, with all frames separated by one background-colored row. The pattern within each frame should be different, and color t must be unique for each frame."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all one-cell wide rectangular frames placed in the grid.",
            "Once the frames are identified, detect the repeated or cyclic pattern within each frame.",
            "The pattern is generally uniform throughout, except for a few cells that disrupt the repetition.",
            "The goal is to recognize the undisturbed portion of the pattern for each frame and use it as a reference.",
            "After identifying the correct patterns within each frame, fix the distorted sections by recoloring a few cells so the pattern becomes fully uniform and cyclic throughout the grid.",
            "In the final output, each frame contains there own specific arrangement which is consistent, repeated pattern with no irregularities."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'cols': random.randint(15, 25),  # Grid width at least 15
            'frame': random.choice([color for color in range(1, 10) if color != 5])  # Frame color (not 5)
        }
        
        # Create train and test data with specific constraints
        num_train_examples = random.randint(3, 6)
        train_examples = []
        
        # Track already used values of m and patterns to ensure variety
        used_m_values = set()
        used_patterns = set()
        
        # Ensure we have required values of m
        required_m = [4, 5, 7]
        
        for i in range(num_train_examples):
            if i < len(required_m):
                m = required_m[i]
            else:
                # For any additional examples, choose from valid m values not used yet
                available_m = [m for m in [4, 5, 6, 7] if m not in used_m_values]
                if not available_m:  # If all values used, reset
                    available_m = [4, 5, 6, 7]
                m = random.choice(available_m)
            
            used_m_values.add(m)
            
            n = random.randint(1, 3)  # Choose a smaller n value to keep grid sizes reasonable
            
            # For m=4, only pattern 1 is valid
            if m == 4:
                pattern_type = 1
            # For m=5, patterns 2 and 5 are valid
            elif m == 5:
                # Use pattern 5 explicitly at least once for m=5
                if i == 1:  # For the second example (m=5)
                    pattern_type = 5
                else:
                    pattern_type = random.choice([2, 5])
            # For m=7, only pattern 2 is valid
            elif m == 7:
                pattern_type = 2
            # For m=6, use patterns 3, 4, or 5, but try to use different ones
            else:
                available_patterns = [p for p in [3, 4, 5] if p not in used_patterns]
                if not available_patterns:
                    available_patterns = [3, 4, 5]
                pattern_type = random.choice(available_patterns)
                used_patterns.add(pattern_type)
            
            # Generate input grid with pattern and disruptions
            input_grid, pattern_info = self.create_input_with_disruptions(taskvars, m, n, pattern_type, True)
            
            # Generate correct output grid by fixing disruptions
            output_grid = self.create_output(input_grid, pattern_info)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # For test example, use vertical arrangement with a different frame color
        m = random.choice([4, 5, 6, 7])
        n = random.randint(1, 3)
        test_frame_color = random.choice([c for c in range(1, 10) if c != taskvars['frame'] and c != 5])
        
        # Select pattern type based on m
        if m == 4:
            pattern_type = 1
        elif m == 5:
            pattern_type = random.choice([2, 5])  # For m=5, use pattern 2 or 5
        elif m == 7:
            pattern_type = 2
        else:
            pattern_type = random.choice([3, 4, 5])
            
        # Generate test input grid with vertical orientation
        test_input, test_pattern_info = self.create_input_with_disruptions(
            taskvars, m, n, pattern_type, False, frame_color=test_frame_color
        )
        
        # Generate correct test output grid
        test_output = self.create_output(test_input, test_pattern_info)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input_with_disruptions(self, taskvars, m, n, pattern_type, horizontal=True, frame_color=None):
        """Create input grid with pattern and intentional disruptions, and return disruption info"""
        try:
            if frame_color is None:
                frame_color = taskvars['frame']
            
            cols = taskvars['cols']
            
            # Choose background color (not same as frame color)
            background_color = random.choice([c for c in range(1, 10) if c != frame_color and c != 5])
            
            if horizontal:
                rows = m * n + 1  # Formula for horizontal arrangement
                grid = np.full((rows, cols), background_color, dtype=int)
                
                # Store frame info for later
                frames_info = []
                
                # Track used patterns and colors to ensure variety and uniqueness within the grid
                used_patterns = set()
                used_pattern_colors = set()
                
                # Create frames
                for i in range(n):
                    # First frame starts at position (1, 1)
                    frame_start_row = 1 + i * m  # First frame starts at row 1, then each m rows
                    frame_height = m - 1  # Frame height is m-1
                    
                    # Choose a unique pattern color (different from background, frame, and previously used patterns)
                    available_colors = [c for c in range(1, 10) 
                                     if c != background_color and c != frame_color and c != 5 
                                     and c not in used_pattern_colors]
                    
                    # If somehow we've used all colors, pick one we haven't used in this grid yet
                    if not available_colors:
                        # This is an edge case - try to find any usable color
                        available_colors = [c for c in range(1, 10) 
                                         if c != background_color and c != frame_color and c != 5]
                        if not available_colors:
                            # Absolute fallback
                            pattern_color = (background_color + 1) % 9 + 1
                            if pattern_color == 5:
                                pattern_color = 6
                        else:
                            pattern_color = random.choice(available_colors)
                    else:
                        pattern_color = random.choice(available_colors)
                    
                    # Track this pattern color as used in this grid
                    used_pattern_colors.add(pattern_color)
                    
                    # Determine which patterns can be used for this frame based on m
                    valid_patterns = []
                    if m == 4:
                        valid_patterns = [1]  # Only pattern 1 is valid for m=4
                    elif m == 5:
                        valid_patterns = [2, 5]  # Patterns 2 and 5 for m=5
                    elif m == 7:
                        valid_patterns = [2]  # Only pattern 2 for m=7
                    else:  # m=6
                        valid_patterns = [3, 4, 5]  # Patterns 3, 4, 5 for m=6
                    
                    # Filter out patterns already used in this grid
                    available_patterns = [p for p in valid_patterns if p not in used_patterns]
                    
                    # If no available patterns (all used already), use a random valid pattern
                    # This should only happen if we have more frames than available patterns
                    if not available_patterns:
                        # Just pick one of the valid patterns - in a grid with many frames, some patterns may repeat
                        current_pattern_type = random.choice(valid_patterns)
                    else:
                        # Use an unused pattern
                        current_pattern_type = random.choice(available_patterns)
                        used_patterns.add(current_pattern_type)
                    
                    # Draw the frame with perfect pattern
                    frame_info = self._create_horizontal_frame(
                        grid, 
                        frame_start_row, 
                        frame_height, 
                        cols, 
                        frame_color, 
                        pattern_color, 
                        background_color, 
                        current_pattern_type
                    )
                    frames_info.append(frame_info)
                    
                    # Add 1-2 disruptions to the pattern
                    disruptions = self._add_disruptions(
                        grid, 
                        frame_start_row+1, 
                        frame_start_row+frame_height-2, 
                        2, 
                        cols-3, 
                        pattern_color, 
                        background_color
                    )
                    
                    # Add disruption info to frame info
                    frame_info['disruptions'] = disruptions
                    
            else:  # Vertical arrangement
                cols_needed = m * n + 1
                rows = cols  # Use the original 'cols' value for the new height
                grid = np.full((rows, cols_needed), background_color, dtype=int)
                
                # Store frame info for later
                frames_info = []
                
                # Track used patterns and colors to ensure variety and uniqueness within the grid
                used_patterns = set()
                used_pattern_colors = set()
                
                # Create frames
                for i in range(n):
                    # First frame starts at position (1, 1)
                    frame_start_col = 1 + i * m  # First frame starts at col 1, then each m columns
                    frame_width = m - 1  # Frame width is m-1
                    
                    # Choose a unique pattern color (different from background, frame, and previously used patterns)
                    available_colors = [c for c in range(1, 10) 
                                     if c != background_color and c != frame_color and c != 5 
                                     and c not in used_pattern_colors]
                    
                    # If somehow we've used all colors, pick one we haven't used in this grid yet
                    if not available_colors:
                        # This is an edge case - try to find any usable color
                        available_colors = [c for c in range(1, 10) 
                                         if c != background_color and c != frame_color and c != 5]
                        if not available_colors:
                            # Absolute fallback
                            pattern_color = (background_color + 1) % 9 + 1
                            if pattern_color == 5:
                                pattern_color = 6
                        else:
                            pattern_color = random.choice(available_colors)
                    else:
                        pattern_color = random.choice(available_colors)
                    
                    # Track this pattern color as used in this grid
                    used_pattern_colors.add(pattern_color)
                    
                    # Determine which patterns can be used for this frame based on m
                    valid_patterns = []
                    if m == 4:
                        valid_patterns = [1]  # Only pattern 1 is valid for m=4
                    elif m == 5:
                        valid_patterns = [2, 5]  # Patterns 2 and 5 for m=5
                    elif m == 7:
                        valid_patterns = [2]  # Only pattern 2 for m=7
                    else:  # m=6
                        valid_patterns = [3, 4, 5]  # Patterns 3, 4, 5 for m=6
                    
                    # Filter out patterns already used in this grid
                    available_patterns = [p for p in valid_patterns if p not in used_patterns]
                    
                    # If no available patterns (all used already), use a random valid pattern
                    # This should only happen if we have more frames than available patterns
                    if not available_patterns:
                        # Just pick one of the valid patterns - in a grid with many frames, some patterns may repeat
                        current_pattern_type = random.choice(valid_patterns)
                    else:
                        # Use an unused pattern
                        current_pattern_type = random.choice(available_patterns)
                        used_patterns.add(current_pattern_type)
                    
                    # Draw the frame with perfect pattern
                    frame_info = self._create_vertical_frame(
                        grid, 
                        frame_start_col, 
                        frame_width, 
                        rows, 
                        frame_color, 
                        pattern_color, 
                        background_color, 
                        current_pattern_type
                    )
                    frames_info.append(frame_info)
                    
                    # Add 1-2 disruptions to the pattern
                    disruptions = self._add_disruptions(
                        grid, 
                        2, 
                        rows-3, 
                        frame_start_col+1, 
                        frame_start_col+frame_width-2, 
                        pattern_color, 
                        background_color
                    )
                    
                    # Add disruption info to frame info
                    frame_info['disruptions'] = disruptions
            
            pattern_info = {
                'horizontal': horizontal,
                'background_color': background_color,
                'frame_color': frame_color,
                'frames': frames_info
            }
            
            return grid, pattern_info
            
        except Exception as e:
            # Fallback in case of any error - create a simple grid with no disruptions
            print(f"Error in create_input_with_disruptions: {e}")
            # Create minimal grid and pattern info
            rows = max(5, m * n + 1)
            cols = max(5, taskvars['cols'])
            grid = np.full((rows, cols), background_color, dtype=int)
            pattern_info = {
                'horizontal': horizontal,
                'background_color': background_color,
                'frame_color': frame_color,
                'frames': []
            }
            return grid, pattern_info
    
    def create_output(self, input_grid, pattern_info):
        """Create correct output grid by fixing the disruptions"""
        try:
            output_grid = input_grid.copy()
            
            # Fix each frame by restoring the original pattern
            for frame in pattern_info['frames']:
                # Restore the perfect pattern by fixing disruptions
                if 'disruptions' in frame:
                    for r, c, original_value in frame['disruptions']:
                        if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                            output_grid[r, c] = original_value
                
            return output_grid
            
        except Exception as e:
            # In case of error, return the input grid unchanged
            print(f"Error in create_output: {e}")
            return input_grid.copy()
    
    def _create_horizontal_frame(self, grid, start_row, height, grid_width, frame_color, pattern_color, background_color, pattern_type):
        """Create a horizontal frame with the specified pattern"""
        try:
            # Check if the frame fits within the grid
            if start_row + height > grid.shape[0] or grid_width > grid.shape[1]:
                # Return empty frame info
                return {
                    'start_row': start_row,
                    'height': height,
                    'pattern_type': pattern_type,
                    'pattern_color': pattern_color,
                    'frame_color': frame_color,
                    'background_color': background_color,
                    'interior': {
                        'top': start_row + 1,
                        'bottom': min(start_row + height - 2, grid.shape[0] - 1),
                        'left': 2,
                        'right': min(grid_width - 3, grid.shape[1] - 1)
                    },
                    'disruptions': []
                }
                
            # Draw the frame borders - cover full width from 1 to cols-2
            # Top border
            grid[start_row, 1:grid_width-1] = frame_color
            # Bottom border
            grid[start_row + height - 1, 1:grid_width-1] = frame_color
            # Left border
            grid[start_row+1:start_row+height-1, 1] = frame_color
            # Right border
            grid[start_row+1:start_row+height-1, grid_width-2] = frame_color
            
            # Calculate interior region of the frame
            interior_top = start_row + 1
            interior_bottom = start_row + height - 2
            interior_left = 2
            interior_right = grid_width - 3
            
            # Store frame info for later fixing
            frame_info = {
                'start_row': start_row,
                'height': height,
                'pattern_type': pattern_type,
                'pattern_color': pattern_color,
                'frame_color': frame_color,
                'background_color': background_color,
                'interior': {
                    'top': interior_top,
                    'bottom': interior_bottom,
                    'left': interior_left,
                    'right': interior_right
                }
            }
            
            # Fill the pattern based on pattern type
            if pattern_type == 1:  # Alternating pattern for m=4 (height=3)
                # For m=4, frame height is 3
                # We need one row of alternating pattern in the middle
                middle_row = start_row + 1
                
                for c in range(2, grid_width-2):
                    if (c - 2) % 2 == 0:
                        grid[middle_row, c] = pattern_color
                    else:
                        grid[middle_row, c] = background_color
            
            elif pattern_type == 2:  # 2x2 checker pattern for m=5,7
                # For m=5, frame height is 4
                # For m=7, frame height is 6
                # We need to fill with 2x2 blocks followed by vertical strips
                
                # For each row in the interior
                for r in range(interior_top, interior_bottom + 1):
                    row_in_pattern = r - interior_top
                    
                    # For each column in the interior
                    for c in range(interior_left, interior_right + 1):
                        col_in_pattern = c - interior_left
                        
                        # Pattern repeats every 3 columns: 2 for the 2x2 block, 1 for the separator
                        block_pos = col_in_pattern % 3
                        
                        if block_pos < 2:  # Inside a 2x2 block
                            # For the first two rows of each block, fill with pattern color
                            if row_in_pattern % 2 == 0 or row_in_pattern % 2 == 1:
                                grid[r, c] = pattern_color
                        else:  # Vertical separator
                            grid[r, c] = background_color
            
            elif pattern_type == 3:  # Checkerboard pattern
                # Fill the interior with a checkerboard pattern
                for r in range(interior_top, interior_bottom + 1):
                    for c in range(interior_left, interior_right + 1):
                        if ((r - interior_top) + (c - interior_left)) % 2 == 0:
                            grid[r, c] = pattern_color
                        else:
                            grid[r, c] = background_color
            
            elif pattern_type == 4:  # Horizontal stripes
                # Fill the interior with horizontal stripes
                for r in range(interior_top, interior_bottom + 1):
                    row_in_pattern = r - interior_top
                    for c in range(interior_left, interior_right + 1):
                        if row_in_pattern % 2 == 0:
                            grid[r, c] = pattern_color
                        else:
                            grid[r, c] = background_color
            
            elif pattern_type == 5:  # U-shaped pattern
                # Create a repeating pattern with units of length 4
                pattern_width = 4  # Width of one pattern repetition
                interior_height = interior_bottom - interior_top + 1
                
                # For each row in the interior
                for r in range(interior_top, interior_bottom + 1):
                    row_in_pattern = r - interior_top
                    
                    for c in range(interior_left, interior_right + 1):
                        position = (c - interior_left) % pattern_width
                        
                        # For m=5 (interior_height=2)
                        if interior_height == 2:
                            if row_in_pattern == 0:  # First interior row
                                # Pattern: t, bg, t, t (repeated)
                                if position in [0, 2, 3]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            elif row_in_pattern == 1:  # Second interior row
                                # Pattern: t, t, t, bg (repeated)
                                if position in [0, 1, 2]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                        # For m=6 (interior_height=3)
                        elif interior_height == 3:
                            # For m=6, the correct pattern for each row
                            if row_in_pattern == 0:  # First interior row
                                # Pattern: t, bg, t, t (repeated)
                                if position in [0, 2, 3]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            elif row_in_pattern == 1:  # Second interior row - CORRECTED PATTERN
                                # Pattern: t, bg, t, bg (alternating)
                                if position in [0, 2]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            elif row_in_pattern == 2:  # Third interior row
                                # Pattern: t, t, t, bg (repeated)
                                if position in [0, 1, 2]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                        # For m>6
                        else:
                            if row_in_pattern == 0:  # First row
                                # Pattern: t, bg, t, t
                                if position in [0, 2, 3]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            elif row_in_pattern == interior_height - 1:  # Last interior row
                                # Pattern: t, t, t, bg
                                if position in [0, 1, 2]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            else:  # Middle rows repeat the patterns alternately
                                if row_in_pattern % 2 == 1:  # Odd rows
                                    if position in [0, 1, 2]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
                                else:  # Even rows
                                    if position in [0, 2, 3]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
            
            return frame_info
            
        except Exception as e:
            # Return minimal frame info in case of error
            print(f"Error in _create_horizontal_frame: {e}")
            return {
                'start_row': start_row,
                'height': height,
                'pattern_type': pattern_type,
                'pattern_color': pattern_color,
                'frame_color': frame_color,
                'background_color': background_color,
                'interior': {
                    'top': start_row + 1,
                    'bottom': min(start_row + height - 2, grid.shape[0] - 1),
                    'left': 2,
                    'right': min(grid_width - 3, grid.shape[1] - 1)
                },
                'disruptions': []
            }
    
    def _create_vertical_frame(self, grid, start_col, width, grid_height, frame_color, pattern_color, background_color, pattern_type):
        """Create a vertical frame with the specified pattern"""
        try:
            # Check if the frame fits within the grid
            if start_col + width > grid.shape[1] or grid_height > grid.shape[0]:
                # Return empty frame info
                return {
                    'start_col': start_col,
                    'width': width,
                    'pattern_type': pattern_type,
                    'pattern_color': pattern_color,
                    'frame_color': frame_color,
                    'background_color': background_color,
                    'interior': {
                        'top': 2,
                        'bottom': min(grid_height - 3, grid.shape[0] - 1),
                        'left': start_col + 1,
                        'right': min(start_col + width - 2, grid.shape[1] - 1)
                    },
                    'disruptions': []
                }
                
            # Draw the frame borders - cover full height from 1 to rows-2
            # Left border
            grid[1:grid_height-1, start_col] = frame_color
            # Right border
            grid[1:grid_height-1, start_col + width - 1] = frame_color
            # Top border
            grid[1, start_col+1:start_col+width-1] = frame_color
            # Bottom border
            grid[grid_height-2, start_col+1:start_col+width-1] = frame_color
            
            # Calculate interior region of the frame
            interior_top = 2
            interior_bottom = grid_height - 3
            interior_left = start_col + 1
            interior_right = start_col + width - 2
            
            # Handle edge case where there's not enough interior space
            if interior_right < interior_left or interior_bottom < interior_top:
                # Return minimal frame info in case of insufficient space
                return {
                    'start_col': start_col,
                    'width': width,
                    'pattern_type': pattern_type,
                    'pattern_color': pattern_color,
                    'frame_color': frame_color,
                    'background_color': background_color,
                    'interior': {
                        'top': interior_top,
                        'bottom': interior_bottom,
                        'left': interior_left,
                        'right': interior_right
                    },
                    'disruptions': []
                }
            
            # Store frame info for later fixing
            frame_info = {
                'start_col': start_col,
                'width': width,
                'pattern_type': pattern_type,
                'pattern_color': pattern_color,
                'frame_color': frame_color,
                'background_color': background_color,
                'interior': {
                    'top': interior_top,
                    'bottom': interior_bottom,
                    'left': interior_left,
                    'right': interior_right
                }
            }
            
            # Fill the pattern based on pattern type, but rotated 90 degrees
            if pattern_type == 1:  # Alternating pattern for m=4 (width=3)
                # For m=4, frame width is 3
                # We need one column of alternating pattern in the middle
                middle_col = start_col + 1
                
                for r in range(2, grid_height-2):
                    if (r - 2) % 2 == 0:
                        grid[r, middle_col] = pattern_color
                    else:
                        grid[r, middle_col] = background_color
            
            elif pattern_type == 2:  # 2x2 checker pattern for m=5,7
                # Rotated pattern 2: 2x2 blocks with horizontal separators
                
                # For each column in the interior
                for c in range(interior_left, interior_right + 1):
                    col_in_pattern = c - interior_left
                    
                    # For each row in the interior
                    for r in range(interior_top, interior_bottom + 1):
                        row_in_pattern = r - interior_top
                        
                        # Pattern repeats every 3 rows: 2 for the 2x2 block, 1 for the separator
                        block_pos = row_in_pattern % 3
                        
                        if block_pos < 2:  # Inside a 2x2 block
                            # For the first two columns of each block, fill with pattern color
                            if col_in_pattern % 2 == 0 or col_in_pattern % 2 == 1:
                                grid[r, c] = pattern_color
                        else:  # Horizontal separator
                            grid[r, c] = background_color
            
            elif pattern_type == 3:  # Checkerboard pattern
                # Fill the interior with a checkerboard pattern
                for c in range(interior_left, interior_right + 1):
                    for r in range(interior_top, interior_bottom + 1):
                        if ((r - interior_top) + (c - interior_left)) % 2 == 0:
                            grid[r, c] = pattern_color
                        else:
                            grid[r, c] = background_color
            
            elif pattern_type == 4:  # Vertical stripes
                # Fill the interior with vertical stripes
                for c in range(interior_left, interior_right + 1):
                    col_in_pattern = c - interior_left
                    for r in range(interior_top, interior_bottom + 1):
                        if col_in_pattern % 2 == 0:
                            grid[r, c] = pattern_color
                        else:
                            grid[r, c] = background_color
            
            elif pattern_type == 5:  # U-shaped pattern rotated
                # Pattern 5 - Updated for vertical orientation
                pattern_height = 4  # Height of one pattern repetition
                interior_width = interior_right - interior_left + 1
                
                # For each column in the interior
                for c in range(interior_left, interior_right + 1):
                    col_in_pattern = c - interior_left
                    
                    for r in range(interior_top, interior_bottom + 1):
                        position = (r - interior_top) % pattern_height
                        
                        # For m=5 (interior_width=2)
                        if interior_width == 2:
                            if col_in_pattern == 0:  # First interior column
                                # Pattern: t, bg, t, t (repeated vertically)
                                if position in [0, 2, 3]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            elif col_in_pattern == 1:  # Second interior column
                                # Pattern: t, t, t, bg (repeated vertically)
                                if position in [0, 1, 2]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                        # For m=6 (interior_width=3)
                        elif interior_width == 3:
                            # For m=6, the correct pattern for each column
                            if col_in_pattern == 0:  # First interior column
                                # Pattern: t, bg, t, t (repeated vertically)
                                if position in [0, 2, 3]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            elif col_in_pattern == 1:  # Second interior column - CORRECTED PATTERN
                                # Pattern: t, bg, t, bg (alternating vertically)
                                if position in [0, 2]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            elif col_in_pattern == 2:  # Third interior column
                                # Pattern: t, t, t, bg (repeated vertically)
                                if position in [0, 1, 2]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                        # For m>6
                        else:
                            if col_in_pattern == 0:  # First column
                                # Pattern: t, bg, t, t
                                if position in [0, 2, 3]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            elif col_in_pattern == interior_width - 1:  # Last interior column
                                # Pattern: t, t, t, bg
                                if position in [0, 1, 2]:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                            else:  # Middle columns repeat the patterns alternately
                                if col_in_pattern % 2 == 1:  # Odd columns
                                    if position in [0, 1, 2]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
                                else:  # Even columns
                                    if position in [0, 2, 3]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
                            
            return frame_info
            
        except Exception as e:
            # Return minimal frame info in case of error
            print(f"Error in _create_vertical_frame: {e}")
            return {
                'start_col': start_col,
                'width': width,
                'pattern_type': pattern_type,
                'pattern_color': pattern_color,
                'frame_color': frame_color,
                'background_color': background_color,
                'interior': {
                    'top': 2,
                    'bottom': min(grid_height - 3, grid.shape[0] - 1), 
                    'left': start_col + 1,
                    'right': min(start_col + width - 2, grid.shape[1] - 1)
                },
                'disruptions': []
            }
    
    def _add_disruptions(self, grid, row_start, row_end, col_start, col_end, pattern_color, background_color):
        """Add 1-2 disruptions to the pattern and return their locations and original values"""
        try:
            num_disruptions = random.randint(1, 2)
            disruptions = []
            
            # Validate input ranges
            if row_end < row_start or col_end < col_start:
                return disruptions  # Return empty list if ranges are invalid
            
            # Ensure the ranges are within grid bounds
            row_start = max(0, row_start)
            row_end = min(grid.shape[0] - 1, row_end)
            col_start = max(0, col_start)
            col_end = min(grid.shape[1] - 1, col_end)
            
            # If after validation, ranges are invalid, return empty disruptions
            if row_end < row_start or col_end < col_start:
                return disruptions
            
            for _ in range(num_disruptions):
                r = random.randint(row_start, row_end)
                c = random.randint(col_start, col_end)
                
                # Save original value before disruption
                original_value = grid[r, c]
                
                # Swap color (pattern to background or vice versa)
                if grid[r, c] == pattern_color:
                    grid[r, c] = background_color
                else:
                    grid[r, c] = pattern_color
                
                # Save location and original value for later fixing
                disruptions.append((r, c, original_value))
            
            return disruptions
            
        except Exception as e:
            # In case of any error, return empty disruptions
            print(f"Error in _add_disruptions: {e}")
            return []
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Required implementation of create_input, calls create_input_with_disruptions"""
        try:
            m = gridvars['m']
            n = gridvars['n']
            horizontal = gridvars.get('horizontal', True)
            pattern_type = gridvars.get('pattern_type', None)
            frame_color = gridvars.get('frame_color', taskvars['frame'])
            
            # Select pattern type based on m if not specified
            if pattern_type is None:
                if m == 4:
                    pattern_type = 1
                elif m == 5:
                    pattern_type = random.choice([2, 5])  # For m=5, use pattern 2 or 5
                elif m == 7:
                    pattern_type = 2
                else:
                    pattern_type = random.choice([3, 4, 5])
            
            # Create input grid and get pattern info (not used here)
            input_grid, _ = self.create_input_with_disruptions(
                taskvars, m, n, pattern_type, horizontal, frame_color
            )
            
            return input_grid
            
        except Exception as e:
            # In case of error, create a minimal valid grid
            print(f"Error in create_input: {e}")
            rows = max(5, gridvars.get('m', 5) * gridvars.get('n', 1) + 1)
            cols = max(5, taskvars.get('cols', 15))
            grid = np.full((rows, cols), 1, dtype=int)
            return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Required implementation of transform_input, uses advanced pattern detection and fixing"""
        try:
            output_grid = grid.copy()
            
            # Detect background, frame, and pattern colors
            background_color = self._detect_background_color(grid)
            frame_colors = self._detect_frame_colors(grid, background_color)
            
            # Determine if this is a horizontal or vertical frame layout
            horizontal_frames = self._detect_orientation(grid, frame_colors, background_color)
            
            # Process and fix patterns in all frames
            if horizontal_frames:
                self._fix_horizontal_frames(output_grid, frame_colors, background_color)
            else:
                self._fix_vertical_frames(output_grid, frame_colors, background_color)
                
            return output_grid
            
        except Exception as e:
            # In case of error, return input grid unchanged
            print(f"Error in transform_input: {e}")
            return grid.copy()
    
    def _detect_background_color(self, grid):
        """Detect the background color of the grid (most common corner color)"""
        try:
            corner_colors = [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]]
            return max(set(corner_colors), key=corner_colors.count)
        except Exception as e:
            # Fallback to a default color
            print(f"Error in _detect_background_color: {e}")
            return 1
    
    def _detect_frame_colors(self, grid, background_color):
        """Detect frame colors in the grid"""
        frame_colors = set()
        
        try:
            # Check rows and columns 1 from the edge - these should contain frame colors
            for r in range(1, grid.shape[0]-1):
                for c in range(1, grid.shape[1]-1):
                    if grid[r, c] != background_color:
                        # A non-background color that forms a line is likely a frame
                        # Check if this cell is part of a horizontal or vertical line
                        horizontal_line = (c > 1 and c < grid.shape[1]-2 and 
                                        grid[r, c-1] == grid[r, c] and grid[r, c+1] == grid[r, c])
                        vertical_line = (r > 1 and r < grid.shape[0]-2 and 
                                        grid[r-1, c] == grid[r, c] and grid[r+1, c] == grid[r, c])
                        
                        if horizontal_line or vertical_line:
                            frame_colors.add(grid[r, c])
        except Exception as e:
            print(f"Error in _detect_frame_colors: {e}")
        
        # If no frame colors detected, use a default different from background
        if not frame_colors:
            frame_colors.add(2 if background_color != 2 else 3)
            
        return frame_colors
    
    def _detect_orientation(self, grid, frame_colors, background_color):
        """Detect if the frame layout is horizontal or vertical"""
        try:
            # Count horizontal and vertical frame lines
            horizontal_lines = 0
            vertical_lines = 0
            
            # Look for horizontal frame rows
            for r in range(1, grid.shape[0]-1):
                row_colors = grid[r, 1:grid.shape[1]-1]
                # Count horizontal segments where frame colors appear consistently
                frame_segment_count = 0
                for c in range(1, len(row_colors)-1):
                    if c < len(row_colors) and row_colors[c] in frame_colors:
                        if c > 0 and c+1 < len(row_colors) and row_colors[c-1] in frame_colors and row_colors[c+1] in frame_colors:
                            frame_segment_count += 1
                if frame_segment_count > 0:
                    horizontal_lines += 1
            
            # Look for vertical frame columns
            for c in range(1, grid.shape[1]-1):
                if c < grid.shape[1]:
                    col_colors = grid[1:grid.shape[0]-1, c]
                    # Count vertical segments where frame colors appear consistently
                    frame_segment_count = 0
                    for r in range(1, len(col_colors)-1):
                        if r < len(col_colors) and col_colors[r] in frame_colors:
                            if r > 0 and r+1 < len(col_colors) and col_colors[r-1] in frame_colors and col_colors[r+1] in frame_colors:
                                frame_segment_count += 1
                    if frame_segment_count > 0:
                        vertical_lines += 1
        except Exception as e:
            print(f"Error in _detect_orientation: {e}")
            # Default to horizontal in case of error
            return True
        
        # If we have more horizontal lines than vertical, it's likely horizontal frames
        return horizontal_lines >= vertical_lines
    
    def _fix_horizontal_frames(self, grid, frame_colors, background_color):
        """Find and fix patterns in horizontal frames"""
        try:
            # Find all horizontal frame rows
            frame_rows = []
            for r in range(1, grid.shape[0]-1):
                row = grid[r, :]
                # Count consecutive frame color cells
                frame_color_count = sum(1 for c in range(1, grid.shape[1]-1) if c < grid.shape[1] and row[c] in frame_colors)
                if frame_color_count > grid.shape[1] // 3:  # If a significant portion is frame color
                    frame_rows.append(r)
            
            # Group consecutive rows to find frames
            i = 0
            while i < len(frame_rows):
                top_row = frame_rows[i]
                
                # Find all consecutive frame rows (the complete frame, including top and bottom)
                j = i + 1
                while j < len(frame_rows) and frame_rows[j] == frame_rows[j-1] + 1:
                    j += 1
                
                # If we have found a frame (at least 3 rows high)
                if j - i >= 3:
                    bottom_row = frame_rows[j-1]
                    
                    # Estimate frame borders
                    left_col = 1
                    right_col = grid.shape[1] - 2
                    
                    # Find pattern color (most common non-background, non-frame color)
                    interior = grid[top_row+1:bottom_row, left_col+1:right_col]
                    pattern_color = self._find_pattern_color(interior, frame_colors, background_color)
                    
                    if pattern_color is not None:
                        # Determine frame height to choose pattern type
                        frame_height = bottom_row - top_row + 1
                        
                        # Fix the pattern based on frame height
                        self._apply_horizontal_pattern_fix(
                            grid, top_row, bottom_row, left_col, right_col,
                            pattern_color, frame_colors, background_color, frame_height
                        )
                
                # Move to next potential frame
                i = j
        except Exception as e:
            print(f"Error in _fix_horizontal_frames: {e}")
    
    def _fix_vertical_frames(self, grid, frame_colors, background_color):
        """Find and fix patterns in vertical frames"""
        try:
            # Find all vertical frame columns
            frame_cols = []
            for c in range(1, grid.shape[1]-1):
                if c < grid.shape[1]:
                    col = grid[:, c]
                    # Count consecutive frame color cells
                    frame_color_count = sum(1 for r in range(1, grid.shape[0]-1) if r < grid.shape[0] and col[r] in frame_colors)
                    if frame_color_count > grid.shape[0] // 3:  # If a significant portion is frame color
                        frame_cols.append(c)
            
            # Group consecutive columns to find frames
            i = 0
            while i < len(frame_cols):
                left_col = frame_cols[i]
                
                # Find all consecutive frame columns (the complete frame, including left and right)
                j = i + 1
                while j < len(frame_cols) and frame_cols[j] == frame_cols[j-1] + 1:
                    j += 1
                
                # If we have found a frame (at least 3 columns wide)
                if j - i >= 3:
                    right_col = frame_cols[j-1]
                    
                    # Estimate frame borders
                    top_row = 1
                    bottom_row = grid.shape[0] - 2
                    
                    # Find pattern color (most common non-background, non-frame color)
                    interior = grid[top_row+1:bottom_row, left_col+1:right_col]
                    pattern_color = self._find_pattern_color(interior, frame_colors, background_color)
                    
                    if pattern_color is not None:
                        # Determine frame width to choose pattern type
                        frame_width = right_col - left_col + 1
                        
                        # Fix the pattern based on frame width
                        self._apply_vertical_pattern_fix(
                            grid, top_row, bottom_row, left_col, right_col,
                            pattern_color, frame_colors, background_color, frame_width
                        )
                
                # Move to next potential frame
                i = j
        except Exception as e:
            print(f"Error in _fix_vertical_frames: {e}")
    
    def _find_pattern_color(self, interior, frame_colors, background_color):
        """Find the most common non-background, non-frame color in the interior"""
        try:
            color_counts = {}
            
            for r in range(interior.shape[0]):
                for c in range(interior.shape[1]):
                    color = interior[r, c]
                    if color != background_color and color not in frame_colors:
                        color_counts[color] = color_counts.get(color, 0) + 1
            
            if not color_counts:
                return None
                
            return max(color_counts, key=color_counts.get)
        except Exception as e:
            print(f"Error in _find_pattern_color: {e}")
            # Return a default pattern color
            return 3 if background_color != 3 and 3 not in frame_colors else 4
    
    def _apply_horizontal_pattern_fix(self, grid, top_row, bottom_row, left_col, right_col, 
                                   pattern_color, frame_colors, background_color, frame_height):
        """Apply the correct pattern fix based on frame height for horizontal frames"""
        try:
            # Calculate interior region of the frame
            interior_top = top_row + 1
            interior_bottom = bottom_row - 1
            interior_left = left_col + 1
            interior_right = right_col - 1
            interior_height = interior_bottom - interior_top + 1
            
            # Choose pattern type based on frame height
            if frame_height == 3:  # Pattern 1 for m=4
                # Fix alternating pattern in middle row
                middle_row = top_row + 1
                for c in range(interior_left, interior_right + 1):
                    if c < grid.shape[1]:
                        if (c - interior_left) % 2 == 0:
                            grid[middle_row, c] = pattern_color
                        else:
                            grid[middle_row, c] = background_color
                        
            elif frame_height == 4:  # Pattern 2 or 5 for m=5
                # Determine if it's pattern 2 or 5 based on the existing cells
                interior = grid[interior_top:interior_bottom+1, interior_left:interior_right+1]
                # Check if pattern 5 (U-shape) is a better match than pattern 2 (2x2 checker)
                if self._is_better_match_for_u_shape(interior, pattern_color, background_color):
                    # Fix U-shaped pattern for m=5 with the correct pattern
                    for r in range(interior_top, interior_bottom + 1):
                        row_in_pattern = r - interior_top
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                position = (c - interior_left) % 4
                                
                                if row_in_pattern == 0:  # First interior row
                                    # Pattern: t, bg, t, t (repeated)
                                    if position in [0, 2, 3]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
                                elif row_in_pattern == 1:  # Second interior row
                                    # Pattern: t, t, t, bg (repeated)
                                    if position in [0, 1, 2]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
                else:
                    # Fix 2x2 checker pattern with vertical separators
                    for r in range(interior_top, interior_bottom + 1):
                        row_in_pattern = r - interior_top
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                col_in_pattern = c - interior_left
                                block_pos = col_in_pattern % 3
                                
                                if block_pos < 2:  # Inside 2x2 block
                                    grid[r, c] = pattern_color
                                else:  # Vertical separator
                                    grid[r, c] = background_color
                            
            elif frame_height == 6:  # Pattern 2 for m=7
                # Same as m=5 but with more rows
                for r in range(interior_top, interior_bottom + 1):
                    row_in_pattern = r - interior_top
                    for c in range(interior_left, interior_right + 1):
                        if c < grid.shape[1]:
                            col_in_pattern = c - interior_left
                            block_pos = col_in_pattern % 3
                            
                            if block_pos < 2:  # Inside 2x2 block
                                grid[r, c] = pattern_color
                            else:  # Vertical separator
                                grid[r, c] = background_color
                            
            elif frame_height == 5:  # Patterns 3, 4, or 5 for m=6
                # Determine which pattern fits best
                interior = grid[interior_top:interior_bottom+1, interior_left:interior_right+1]
                pattern_type = self._identify_best_pattern(interior, pattern_color, background_color)
                
                # Apply the appropriate pattern
                if pattern_type == 3:  # Checkerboard
                    for r in range(interior_top, interior_bottom + 1):
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                if ((r - interior_top) + (c - interior_left)) % 2 == 0:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                                
                elif pattern_type == 4:  # Horizontal stripes
                    for r in range(interior_top, interior_bottom + 1):
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                if (r - interior_top) % 2 == 0:
                                    grid[r, c] = pattern_color
                                else:
                                    grid[r, c] = background_color
                                
                elif pattern_type == 5:  # U-shaped pattern
                    # Pattern 5 with the correct pattern for m=6
                    for r in range(interior_top, interior_bottom + 1):
                        row_in_pattern = r - interior_top
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                position = (c - interior_left) % 4
                                
                                # For m=6, the correct pattern for each row
                                if row_in_pattern == 0:  # First interior row
                                    # Pattern: t, bg, t, t (repeated)
                                    if position in [0, 2, 3]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
                                elif row_in_pattern == 1:  # Second interior row - CORRECTED PATTERN
                                    # Pattern: t, bg, t, bg (alternating)
                                    if position in [0, 2]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
                                elif row_in_pattern == 2:  # Third interior row
                                    # Pattern: t, t, t, bg (repeated)
                                    if position in [0, 1, 2]:
                                        grid[r, c] = pattern_color
                                    else:
                                        grid[r, c] = background_color
        except Exception as e:
            print(f"Error in _apply_horizontal_pattern_fix: {e}")
    
    def _apply_vertical_pattern_fix(self, grid, top_row, bottom_row, left_col, right_col, 
                                  pattern_color, frame_colors, background_color, frame_width):
        """Apply the correct pattern fix based on frame width for vertical frames"""
        try:
            # Calculate interior region of the frame
            interior_top = top_row + 1
            interior_bottom = bottom_row - 1
            interior_left = left_col + 1
            interior_right = right_col - 1
            interior_width = interior_right - interior_left + 1
            
            # Choose pattern type based on frame width
            if frame_width == 3:  # Pattern 1 for m=4
                # Fix alternating pattern in middle column
                middle_col = left_col + 1
                for r in range(interior_top, interior_bottom + 1):
                    if r < grid.shape[0] and middle_col < grid.shape[1]:
                        if (r - interior_top) % 2 == 0:
                            grid[r, middle_col] = pattern_color
                        else:
                            grid[r, middle_col] = background_color
                        
            elif frame_width == 4:  # Pattern 2 or 5 for m=5
                # Get interior region for pattern detection
                if interior_top < interior_bottom and interior_left < interior_right:
                    interior = grid[interior_top:interior_bottom+1, interior_left:interior_right+1].T
                    # Check if pattern 5 (U-shape) is a better match than pattern 2 (2x2 checker)
                    if self._is_better_match_for_u_shape(interior, pattern_color, background_color):
                        # Fix U-shaped pattern for m=5 (rotated) with the correct pattern
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                col_in_pattern = c - interior_left
                                for r in range(interior_top, interior_bottom + 1):
                                    if r < grid.shape[0]:
                                        position = (r - interior_top) % 4
                                        
                                        if col_in_pattern == 0:  # First column
                                            # Pattern: t, bg, t, t (rotated)
                                            if position in [0, 2, 3]:
                                                grid[r, c] = pattern_color
                                            else:
                                                grid[r, c] = background_color
                                        elif col_in_pattern == 1:  # Second column
                                            # Pattern: t, t, t, bg (rotated)
                                            if position in [0, 1, 2]:
                                                grid[r, c] = pattern_color
                                            else:
                                                grid[r, c] = background_color
                    else:
                        # Fix 2x2 checker pattern with horizontal separators
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                col_in_pattern = c - interior_left
                                for r in range(interior_top, interior_bottom + 1):
                                    if r < grid.shape[0]:
                                        row_in_pattern = r - interior_top
                                        block_pos = row_in_pattern % 3
                                        
                                        if block_pos < 2:  # Inside 2x2 block
                                            grid[r, c] = pattern_color
                                        else:  # Horizontal separator
                                            grid[r, c] = background_color
                        
            elif frame_width == 6:  # Pattern 2 for m=7
                # Same as m=5 but with more columns
                for c in range(interior_left, interior_right + 1):
                    if c < grid.shape[1]:
                        col_in_pattern = c - interior_left
                        for r in range(interior_top, interior_bottom + 1):
                            if r < grid.shape[0]:
                                row_in_pattern = r - interior_top
                                block_pos = row_in_pattern % 3
                                
                                if block_pos < 2:  # Inside 2x2 block
                                    grid[r, c] = pattern_color
                                else:  # Horizontal separator
                                    grid[r, c] = background_color
                        
            elif frame_width == 5:  # Patterns 3, 4, or 5 for m=6
                # Get interior region for pattern detection
                if interior_top < interior_bottom and interior_left < interior_right:
                    interior = grid[interior_top:interior_bottom+1, interior_left:interior_right+1].T
                    pattern_type = self._identify_best_pattern(interior, pattern_color, background_color)
                    
                    # Apply the appropriate pattern
                    if pattern_type == 3:  # Checkerboard (same both ways)
                        for r in range(interior_top, interior_bottom + 1):
                            if r < grid.shape[0]:
                                for c in range(interior_left, interior_right + 1):
                                    if c < grid.shape[1]:
                                        if ((r - interior_top) + (c - interior_left)) % 2 == 0:
                                            grid[r, c] = pattern_color
                                        else:
                                            grid[r, c] = background_color
                                
                    elif pattern_type == 4:  # Vertical stripes
                        for r in range(interior_top, interior_bottom + 1):
                            if r < grid.shape[0]:
                                for c in range(interior_left, interior_right + 1):
                                    if c < grid.shape[1]:
                                        if (c - interior_left) % 2 == 0:
                                            grid[r, c] = pattern_color
                                        else:
                                            grid[r, c] = background_color
                                
                    elif pattern_type == 5:  # U-shaped pattern rotated
                        # Pattern 5 with the correct pattern for m=6 vertical frames
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                col_in_pattern = c - interior_left
                                for r in range(interior_top, interior_bottom + 1):
                                    if r < grid.shape[0]:
                                        position = (r - interior_top) % 4
                                        
                                        # For m=6, the correct pattern for each column
                                        if col_in_pattern == 0:  # First interior column
                                            # Pattern: t, bg, t, t (repeated vertically)
                                            if position in [0, 2, 3]:
                                                grid[r, c] = pattern_color
                                            else:
                                                grid[r, c] = background_color
                                        elif col_in_pattern == 1:  # Second interior column - CORRECTED PATTERN
                                            # Pattern: t, bg, t, bg (alternating vertically)
                                            if position in [0, 2]:
                                                grid[r, c] = pattern_color
                                            else:
                                                grid[r, c] = background_color
                                        elif col_in_pattern == 2:  # Third interior column
                                            # Pattern: t, t, t, bg (repeated vertically)
                                            if position in [0, 1, 2]:
                                                grid[r, c] = pattern_color
                                            else:
                                                grid[r, c] = background_color
        except Exception as e:
            print(f"Error in _apply_vertical_pattern_fix: {e}")
    
    def _is_better_match_for_u_shape(self, interior, pattern_color, background_color):
        """Check if pattern 5 (U-shape) is a better match than pattern 2 (2x2 checker)"""
        try:
            # Calculate match scores for U-shape and 2x2 checker patterns
            u_shape_score = self._calculate_u_shape_match(interior, pattern_color, background_color)
            checker_score = self._calculate_2x2_checker_match(interior, pattern_color, background_color)
            
            # Return True if U-shape is a better match
            return u_shape_score > checker_score
        except Exception as e:
            print(f"Error in _is_better_match_for_u_shape: {e}")
            return False
    
    def _calculate_u_shape_match(self, interior, color1, color2):
        """Calculate how well the U-shape pattern matches the interior"""
        try:
            if interior.shape[0] < 2:  # Need at least 2 rows
                return 0
                
            height, width = interior.shape
            matches = 0
            total = 0
            
            # Check rows to identify the pattern
            for r in range(min(3, height)):
                for c in range(width):
                    total += 1
                    position = c % 4  # 4-position repeat pattern
                    
                    if r == 0:  # First row
                        # Pattern: t, bg, t, t
                        expected = color1 if position in [0, 2, 3] else color2
                    elif r == 1 and height == 3:  # For m=6, second row
                        # Pattern: t, bg, t, bg (alternating)
                        expected = color1 if position in [0, 2] else color2
                    elif r == 1 or (r == 2 and height == 3):  # Second row for m=5 or third row for m=6
                        # Pattern: t, t, t, bg
                        expected = color1 if position in [0, 1, 2] else color2
                    else:
                        # Default
                        expected = color1
                    
                    if interior[r, c] == expected:
                        matches += 1
            
            return matches / total if total > 0 else 0
        except Exception as e:
            print(f"Error in _calculate_u_shape_match: {e}")
            return 0
    
    def _calculate_2x2_checker_match(self, interior, color1, color2):
        """Calculate how well the 2x2 checker pattern matches the interior"""
        try:
            height, width = interior.shape
            matches = 0
            total = 0
            
            for r in range(height):
                for c in range(width):
                    total += 1
                    block_pos = c % 3  # 3-column repeat pattern
                    
                    if block_pos < 2:  # Inside 2x2 block
                        expected = color1
                    else:  # Separator
                        expected = color2
                    
                    if interior[r, c] == expected:
                        matches += 1
            
            return matches / total if total > 0 else 0
        except Exception as e:
            print(f"Error in _calculate_2x2_checker_match: {e}")
            return 0
    
    def _identify_best_pattern(self, interior, pattern_color, background_color):
        """Identify which pattern type (3, 4, or 5) best matches the interior"""
        try:
            # Calculate match scores for each pattern
            checkerboard_score = self._calculate_pattern_match(
                interior, pattern_color, background_color, self._is_checkerboard_pattern
            )
            
            stripes_score = self._calculate_pattern_match(
                interior, pattern_color, background_color, self._is_stripe_pattern
            )
            
            u_shape_score = self._calculate_pattern_match(
                interior, pattern_color, background_color, self._is_u_shape_pattern
            )
            
            # Return the pattern with highest match score
            scores = {3: checkerboard_score, 4: stripes_score, 5: u_shape_score}
            return max(scores, key=scores.get)
        except Exception as e:
            print(f"Error in _identify_best_pattern: {e}")
            return 5  # Default to U-shape pattern
    
    def _calculate_pattern_match(self, interior, color1, color2, pattern_func):
        """Calculate how well a pattern matches the interior"""
        try:
            height, width = interior.shape
            matches = 0
            total = 0
            
            for r in range(height):
                for c in range(width):
                    total += 1
                    expected = pattern_func(r, c, height, color1, color2)
                    if interior[r, c] == expected:
                        matches += 1
            
            return matches / total if total > 0 else 0
        except Exception as e:
            print(f"Error in _calculate_pattern_match: {e}")
            return 0
    
    def _is_checkerboard_pattern(self, r, c, height, color1, color2):
        """Return expected color for checkerboard pattern at (r,c)"""
        return color1 if (r + c) % 2 == 0 else color2
    
    def _is_stripe_pattern(self, r, c, height, color1, color2):
        """Return expected color for horizontal stripe pattern at (r,c)"""
        return color1 if r % 2 == 0 else color2
    
    def _is_u_shape_pattern(self, r, c, height, color1, color2):
        """Return expected color for U-shaped pattern at (r,c)"""
        try:
            position = c % 4  # 4-position pattern
            
            # Special case for m=6 (interior_height=3)
            if height == 3:
                if r == 0:  # First row
                    return color1 if position in [0, 2, 3] else color2
                elif r == 1:  # Second row - special pattern for m=6
                    return color1 if position in [0, 2] else color2
                elif r == 2:  # Third row
                    return color1 if position in [0, 1, 2] else color2
            # For m=5 or other cases
            else:
                if r == 0:  # First row
                    return color1 if position in [0, 2, 3] else color2
                elif r == height - 1:  # Last row
                    return color1 if position in [0, 1, 2] else color2
                else:  # Middle rows alternate
                    if r % 2 == 1:  # Odd rows
                        return color1 if position in [0, 1, 2] else color2
                    else:  # Even rows
                        return color1 if position in [0, 2, 3] else color2
        except Exception as e:
            print(f"Error in _is_u_shape_pattern: {e}")
            return color1  # Default

