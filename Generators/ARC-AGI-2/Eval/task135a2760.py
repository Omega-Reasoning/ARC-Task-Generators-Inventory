from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple
from Framework.input_library import create_object, retry, Contiguity
from Framework.transformation_library import find_connected_objects

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
        taskvars = {
            'cols': random.randint(15, 25),
            'frame': random.choice([color for color in range(1, 10) if color != 5])
        }
        
        num_train_examples = random.randint(3, 6)
        train_examples = []
        
        used_m_values = set()
        used_patterns = set()
        required_m = [4, 5, 7]
        
        for i in range(num_train_examples):
            if i < len(required_m):
                m = required_m[i]
            else:
                available_m = [m for m in [4, 5, 6, 7] if m not in used_m_values]
                if not available_m:
                    available_m = [4, 5, 6, 7]
                m = random.choice(available_m)
            
            used_m_values.add(m)
            n = random.randint(1, 3)
            
            if m == 4:
                pattern_type = 1
            elif m == 5:
                pattern_type = 5 if i == 1 else random.choice([2, 5])
            elif m == 7:
                pattern_type = 2
            else:
                available_patterns = [p for p in [3, 4, 5] if p not in used_patterns]
                if not available_patterns:
                    available_patterns = [3, 4, 5]
                pattern_type = random.choice(available_patterns)
                used_patterns.add(pattern_type)
            
            input_grid, pattern_info = self.create_input_with_disruptions(taskvars, m, n, pattern_type, True)
            output_grid = self.create_output(input_grid, pattern_info)
            
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Test case: vertical arrangement, different frame color
        m = random.choice([4, 5, 6, 7])
        n = random.randint(1, 3)
        test_frame_color = random.choice([c for c in range(1, 10) if c != taskvars['frame'] and c != 5])
        
        if m == 4:
            pattern_type = 1
        elif m == 5:
            pattern_type = random.choice([2, 5])
        elif m == 7:
            pattern_type = 2
        else:
            pattern_type = random.choice([3, 4, 5])
            
        test_input, test_pattern_info = self.create_input_with_disruptions(
            taskvars, m, n, pattern_type, False, frame_color=test_frame_color
        )
        test_output = self.create_output(test_input, test_pattern_info)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input_with_disruptions(self, taskvars, m, n, pattern_type, horizontal=True, frame_color=None):
        """Create input grid with pattern and intentional disruptions, and return disruption info"""
        try:
            if frame_color is None:
                frame_color = taskvars['frame']
            
            cols = taskvars['cols']
            background_color = random.choice([c for c in range(1, 10) if c != frame_color and c != 5])
            
            if horizontal:
                rows = m * n + 1
                grid = np.full((rows, cols), background_color, dtype=int)
                frames_info = []
                used_patterns = set()
                used_pattern_colors = set()
                
                for i in range(n):
                    frame_start_row = 1 + i * m
                    frame_height = m - 1
                    
                    available_colors = [c for c in range(1, 10) 
                                     if c != background_color and c != frame_color and c != 5 
                                     and c not in used_pattern_colors]
                    if not available_colors:
                        available_colors = [c for c in range(1, 10) 
                                         if c != background_color and c != frame_color and c != 5]
                        if not available_colors:
                            pattern_color = (background_color + 1) % 9 + 1
                            if pattern_color == 5:
                                pattern_color = 6
                        else:
                            pattern_color = random.choice(available_colors)
                    else:
                        pattern_color = random.choice(available_colors)
                    
                    used_pattern_colors.add(pattern_color)
                    
                    valid_patterns = []
                    if m == 4:
                        valid_patterns = [1]
                    elif m == 5:
                        valid_patterns = [2, 5]
                    elif m == 7:
                        valid_patterns = [2]
                    else:
                        valid_patterns = [3, 4, 5]
                    
                    available_patterns = [p for p in valid_patterns if p not in used_patterns]
                    if not available_patterns:
                        current_pattern_type = random.choice(valid_patterns)
                    else:
                        current_pattern_type = random.choice(available_patterns)
                        used_patterns.add(current_pattern_type)
                    
                    frame_info = self._create_horizontal_frame(
                        grid, frame_start_row, frame_height, cols,
                        frame_color, pattern_color, background_color, current_pattern_type
                    )
                    frames_info.append(frame_info)
                    
                    disruptions = self._add_disruptions(
                        grid, frame_start_row+1, frame_start_row+frame_height-2,
                        2, cols-3, pattern_color, background_color
                    )
                    frame_info['disruptions'] = disruptions
                    
            else:  # Vertical arrangement
                cols_needed = m * n + 1
                rows = cols
                grid = np.full((rows, cols_needed), background_color, dtype=int)
                frames_info = []
                used_patterns = set()
                used_pattern_colors = set()
                
                for i in range(n):
                    frame_start_col = 1 + i * m
                    frame_width = m - 1
                    
                    available_colors = [c for c in range(1, 10) 
                                     if c != background_color and c != frame_color and c != 5 
                                     and c not in used_pattern_colors]
                    if not available_colors:
                        available_colors = [c for c in range(1, 10) 
                                         if c != background_color and c != frame_color and c != 5]
                        if not available_colors:
                            pattern_color = (background_color + 1) % 9 + 1
                            if pattern_color == 5:
                                pattern_color = 6
                        else:
                            pattern_color = random.choice(available_colors)
                    else:
                        pattern_color = random.choice(available_colors)
                    
                    used_pattern_colors.add(pattern_color)
                    
                    valid_patterns = []
                    if m == 4:
                        valid_patterns = [1]
                    elif m == 5:
                        valid_patterns = [2, 5]
                    elif m == 7:
                        valid_patterns = [2]
                    else:
                        valid_patterns = [3, 4, 5]
                    
                    available_patterns = [p for p in valid_patterns if p not in used_patterns]
                    if not available_patterns:
                        current_pattern_type = random.choice(valid_patterns)
                    else:
                        current_pattern_type = random.choice(available_patterns)
                        used_patterns.add(current_pattern_type)
                    
                    frame_info = self._create_vertical_frame(
                        grid, frame_start_col, frame_width, rows,
                        frame_color, pattern_color, background_color, current_pattern_type
                    )
                    frames_info.append(frame_info)
                    
                    disruptions = self._add_disruptions(
                        grid, 2, rows-3, frame_start_col+1, frame_start_col+frame_width-2,
                        pattern_color, background_color
                    )
                    frame_info['disruptions'] = disruptions
            
            pattern_info = {
                'horizontal': horizontal,
                'background_color': background_color,
                'frame_color': frame_color,
                'frames': frames_info
            }
            
            return grid, pattern_info
            
        except Exception as e:
            print(f"Error in create_input_with_disruptions: {e}")
            rows = max(5, m * n + 1)
            cols_val = max(5, taskvars['cols'])
            background_color = 1
            grid = np.full((rows, cols_val), background_color, dtype=int)
            pattern_info = {
                'horizontal': horizontal,
                'background_color': background_color,
                'frame_color': frame_color if frame_color else taskvars['frame'],
                'frames': []
            }
            return grid, pattern_info
    
    def create_output(self, input_grid, pattern_info):
        """Create correct output grid by fixing the disruptions"""
        try:
            output_grid = input_grid.copy()
            for frame in pattern_info['frames']:
                if 'disruptions' in frame:
                    for r, c, original_value in frame['disruptions']:
                        if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                            output_grid[r, c] = original_value
            return output_grid
        except Exception as e:
            print(f"Error in create_output: {e}")
            return input_grid.copy()
    
    def _create_horizontal_frame(self, grid, start_row, height, grid_width, frame_color, pattern_color, background_color, pattern_type):
        """Create a horizontal frame with the specified pattern"""
        try:
            if start_row + height > grid.shape[0] or grid_width > grid.shape[1]:
                return {
                    'start_row': start_row, 'height': height, 'pattern_type': pattern_type,
                    'pattern_color': pattern_color, 'frame_color': frame_color,
                    'background_color': background_color,
                    'interior': {'top': start_row+1, 'bottom': min(start_row+height-2, grid.shape[0]-1),
                                 'left': 2, 'right': min(grid_width-3, grid.shape[1]-1)},
                    'disruptions': []
                }
            
            grid[start_row, 1:grid_width-1] = frame_color
            grid[start_row + height - 1, 1:grid_width-1] = frame_color
            grid[start_row+1:start_row+height-1, 1] = frame_color
            grid[start_row+1:start_row+height-1, grid_width-2] = frame_color
            
            interior_top = start_row + 1
            interior_bottom = start_row + height - 2
            interior_left = 2
            interior_right = grid_width - 3
            
            frame_info = {
                'start_row': start_row, 'height': height, 'pattern_type': pattern_type,
                'pattern_color': pattern_color, 'frame_color': frame_color,
                'background_color': background_color,
                'interior': {'top': interior_top, 'bottom': interior_bottom,
                             'left': interior_left, 'right': interior_right}
            }
            
            if pattern_type == 1:
                middle_row = start_row + 1
                for c in range(2, grid_width-2):
                    grid[middle_row, c] = pattern_color if (c - 2) % 2 == 0 else background_color
            
            elif pattern_type == 2:
                for r in range(interior_top, interior_bottom + 1):
                    for c in range(interior_left, interior_right + 1):
                        grid[r, c] = pattern_color if (c - interior_left) % 3 < 2 else background_color
            
            elif pattern_type == 3:
                for r in range(interior_top, interior_bottom + 1):
                    for c in range(interior_left, interior_right + 1):
                        grid[r, c] = pattern_color if ((r - interior_top) + (c - interior_left)) % 2 == 0 else background_color
            
            elif pattern_type == 4:
                for r in range(interior_top, interior_bottom + 1):
                    for c in range(interior_left, interior_right + 1):
                        grid[r, c] = pattern_color if (r - interior_top) % 2 == 0 else background_color
            
            elif pattern_type == 5:
                pattern_width = 4
                interior_height = interior_bottom - interior_top + 1
                for r in range(interior_top, interior_bottom + 1):
                    row_in_pattern = r - interior_top
                    for c in range(interior_left, interior_right + 1):
                        position = (c - interior_left) % pattern_width
                        if interior_height == 2:
                            if row_in_pattern == 0:
                                grid[r, c] = pattern_color if position in [0, 2, 3] else background_color
                            else:
                                grid[r, c] = pattern_color if position in [0, 1, 2] else background_color
                        elif interior_height == 3:
                            if row_in_pattern == 0:
                                grid[r, c] = pattern_color if position in [0, 2, 3] else background_color
                            elif row_in_pattern == 1:
                                grid[r, c] = pattern_color if position in [0, 2] else background_color
                            else:
                                grid[r, c] = pattern_color if position in [0, 1, 2] else background_color
                        else:
                            if row_in_pattern == 0:
                                grid[r, c] = pattern_color if position in [0, 2, 3] else background_color
                            elif row_in_pattern == interior_height - 1:
                                grid[r, c] = pattern_color if position in [0, 1, 2] else background_color
                            elif row_in_pattern % 2 == 1:
                                grid[r, c] = pattern_color if position in [0, 1, 2] else background_color
                            else:
                                grid[r, c] = pattern_color if position in [0, 2, 3] else background_color
            
            return frame_info
            
        except Exception as e:
            print(f"Error in _create_horizontal_frame: {e}")
            return {
                'start_row': start_row, 'height': height, 'pattern_type': pattern_type,
                'pattern_color': pattern_color, 'frame_color': frame_color,
                'background_color': background_color,
                'interior': {'top': start_row+1, 'bottom': min(start_row+height-2, grid.shape[0]-1),
                             'left': 2, 'right': min(grid_width-3, grid.shape[1]-1)},
                'disruptions': []
            }
    
    def _create_vertical_frame(self, grid, start_col, width, grid_height, frame_color, pattern_color, background_color, pattern_type):
        """Create a vertical frame with the specified pattern"""
        try:
            if start_col + width > grid.shape[1] or grid_height > grid.shape[0]:
                return {
                    'start_col': start_col, 'width': width, 'pattern_type': pattern_type,
                    'pattern_color': pattern_color, 'frame_color': frame_color,
                    'background_color': background_color,
                    'interior': {'top': 2, 'bottom': min(grid_height-3, grid.shape[0]-1),
                                 'left': start_col+1, 'right': min(start_col+width-2, grid.shape[1]-1)},
                    'disruptions': []
                }
            
            grid[1:grid_height-1, start_col] = frame_color
            grid[1:grid_height-1, start_col + width - 1] = frame_color
            grid[1, start_col+1:start_col+width-1] = frame_color
            grid[grid_height-2, start_col+1:start_col+width-1] = frame_color
            
            interior_top = 2
            interior_bottom = grid_height - 3
            interior_left = start_col + 1
            interior_right = start_col + width - 2
            
            if interior_right < interior_left or interior_bottom < interior_top:
                return {
                    'start_col': start_col, 'width': width, 'pattern_type': pattern_type,
                    'pattern_color': pattern_color, 'frame_color': frame_color,
                    'background_color': background_color,
                    'interior': {'top': interior_top, 'bottom': interior_bottom,
                                 'left': interior_left, 'right': interior_right},
                    'disruptions': []
                }
            
            frame_info = {
                'start_col': start_col, 'width': width, 'pattern_type': pattern_type,
                'pattern_color': pattern_color, 'frame_color': frame_color,
                'background_color': background_color,
                'interior': {'top': interior_top, 'bottom': interior_bottom,
                             'left': interior_left, 'right': interior_right}
            }
            
            if pattern_type == 1:
                middle_col = start_col + 1
                for r in range(2, grid_height-2):
                    grid[r, middle_col] = pattern_color if (r - 2) % 2 == 0 else background_color
            
            elif pattern_type == 2:
                for c in range(interior_left, interior_right + 1):
                    for r in range(interior_top, interior_bottom + 1):
                        grid[r, c] = pattern_color if (r - interior_top) % 3 < 2 else background_color
            
            elif pattern_type == 3:
                for c in range(interior_left, interior_right + 1):
                    for r in range(interior_top, interior_bottom + 1):
                        grid[r, c] = pattern_color if ((r - interior_top) + (c - interior_left)) % 2 == 0 else background_color
            
            elif pattern_type == 4:
                for c in range(interior_left, interior_right + 1):
                    for r in range(interior_top, interior_bottom + 1):
                        grid[r, c] = pattern_color if (c - interior_left) % 2 == 0 else background_color
            
            elif pattern_type == 5:
                pattern_height = 4
                interior_width = interior_right - interior_left + 1
                for c in range(interior_left, interior_right + 1):
                    col_in_pattern = c - interior_left
                    for r in range(interior_top, interior_bottom + 1):
                        position = (r - interior_top) % pattern_height
                        if interior_width == 2:
                            if col_in_pattern == 0:
                                grid[r, c] = pattern_color if position in [0, 2, 3] else background_color
                            else:
                                grid[r, c] = pattern_color if position in [0, 1, 2] else background_color
                        elif interior_width == 3:
                            if col_in_pattern == 0:
                                grid[r, c] = pattern_color if position in [0, 2, 3] else background_color
                            elif col_in_pattern == 1:
                                grid[r, c] = pattern_color if position in [0, 2] else background_color
                            else:
                                grid[r, c] = pattern_color if position in [0, 1, 2] else background_color
                        else:
                            if col_in_pattern == 0:
                                grid[r, c] = pattern_color if position in [0, 2, 3] else background_color
                            elif col_in_pattern == interior_width - 1:
                                grid[r, c] = pattern_color if position in [0, 1, 2] else background_color
                            elif col_in_pattern % 2 == 1:
                                grid[r, c] = pattern_color if position in [0, 1, 2] else background_color
                            else:
                                grid[r, c] = pattern_color if position in [0, 2, 3] else background_color
                            
            return frame_info
            
        except Exception as e:
            print(f"Error in _create_vertical_frame: {e}")
            return {
                'start_col': start_col, 'width': width, 'pattern_type': pattern_type,
                'pattern_color': pattern_color, 'frame_color': frame_color,
                'background_color': background_color,
                'interior': {'top': 2, 'bottom': min(grid_height-3, grid.shape[0]-1),
                             'left': start_col+1, 'right': min(start_col+width-2, grid.shape[1]-1)},
                'disruptions': []
            }
    
    def _add_disruptions(self, grid, row_start, row_end, col_start, col_end, pattern_color, background_color):
        """Add 1-2 disruptions to the pattern and return their locations and original values"""
        try:
            num_disruptions = random.randint(1, 2)
            disruptions = []
            
            if row_end < row_start or col_end < col_start:
                return disruptions
            
            row_start = max(0, row_start)
            row_end = min(grid.shape[0] - 1, row_end)
            col_start = max(0, col_start)
            col_end = min(grid.shape[1] - 1, col_end)
            
            if row_end < row_start or col_end < col_start:
                return disruptions
            
            for _ in range(num_disruptions):
                r = random.randint(row_start, row_end)
                c = random.randint(col_start, col_end)
                original_value = grid[r, c]
                grid[r, c] = background_color if grid[r, c] == pattern_color else pattern_color
                disruptions.append((r, c, original_value))
            
            return disruptions
            
        except Exception as e:
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
            
            if pattern_type is None:
                if m == 4:
                    pattern_type = 1
                elif m == 5:
                    pattern_type = random.choice([2, 5])
                elif m == 7:
                    pattern_type = 2
                else:
                    pattern_type = random.choice([3, 4, 5])
            
            input_grid, _ = self.create_input_with_disruptions(
                taskvars, m, n, pattern_type, horizontal, frame_color
            )
            return input_grid
            
        except Exception as e:
            print(f"Error in create_input: {e}")
            rows = max(5, gridvars.get('m', 5) * gridvars.get('n', 1) + 1)
            cols = max(5, taskvars.get('cols', 15))
            return np.full((rows, cols), 1, dtype=int)
    
    def transform_input(self, grid, taskvars):
        import numpy as np

        rows, cols = grid.shape
        output = grid.copy()

        # --- background: most common corner color ---
        corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
        background = max(set(corners), key=corners.count)

        # --- find connected components for every non-background color ---
        visited = np.zeros((rows, cols), dtype=bool)
        components_by_color = {}

        for r in range(rows):
            for c in range(cols):
                color = int(grid[r, c])
                if color == background or visited[r, c]:
                    continue

                stack = [(r, c)]
                visited[r, c] = True
                cells = []

                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))

                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < rows
                            and 0 <= ny < cols
                            and not visited[nx, ny]
                            and int(grid[nx, ny]) == color
                        ):
                            visited[nx, ny] = True
                            stack.append((nx, ny))

                components_by_color.setdefault(color, []).append(cells)

        # --- detect frame color by finding hollow rectangular components ---
        frame_color = None
        frame_boxes = []
        best_count = -1

        for color, comps in components_by_color.items():
            valid_boxes = []

            for cells in comps:
                rs = [p[0] for p in cells]
                cs = [p[1] for p in cells]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)

                h = r1 - r0 + 1
                w = c1 - c0 + 1
                if h < 3 or w < 3:
                    continue

                cellset = set(cells)
                perimeter_size = 2 * w + 2 * h - 4

                # frame component must be exactly the 1-cell-thick perimeter
                if len(cellset) != perimeter_size:
                    continue

                ok = True
                for cc in range(c0, c1 + 1):
                    if int(grid[r0, cc]) != color or int(grid[r1, cc]) != color:
                        ok = False
                        break
                if not ok:
                    continue

                for rr in range(r0, r1 + 1):
                    if int(grid[rr, c0]) != color or int(grid[rr, c1]) != color:
                        ok = False
                        break
                if not ok:
                    continue

                valid_boxes.append((r0, r1, c0, c1))

            if len(valid_boxes) > best_count:
                best_count = len(valid_boxes)
                frame_color = color
                frame_boxes = valid_boxes

        if frame_color is None:
            return output

        # --- helper: dominant non-background / non-frame color inside one frame ---
        def find_pattern_color(interior):
            counts = {}
            for v in interior.flat:
                v = int(v)
                if v != background and v != frame_color:
                    counts[v] = counts.get(v, 0) + 1
            if not counts:
                return None
            return max(counts, key=counts.get)

        # --- pattern scoring / reconstruction ---
        def score_pattern(interior, pattern_color, horizontal, pattern_type):
            h, w = interior.shape
            repaired = np.empty_like(interior)
            score = 0

            for r in range(h):
                for c in range(w):
                    expected_is_pattern = False

                    if horizontal:
                        if pattern_type == 1:
                            expected_is_pattern = (c % 2 == 0)

                        elif pattern_type == 2:
                            expected_is_pattern = (c % 3 < 2)

                        elif pattern_type == 3:
                            expected_is_pattern = ((r + c) % 2 == 0)

                        elif pattern_type == 4:
                            expected_is_pattern = (r % 2 == 0)

                        elif pattern_type == 5:
                            pos = c % 4
                            if h == 2:
                                if r == 0:
                                    expected_is_pattern = pos in [0, 2, 3]
                                else:
                                    expected_is_pattern = pos in [0, 1, 2]
                            elif h == 3:
                                if r == 0:
                                    expected_is_pattern = pos in [0, 2, 3]
                                elif r == 1:
                                    expected_is_pattern = pos in [0, 2]
                                else:
                                    expected_is_pattern = pos in [0, 1, 2]
                            else:
                                if r == 0:
                                    expected_is_pattern = pos in [0, 2, 3]
                                elif r == h - 1:
                                    expected_is_pattern = pos in [0, 1, 2]
                                elif r % 2 == 1:
                                    expected_is_pattern = pos in [0, 1, 2]
                                else:
                                    expected_is_pattern = pos in [0, 2, 3]

                    else:
                        if pattern_type == 1:
                            expected_is_pattern = (r % 2 == 0)

                        elif pattern_type == 2:
                            expected_is_pattern = (r % 3 < 2)

                        elif pattern_type == 3:
                            expected_is_pattern = ((r + c) % 2 == 0)

                        elif pattern_type == 4:
                            expected_is_pattern = (c % 2 == 0)

                        elif pattern_type == 5:
                            pos = r % 4
                            if w == 2:
                                if c == 0:
                                    expected_is_pattern = pos in [0, 2, 3]
                                else:
                                    expected_is_pattern = pos in [0, 1, 2]
                            elif w == 3:
                                if c == 0:
                                    expected_is_pattern = pos in [0, 2, 3]
                                elif c == 1:
                                    expected_is_pattern = pos in [0, 2]
                                else:
                                    expected_is_pattern = pos in [0, 1, 2]
                            else:
                                if c == 0:
                                    expected_is_pattern = pos in [0, 2, 3]
                                elif c == w - 1:
                                    expected_is_pattern = pos in [0, 1, 2]
                                elif c % 2 == 1:
                                    expected_is_pattern = pos in [0, 1, 2]
                                else:
                                    expected_is_pattern = pos in [0, 2, 3]

                    expected = pattern_color if expected_is_pattern else background
                    repaired[r, c] = expected
                    if int(interior[r, c]) == expected:
                        score += 1

            return score, repaired

        def get_candidate_patterns(horizontal, h, w):
            if horizontal:
                if h == 1:
                    return [1]
                if h == 2:
                    return [2, 5]
                if h == 3:
                    return [3, 4, 5]
                if h == 4:
                    return [2]
            else:
                if w == 1:
                    return [1]
                if w == 2:
                    return [2, 5]
                if w == 3:
                    return [3, 4, 5]
                if w == 4:
                    return [2]
            return [1, 2, 3, 4, 5]

        # --- repair every detected frame independently ---
        for r0, r1, c0, c1 in frame_boxes:
            interior = grid[r0 + 1:r1, c0 + 1:c1].copy()
            if interior.size == 0:
                continue

            horizontal = (c1 - c0) >= (r1 - r0)
            pattern_color = find_pattern_color(interior)
            if pattern_color is None:
                continue

            candidates = get_candidate_patterns(horizontal, interior.shape[0], interior.shape[1])

            best_score = -1
            best_repaired = interior

            for pattern_type in candidates:
                score, repaired = score_pattern(interior, pattern_color, horizontal, pattern_type)
                if score > best_score:
                    best_score = score
                    best_repaired = repaired

            output[r0 + 1:r1, c0 + 1:c1] = best_repaired

        return output
    
    def _detect_background_color(self, grid):
        try:
            corner_colors = [grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1]]
            return max(set(corner_colors), key=corner_colors.count)
        except Exception as e:
            print(f"Error in _detect_background_color: {e}")
            return 1
    
    def _detect_frame_colors(self, grid, background_color):
        frame_colors = set()
        try:
            for r in range(1, grid.shape[0]-1):
                for c in range(1, grid.shape[1]-1):
                    if grid[r, c] != background_color:
                        horizontal_line = (c > 1 and c < grid.shape[1]-2 and 
                                        grid[r, c-1] == grid[r, c] and grid[r, c+1] == grid[r, c])
                        vertical_line = (r > 1 and r < grid.shape[0]-2 and 
                                        grid[r-1, c] == grid[r, c] and grid[r+1, c] == grid[r, c])
                        if horizontal_line or vertical_line:
                            frame_colors.add(grid[r, c])
        except Exception as e:
            print(f"Error in _detect_frame_colors: {e}")
        if not frame_colors:
            frame_colors.add(2 if background_color != 2 else 3)
        return frame_colors
    
    def _detect_orientation(self, grid, frame_colors, background_color):
        try:
            horizontal_lines = 0
            vertical_lines = 0
            for r in range(1, grid.shape[0]-1):
                row_colors = grid[r, 1:grid.shape[1]-1]
                frame_segment_count = sum(
                    1 for c in range(1, len(row_colors)-1)
                    if row_colors[c] in frame_colors
                    and row_colors[c-1] in frame_colors
                    and row_colors[c+1] in frame_colors
                )
                if frame_segment_count > 0:
                    horizontal_lines += 1
            for c in range(1, grid.shape[1]-1):
                col_colors = grid[1:grid.shape[0]-1, c]
                frame_segment_count = sum(
                    1 for r in range(1, len(col_colors)-1)
                    if col_colors[r] in frame_colors
                    and col_colors[r-1] in frame_colors
                    and col_colors[r+1] in frame_colors
                )
                if frame_segment_count > 0:
                    vertical_lines += 1
        except Exception as e:
            print(f"Error in _detect_orientation: {e}")
            return True
        return horizontal_lines >= vertical_lines
    
    def _fix_horizontal_frames(self, grid, frame_colors, background_color):
        try:
            frame_rows = [r for r in range(1, grid.shape[0]-1)
                         if sum(1 for c in range(1, grid.shape[1]-1) if grid[r, c] in frame_colors) > grid.shape[1] // 3]
            i = 0
            while i < len(frame_rows):
                top_row = frame_rows[i]
                j = i + 1
                while j < len(frame_rows) and frame_rows[j] == frame_rows[j-1] + 1:
                    j += 1
                if j - i >= 3:
                    bottom_row = frame_rows[j-1]
                    interior = grid[top_row+1:bottom_row, 2:grid.shape[1]-2]
                    pattern_color = self._find_pattern_color(interior, frame_colors, background_color)
                    if pattern_color is not None:
                        self._apply_horizontal_pattern_fix(
                            grid, top_row, bottom_row, 1, grid.shape[1]-2,
                            pattern_color, frame_colors, background_color, bottom_row - top_row + 1
                        )
                i = j
        except Exception as e:
            print(f"Error in _fix_horizontal_frames: {e}")
    
    def _fix_vertical_frames(self, grid, frame_colors, background_color):
        try:
            frame_cols = [c for c in range(1, grid.shape[1]-1)
                         if sum(1 for r in range(1, grid.shape[0]-1) if grid[r, c] in frame_colors) > grid.shape[0] // 3]
            i = 0
            while i < len(frame_cols):
                left_col = frame_cols[i]
                j = i + 1
                while j < len(frame_cols) and frame_cols[j] == frame_cols[j-1] + 1:
                    j += 1
                if j - i >= 3:
                    right_col = frame_cols[j-1]
                    interior = grid[2:grid.shape[0]-2, left_col+1:right_col]
                    pattern_color = self._find_pattern_color(interior, frame_colors, background_color)
                    if pattern_color is not None:
                        self._apply_vertical_pattern_fix(
                            grid, 1, grid.shape[0]-2, left_col, right_col,
                            pattern_color, frame_colors, background_color, right_col - left_col + 1
                        )
                i = j
        except Exception as e:
            print(f"Error in _fix_vertical_frames: {e}")
    
    def _find_pattern_color(self, interior, frame_colors, background_color):
        try:
            color_counts = {}
            for r in range(interior.shape[0]):
                for c in range(interior.shape[1]):
                    color = interior[r, c]
                    if color != background_color and color not in frame_colors:
                        color_counts[color] = color_counts.get(color, 0) + 1
            return max(color_counts, key=color_counts.get) if color_counts else None
        except Exception as e:
            print(f"Error in _find_pattern_color: {e}")
            return 3 if background_color != 3 and 3 not in frame_colors else 4
    
    def _apply_horizontal_pattern_fix(self, grid, top_row, bottom_row, left_col, right_col,
                                   pattern_color, frame_colors, background_color, frame_height):
        try:
            interior_top = top_row + 1
            interior_bottom = bottom_row - 1
            interior_left = left_col + 1
            interior_right = right_col - 1

            if frame_height == 3:
                middle_row = top_row + 1
                for c in range(interior_left, interior_right + 1):
                    if c < grid.shape[1]:
                        grid[middle_row, c] = pattern_color if (c - interior_left) % 2 == 0 else background_color
            elif frame_height == 4:
                interior = grid[interior_top:interior_bottom+1, interior_left:interior_right+1]
                if self._is_better_match_for_u_shape(interior, pattern_color, background_color):
                    for r in range(interior_top, interior_bottom + 1):
                        row_in_pattern = r - interior_top
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                position = (c - interior_left) % 4
                                grid[r, c] = pattern_color if position in ([0,2,3] if row_in_pattern==0 else [0,1,2]) else background_color
                else:
                    for r in range(interior_top, interior_bottom + 1):
                        for c in range(interior_left, interior_right + 1):
                            if c < grid.shape[1]:
                                grid[r, c] = pattern_color if (c - interior_left) % 3 < 2 else background_color
            elif frame_height == 6:
                for r in range(interior_top, interior_bottom + 1):
                    for c in range(interior_left, interior_right + 1):
                        if c < grid.shape[1]:
                            grid[r, c] = pattern_color if (c - interior_left) % 3 < 2 else background_color
            elif frame_height == 5:
                interior = grid[interior_top:interior_bottom+1, interior_left:interior_right+1]
                pattern_type = self._identify_best_pattern(interior, pattern_color, background_color)
                for r in range(interior_top, interior_bottom + 1):
                    for c in range(interior_left, interior_right + 1):
                        if c < grid.shape[1]:
                            if pattern_type == 3:
                                grid[r, c] = pattern_color if ((r-interior_top)+(c-interior_left))%2==0 else background_color
                            elif pattern_type == 4:
                                grid[r, c] = pattern_color if (r-interior_top)%2==0 else background_color
                            else:
                                position = (c - interior_left) % 4
                                row_in_pattern = r - interior_top
                                if row_in_pattern == 0:
                                    grid[r, c] = pattern_color if position in [0,2,3] else background_color
                                elif row_in_pattern == 1:
                                    grid[r, c] = pattern_color if position in [0,2] else background_color
                                else:
                                    grid[r, c] = pattern_color if position in [0,1,2] else background_color
        except Exception as e:
            print(f"Error in _apply_horizontal_pattern_fix: {e}")
    
    def _apply_vertical_pattern_fix(self, grid, top_row, bottom_row, left_col, right_col,
                                  pattern_color, frame_colors, background_color, frame_width):
        try:
            interior_top = top_row + 1
            interior_bottom = bottom_row - 1
            interior_left = left_col + 1
            interior_right = right_col - 1

            if frame_width == 3:
                middle_col = left_col + 1
                for r in range(interior_top, interior_bottom + 1):
                    if r < grid.shape[0]:
                        grid[r, middle_col] = pattern_color if (r-interior_top)%2==0 else background_color
            elif frame_width == 4:
                if interior_top < interior_bottom and interior_left < interior_right:
                    interior = grid[interior_top:interior_bottom+1, interior_left:interior_right+1].T
                    if self._is_better_match_for_u_shape(interior, pattern_color, background_color):
                        for c in range(interior_left, interior_right + 1):
                            col_in_pattern = c - interior_left
                            for r in range(interior_top, interior_bottom + 1):
                                if r < grid.shape[0]:
                                    position = (r - interior_top) % 4
                                    grid[r, c] = pattern_color if position in ([0,2,3] if col_in_pattern==0 else [0,1,2]) else background_color
                    else:
                        for c in range(interior_left, interior_right + 1):
                            for r in range(interior_top, interior_bottom + 1):
                                if r < grid.shape[0]:
                                    grid[r, c] = pattern_color if (r-interior_top)%3 < 2 else background_color
            elif frame_width == 6:
                for c in range(interior_left, interior_right + 1):
                    for r in range(interior_top, interior_bottom + 1):
                        if r < grid.shape[0]:
                            grid[r, c] = pattern_color if (r-interior_top)%3 < 2 else background_color
            elif frame_width == 5:
                if interior_top < interior_bottom and interior_left < interior_right:
                    interior = grid[interior_top:interior_bottom+1, interior_left:interior_right+1].T
                    pattern_type = self._identify_best_pattern(interior, pattern_color, background_color)
                    for r in range(interior_top, interior_bottom + 1):
                        for c in range(interior_left, interior_right + 1):
                            if r < grid.shape[0] and c < grid.shape[1]:
                                if pattern_type == 3:
                                    grid[r, c] = pattern_color if ((r-interior_top)+(c-interior_left))%2==0 else background_color
                                elif pattern_type == 4:
                                    grid[r, c] = pattern_color if (c-interior_left)%2==0 else background_color
                                else:
                                    col_in_pattern = c - interior_left
                                    position = (r - interior_top) % 4
                                    if col_in_pattern == 0:
                                        grid[r, c] = pattern_color if position in [0,2,3] else background_color
                                    elif col_in_pattern == 1:
                                        grid[r, c] = pattern_color if position in [0,2] else background_color
                                    else:
                                        grid[r, c] = pattern_color if position in [0,1,2] else background_color
        except Exception as e:
            print(f"Error in _apply_vertical_pattern_fix: {e}")
    
    def _is_better_match_for_u_shape(self, interior, pattern_color, background_color):
        try:
            return self._calculate_u_shape_match(interior, pattern_color, background_color) > \
                   self._calculate_2x2_checker_match(interior, pattern_color, background_color)
        except Exception as e:
            print(f"Error in _is_better_match_for_u_shape: {e}")
            return False
    
    def _calculate_u_shape_match(self, interior, color1, color2):
        try:
            if interior.shape[0] < 2:
                return 0
            height, width = interior.shape
            matches = 0
            total = 0
            for r in range(min(3, height)):
                for c in range(width):
                    total += 1
                    position = c % 4
                    if r == 0:
                        expected = color1 if position in [0,2,3] else color2
                    elif r == 1 and height == 3:
                        expected = color1 if position in [0,2] else color2
                    elif r == 1 or (r == 2 and height == 3):
                        expected = color1 if position in [0,1,2] else color2
                    else:
                        expected = color1
                    if interior[r, c] == expected:
                        matches += 1
            return matches / total if total > 0 else 0
        except Exception as e:
            print(f"Error in _calculate_u_shape_match: {e}")
            return 0
    
    def _calculate_2x2_checker_match(self, interior, color1, color2):
        try:
            height, width = interior.shape
            matches = sum(
                1 for r in range(height) for c in range(width)
                if interior[r, c] == (color1 if c % 3 < 2 else color2)
            )
            total = height * width
            return matches / total if total > 0 else 0
        except Exception as e:
            print(f"Error in _calculate_2x2_checker_match: {e}")
            return 0
    
    def _identify_best_pattern(self, interior, pattern_color, background_color):
        try:
            scores = {
                3: self._calculate_pattern_match(interior, pattern_color, background_color, self._is_checkerboard_pattern),
                4: self._calculate_pattern_match(interior, pattern_color, background_color, self._is_stripe_pattern),
                5: self._calculate_pattern_match(interior, pattern_color, background_color, self._is_u_shape_pattern),
            }
            return max(scores, key=scores.get)
        except Exception as e:
            print(f"Error in _identify_best_pattern: {e}")
            return 5
    
    def _calculate_pattern_match(self, interior, color1, color2, pattern_func):
        try:
            height, width = interior.shape
            matches = sum(
                1 for r in range(height) for c in range(width)
                if interior[r, c] == pattern_func(r, c, height, color1, color2)
            )
            total = height * width
            return matches / total if total > 0 else 0
        except Exception as e:
            print(f"Error in _calculate_pattern_match: {e}")
            return 0
    
    def _is_checkerboard_pattern(self, r, c, height, color1, color2):
        return color1 if (r + c) % 2 == 0 else color2
    
    def _is_stripe_pattern(self, r, c, height, color1, color2):
        return color1 if r % 2 == 0 else color2
    
    def _is_u_shape_pattern(self, r, c, height, color1, color2):
        try:
            position = c % 4
            if height == 3:
                if r == 0:   return color1 if position in [0,2,3] else color2
                elif r == 1: return color1 if position in [0,2] else color2
                else:        return color1 if position in [0,1,2] else color2
            else:
                if r == 0:              return color1 if position in [0,2,3] else color2
                elif r == height - 1:   return color1 if position in [0,1,2] else color2
                elif r % 2 == 1:        return color1 if position in [0,1,2] else color2
                else:                   return color1 if position in [0,2,3] else color2
        except Exception as e:
            print(f"Error in _is_u_shape_pattern: {e}")
            return color1

