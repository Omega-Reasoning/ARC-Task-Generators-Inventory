from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects, BorderBehavior, GridObject, GridObjects

class Task05a7bcf2Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "The grids contain several {color('object_color')} objects, one {color('line')} line, and one {color('histogram')} histogram-shaped object.",
            "To construct the {color('object_color')} objects, first create a {color('object_color')} rectangular block. Then, occasionally add one or two single {color('object_color')} cells to extend one of the sides of the rectangle.",
            "The input grids can have one of the two possible orientations: vertical or horizontal.",
            "If the orientation is vertical, divide the grid vertically into three parts, creating three vertical sections.",
            "In the first vertical section (left-most), add several {color('object_color')} objects positioned vertically, meaning one object is placed above the other. However, the objects must not be 4-way connected to each other.",
            "The end of the first section is marked by a {color('line')} vertical line that starts from the first row and extends to the bottom of the section.",
            "The {color('histogram')} histogram-shaped object is designed with a vertical baseline, and several horizontal bars are connected to this baseline, making it resemble a histogram.",
            "The horizontal bars of the histogram must not exceed four cells in width.",
            "These bars are positioned on one side of the vertical baseline and must point towards the {color('object_color')} objects.",
            "The {color('object_color')} objects, the {color('line')} vertical line, and the {color('histogram')} histogram-shaped object must not be connected to each other.",
            "This configuration ensures that the last vertical section is completely filled with empty (0) cells.",
            "If the orientation is horizontal, divide the grid horizontally into three parts, creating three horizontal sections.",
            "In the first horizontal section (top-most), add several {color('object_color')} objects positioned horizontally, meaning objects are placed next to the each other. However, the objects must not be 4-way connected to each other.",
            "The end of the first section is marked by a {color('line')} horizontal line that starts from the first column and extends to the far right.",
            "The {color('histogram')} histogram-shaped object is designed with a horizontal baseline, and several vertical bars are connected to this baseline, making it resemble a histogram.",
            "The vertical bars of the histogram must not exceed four cells in height.",
            "These bars are positioned on one side of the horizontal base line and must point towards the {color('object_color')} objects.",
            "The {color('object_color')} objects, the {color('line')} horizontal line, and the {color('histogram')} histogram-shaped object must not be connected to each other.",
            "This configuration ensures that the last horizontal section is completely filled with empty (0) cells.",
            "To construct the {color('object_color')} objects in horizontal orientation, first create a rectangular block. Then, occasionally add one or two single {color('object_color')} cells to extend the side of the rectangle horizontally that is closest to the {color('line')} line."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid, changing the color of all {color('object_color')} objects to {color('new_color')}, and determining whether the input grid is vertically or horizontally oriented by checking if the {color('line')} line is vertical or horizontal.",
            "If the orientation is vertical, then for each row containing {color('new_color')} cells, add a short horizontal line made of {color('object_color')}, starting from the {color('new_color')} cell that is closest to the {color('line')} line and extending until the {color('line')} line is reached.",
            "Once the {color('line')} line is reached, extend the same lines further, but now using the {color('line')} color.",
            "These lines continue until they reach the {color('histogram')} histogram-shaped object.",
            "Whenever a {color('line')} cell encounters the first {color('histogram')} cell, it pushes that cell, along with any horizontally connected cells, further horizontally until a {color('histogram')} cell reaches the last column.",
            "If the orientation is horizontal, then for each column containing {color('new_color')} cells, add a short vertical line made of {color('object_color')} color, starting from the {color('new_color')} cell that is closest to the {color('line')} line and extending until the {color('line')} line is reached.",
            "Once the {color('line')} line is reached, extend the same lines further, but now using the {color('line')} color.",
            "These lines continue downward until they reach the {color('histogram')} histogram-shaped object.",
            "Whenever a {color('line')} cell encounters the first {color('histogram')} cell, it pushes that cell, along with any vertically connected cells, further downward until the {color('histogram')} cells reach the last row.",
            "Ensure in both orientations, the original {color('line')} line from the input grid remains unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Define task variables for grid
        taskvars = {
            'grid_size': random.randint(20, 30),
            'object_color': random.randint(1, 9)
        }
        
        # Select other colors that are different from object_color
        available_colors = [i for i in range(1, 10) if i != taskvars['object_color']]
        taskvars['line'] = random.choice(available_colors)
        available_colors = [i for i in available_colors if i != taskvars['line']]
        taskvars['histogram'] = random.choice(available_colors)
        available_colors = [i for i in available_colors if i != taskvars['histogram']]
        taskvars['new_color'] = random.choice(available_colors)
        
        # Create train-test data
        train_pairs = []
        
        # Create one vertical-orientation grid for training
        gridvars = {'orientation': 'vertical', 'objects_position': 'left', 'is_test': False}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Create two horizontal-orientation grids for training
        for _ in range(2):
            gridvars = {'orientation': 'horizontal', 'objects_position': 'top', 'is_test': False}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Create test grid with vertical orientation and objects on right
        test_pairs = []
        gridvars = {'orientation': 'vertical', 'objects_position': 'right', 'is_test': True}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Create another test grid (random orientation)
        gridvars = {'orientation': random.choice(['vertical', 'horizontal']), 
                   'objects_position': random.choice(['left', 'top']), 
                   'is_test': True}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Return task variables and train-test data
        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }
        
        return taskvars, train_test_data
    
    def place_shape(self, grid, top, left, shape_id, color):
        """
        Place a shape at the specified position
        Returns the grid and the shape bounds (top, left, bottom, right)
        """
        # Shape definitions
        if shape_id == 0:  # 2x2 shape: [[c,c], [0,c]]
            shape = np.array([[color, color], [0, color]])
            height, width = 2, 2
        elif shape_id == 1:  # 2x3 shape: [[c,c], [c,c], [0,c]]
            shape = np.array([[color, color], [color, color], [0, color]])
            height, width = 3, 2
        elif shape_id == 2:  # 1x3 shape
            shape = np.array([[color], [color], [color]])
            height, width = 3, 1
        elif shape_id == 3:  # Alternate 2x2 shape
            shape = np.array([[color, color], [color, 0]])
            height, width = 2, 2
        
        # Check if shape fits
        if top + height > grid.shape[0] or left + width > grid.shape[1]:
            return grid, None
        
        # Place shape
        for r in range(height):
            for c in range(width):
                if r < shape.shape[0] and c < shape.shape[1] and shape[r, c] != 0:
                    grid[top + r, left + c] = shape[r, c]
        
        return grid, (top, left, top + height - 1, left + width - 1)
    
    def place_shape_horizontal(self, grid, top, left, shape_id, color):
        """Place a horizontally oriented shape"""
        # Shape definitions for horizontal orientation
        if shape_id == 0:  # 2x2 shape: [[c,c], [0,c]] -> no change
            shape = np.array([[color, color], [0, color]])
            height, width = 2, 2
        elif shape_id == 1:  # 2x3 flipped to 3x2: [[c,0], [c,c], [c,c]]
            shape = np.array([[color, 0], [color, color], [color, color]])
            height, width = 3, 2
        elif shape_id == 2:  # 1x3 flipped to 3x1
            shape = np.array([[color, color, color]])
            height, width = 1, 3
        elif shape_id == 3:  # Alternate 2x2 shape
            shape = np.array([[color, color], [color, 0]])
            height, width = 2, 2
        
        # Check if shape fits
        if top + height > grid.shape[0] or left + width > grid.shape[1]:
            return grid, None
        
        # Place shape
        for r in range(height):
            for c in range(width):
                if r < shape.shape[0] and c < shape.shape[1] and shape[r, c] != 0:
                    grid[top + r, left + c] = shape[r, c]
        
        return grid, (top, left, top + height - 1, left + width - 1)
    
    def shapes_are_separated(self, bounds1, bounds2, min_distance=2):
        """Check if shapes are separated by minimum distance"""
        if bounds1 is None or bounds2 is None:
            return True
            
        top1, left1, bottom1, right1 = bounds1
        top2, left2, bottom2, right2 = bounds2
        
        # Objects are separate if they have enough distance horizontally or vertically
        horizontal_separation = (right1 + min_distance <= left2) or (right2 + min_distance <= left1)
        vertical_separation = (bottom1 + min_distance <= top2) or (bottom2 + min_distance <= top1)
        
        return horizontal_separation or vertical_separation
    
    def create_histogram(self, grid, baseline_pos, orientation, objects_position, line_pos, histogram_color, is_test=False):
        """Create a histogram with exactly grid_size-5 bars of different lengths"""
        grid_size = grid.shape[0]
        
        # Number of bars should be grid_size-5
        num_bars = grid_size - 5
        
        # For test grids, ensure histogram baseline is not on first or last row/column
        if is_test:
            if orientation == 'vertical':
                # Adjust baseline position if needed (not on first or last column)
                if baseline_pos == 0:
                    baseline_pos = 1
                elif baseline_pos == grid_size - 1:
                    baseline_pos = grid_size - 2
            else:  # horizontal
                # Adjust baseline position if needed (not on first or last row)
                if baseline_pos == 0:
                    baseline_pos = 1
                elif baseline_pos == grid_size - 1:
                    baseline_pos = grid_size - 2
        
        # Generate different bar lengths
        prev_length = None
        bar_lengths = []
        
        for _ in range(num_bars):
            # Choose a different length from the previous one
            available_lengths = [1, 2, 3, 4]
            if prev_length is not None and prev_length in available_lengths:
                available_lengths.remove(prev_length)
            length = random.choice(available_lengths)
            bar_lengths.append(length)
            prev_length = length
        
        if orientation == 'vertical':
            # Draw vertical baseline
            for r in range(grid_size):
                grid[r, baseline_pos] = histogram_color
            
            # Select evenly spaced positions for bars
            bar_rows = np.linspace(2, grid_size - 3, num_bars, dtype=int)
            
            for i, r in enumerate(bar_rows):
                # Use different lengths for consecutive bars
                bar_length = bar_lengths[i % len(bar_lengths)]
                
                # Draw horizontal bar
                if objects_position == 'left':
                    # Bars point left (toward line)
                    for c in range(baseline_pos - bar_length, baseline_pos):
                        if c > line_pos + 1:  # Keep separation from line
                            grid[r, c] = histogram_color
                else:  # objects on right
                    # Bars point right (toward line)
                    for c in range(baseline_pos + 1, baseline_pos + 1 + bar_length):
                        if c < line_pos - 1:  # Keep separation from line
                            grid[r, c] = histogram_color
        
        else:  # horizontal orientation
            # Draw horizontal baseline
            for c in range(grid_size):
                grid[baseline_pos, c] = histogram_color
            
            # Select evenly spaced positions for bars
            bar_cols = np.linspace(2, grid_size - 3, num_bars, dtype=int)
            
            for i, c in enumerate(bar_cols):
                # Use different lengths for consecutive bars
                bar_length = bar_lengths[i % len(bar_lengths)]
                
                # Draw vertical bar
                if objects_position == 'top':
                    # Bars point up (toward line)
                    for r in range(baseline_pos - bar_length, baseline_pos):
                        if r > line_pos + 1:  # Keep separation from line
                            grid[r, c] = histogram_color
                else:  # objects on bottom (not used)
                    # Bars point down (toward line)
                    for r in range(baseline_pos + 1, baseline_pos + 1 + bar_length):
                        if r < line_pos - 1:  # Keep separation from line
                            grid[r, c] = histogram_color
        
        return grid
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        line_color = taskvars['line']
        histogram_color = taskvars['histogram']
        orientation = gridvars['orientation']
        objects_position = gridvars['objects_position']
        is_test = gridvars.get('is_test', False)
        
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Section size for dividing the grid
        section_size = grid_size // 3
        
        # Set positions for objects, line, and histogram
        if orientation == 'vertical':
            if objects_position == 'left':
                objects_end = section_size - 2  # Leave 2-cell gap from line
                line_pos = section_size
                histogram_pos = 2 * section_size - 1
            else:  # objects on right
                histogram_pos = section_size - 1  # Position histogram in the first section
                line_pos = 2 * section_size - 1
                objects_end = grid_size - 1
            
            # For test grids, ensure histogram baseline is not on edge
            if is_test:
                if histogram_pos == 0:
                    histogram_pos = 1
                elif histogram_pos == grid_size - 1:
                    histogram_pos = grid_size - 2
            
            # Create vertical line
            for r in range(grid_size):
                grid[r, line_pos] = line_color
            
            # Place objects in a vertical arrangement
            num_objects = random.randint(3, 6)
            row_height = grid_size // (num_objects + 1)
            placed_objects = []
            
            for i in range(num_objects):
                # Try several positions until finding one that works
                for _ in range(10):
                    # Calculate vertical position in its row
                    top = (i + 1) * row_height - random.randint(1, 3)
                    top = min(top, grid_size - 4)  # Ensure objects fit
                    
                    # Choose shape
                    shape_id = random.randint(0, 3)
                    
                    # Calculate horizontal position
                    if objects_position == 'left':
                        left = random.randint(0, objects_end - 3)
                    else:
                        left = random.randint(line_pos + 2, grid_size - 3)
                    
                    # Check if this position would separate this object from others
                    new_bounds = (top, left, top + 3, left + 3)  # Maximum possible bounds
                    all_separate = True
                    
                    for bounds in placed_objects:
                        if not self.shapes_are_separated(new_bounds, bounds):
                            all_separate = False
                            break
                    
                    if all_separate:
                        # Place the object using the correct template
                        grid, actual_bounds = self.place_shape(grid, top, left, shape_id, object_color)
                        if actual_bounds is not None:
                            placed_objects.append(actual_bounds)
                            break
            
            # Create histogram
            self.create_histogram(grid, histogram_pos, orientation, objects_position, line_pos, histogram_color, is_test)
        
        else:  # horizontal orientation
            if objects_position == 'top':
                objects_end = section_size - 2
                line_pos = section_size
                histogram_pos = 2 * section_size - 1
            else:
                histogram_pos = section_size - 1
                line_pos = 2 * section_size - 1
                objects_end = grid_size - 1
            
            # For test grids, ensure histogram baseline is not on edge
            if is_test:
                if histogram_pos == 0:
                    histogram_pos = 1
                elif histogram_pos == grid_size - 1:
                    histogram_pos = grid_size - 2
            
            # Create horizontal line
            for c in range(grid_size):
                grid[line_pos, c] = line_color
            
            # Place objects in a horizontal arrangement
            num_objects = random.randint(3, 6)
            col_width = grid_size // (num_objects + 1)
            placed_objects = []
            
            for i in range(num_objects):
                # Try several positions until finding one that works
                for _ in range(10):
                    # Calculate horizontal position in its column
                    left = (i + 1) * col_width - random.randint(1, 3)
                    left = min(left, grid_size - 4)  # Ensure objects fit
                    
                    # Choose shape
                    shape_id = random.randint(0, 3)
                    
                    # Calculate vertical position
                    if objects_position == 'top':
                        top = random.randint(0, objects_end - 3)
                    else:
                        top = random.randint(line_pos + 2, grid_size - 3)
                    
                    # Check if this position would separate this object from others
                    new_bounds = (top, left, top + 3, left + 3)  # Maximum possible bounds
                    all_separate = True
                    
                    for bounds in placed_objects:
                        if not self.shapes_are_separated(new_bounds, bounds):
                            all_separate = False
                            break
                    
                    if all_separate:
                        # Place the object using the correct horizontal template
                        grid, actual_bounds = self.place_shape_horizontal(grid, top, left, shape_id, object_color)
                        if actual_bounds is not None:
                            placed_objects.append(actual_bounds)
                            break
            
            # Create histogram
            self.create_histogram(grid, histogram_pos, orientation, objects_position, line_pos, histogram_color, is_test)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        # Initialize output grid by copying input
        result = grid.copy()
        
        object_color = taskvars['object_color']
        new_color = taskvars['new_color']
        line_color = taskvars['line']
        histogram_color = taskvars['histogram']
        
        # Convert object_color to new_color
        result[result == object_color] = new_color
        
        # Determine orientation by checking for vertical or horizontal line
        is_vertical_orientation = False
        line_position = -1  # Initialize with default value
        
        # Check for vertical line
        for col in range(grid.shape[1]):
            # Consider it a vertical line if all or most cells in column are line_color
            if (grid[:, col] == line_color).sum() > 0.8 * grid.shape[0]:
                is_vertical_orientation = True
                line_position = col
                break
        
        # If no vertical line found, check for horizontal line
        if line_position == -1:
            for row in range(grid.shape[0]):
                # Consider it a horizontal line if all or most cells in row are line_color
                if (grid[row, :] == line_color).sum() > 0.8 * grid.shape[1]:
                    line_position = row
                    break
        
        # If still no line found, default to first occurrence of line_color
        if line_position == -1:
            line_positions = np.where(grid == line_color)
            if len(line_positions[0]) > 0:
                if is_vertical_orientation:
                    line_position = line_positions[1][0]  # Column of first occurrence
                else:
                    line_position = line_positions[0][0]  # Row of first occurrence
            else:
                # No line color found at all, use middle of grid as fallback
                line_position = grid.shape[0] // 2
        
        # Process the grid based on orientation
        grid_size = grid.shape[0]  # Assuming square grid
        
        if is_vertical_orientation:
            # Find the histogram baseline column
            histogram_col = -1
            for col in range(grid_size):
                if col != line_position and (grid[:, col] == histogram_color).sum() > 0.8 * grid.shape[0]:
                    histogram_col = col
                    break
            
            # If histogram baseline not found, find most frequent histogram column
            if histogram_col == -1:
                histogram_positions = np.where(grid == histogram_color)
                if len(histogram_positions[0]) > 0:
                    # Count occurrences of each column
                    unique_cols, counts = np.unique(histogram_positions[1], return_counts=True)
                    # Find column with most histogram cells (likely the baseline)
                    histogram_col = unique_cols[np.argmax(counts)]
            
            # For each row containing new_color cells
            for row in range(grid_size):
                # Find new_color cells in this row
                new_color_indices = np.where(result[row, :] == new_color)[0]
                if len(new_color_indices) == 0:
                    continue
                
                # Determine if objects are left or right of line
                objects_on_left = np.min(new_color_indices) < line_position
                
                if objects_on_left:
                    # Find rightmost new_color cell in this row
                    start_col = np.max(new_color_indices)
                    
                    # Draw line from object to line with object_color
                    for col in range(start_col + 1, line_position):
                        result[row, col] = object_color
                    
                    # Continue with line_color until finding histogram or reaching right edge
                    found_histogram = False
                    histogram_cells = []
                    histogram_start = None
                    
                    # Fill with line_color from line to histogram or edge
                    for col in range(line_position + 1, grid_size):
                        if grid[row, col] == histogram_color:
                            histogram_start = col
                            found_histogram = True
                            break
                        result[row, col] = line_color
                    
                    # If no histogram found in this row, continue to next row
                    if not found_histogram:
                        continue
                    
                    # Find all histogram cells in this row
                    col = histogram_start
                    while col < grid_size and grid[row, col] == histogram_color:
                        histogram_cells.append(col)
                        col += 1
                    
                    # Calculate how many cells to shift the histogram
                    if histogram_cells:
                        # Determine last column of grid
                        last_col = grid_size - 1
                        
                        # Calculate how many cells to shift
                        shift = last_col - histogram_cells[-1]
                        
                        # Clear original histogram positions
                        for col in histogram_cells:
                            result[row, col] = 0
                        
                        # Place histogram cells at the edge, keeping their relative positions
                        for col in histogram_cells:
                            new_col = col + shift
                            if new_col <= last_col:  # Safety check
                                result[row, new_col] = histogram_color
                        
                        # Fill the gap between line color and histogram with line_color
                        # to ensure continuity
                        last_line_col = histogram_start - 1
                        first_hist_col = histogram_cells[0] + shift
                        for col in range(last_line_col + 1, first_hist_col):
                            result[row, col] = line_color
                
                else:  # Objects on right
                    # Find leftmost new_color cell in this row
                    start_col = np.min(new_color_indices)
                    
                    # Draw line from object to line with object_color
                    for col in range(start_col - 1, line_position, -1):
                        result[row, col] = object_color
                    
                    # Continue with line_color until finding histogram or reaching left edge
                    found_histogram = False
                    histogram_cells = []
                    histogram_start = None
                    
                    # Fill with line_color from line to histogram or edge
                    for col in range(line_position - 1, -1, -1):
                        if grid[row, col] == histogram_color:
                            histogram_start = col
                            found_histogram = True
                            break
                        result[row, col] = line_color
                    
                    # If no histogram found in this row, continue to next row
                    if not found_histogram:
                        continue
                    
                    # Find all histogram cells in this row
                    col = histogram_start
                    while col >= 0 and grid[row, col] == histogram_color:
                        histogram_cells.append(col)
                        col -= 1
                    
                    # Calculate how many cells to shift the histogram
                    if histogram_cells:
                        # Determine leftmost column to place histogram
                        first_col = 0
                        
                        # Sort histogram cells in ascending order (left to right)
                        histogram_cells.sort()
                        
                        # Clear original histogram positions
                        for col in histogram_cells:
                            result[row, col] = 0
                        
                        # Calculate new positions with histogram at the left edge
                        shifted_positions = [first_col + i for i in range(len(histogram_cells))]
                        
                        # Place histogram cells at the edge
                        for i, col in enumerate(shifted_positions):
                            result[row, col] = histogram_color
                        
                        # Fill the gap between line color and histogram with line_color
                        # to ensure continuity
                        last_hist_col = shifted_positions[-1]
                        first_line_col = histogram_start + 1
                        for col in range(last_hist_col + 1, first_line_col):
                            result[row, col] = line_color
        
        else:  # Horizontal orientation
            # Find the histogram baseline row
            histogram_row = -1
            for row in range(grid_size):
                if row != line_position and (grid[row, :] == histogram_color).sum() > 0.8 * grid.shape[1]:
                    histogram_row = row
                    break
            
            # If histogram baseline not found, find most frequent histogram row
            if histogram_row == -1:
                histogram_positions = np.where(grid == histogram_color)
                if len(histogram_positions[0]) > 0:
                    # Count occurrences of each row
                    unique_rows, counts = np.unique(histogram_positions[0], return_counts=True)
                    # Find row with most histogram cells (likely the baseline)
                    histogram_row = unique_rows[np.argmax(counts)]
            
            # For each column containing new_color cells
            for col in range(grid_size):
                # Find new_color cells in this column
                new_color_indices = np.where(result[:, col] == new_color)[0]
                if len(new_color_indices) == 0:
                    continue
                
                # Determine if objects are above or below line
                objects_on_top = np.min(new_color_indices) < line_position
                
                if objects_on_top:
                    # Find bottommost new_color cell in this column
                    start_row = np.max(new_color_indices)
                    
                    # Draw line from object to line with object_color
                    for row in range(start_row + 1, line_position):
                        result[row, col] = object_color
                    
                    # Continue with line_color until finding histogram or reaching bottom edge
                    found_histogram = False
                    histogram_cells = []
                    histogram_start = None
                    
                    # Fill with line_color from line to histogram or edge
                    for row in range(line_position + 1, grid_size):
                        if grid[row, col] == histogram_color:
                            histogram_start = row
                            found_histogram = True
                            break
                        result[row, col] = line_color
                    
                    # If no histogram found in this column, continue to next column
                    if not found_histogram:
                        continue
                    
                    # Find all histogram cells in this column
                    row = histogram_start
                    while row < grid_size and grid[row, col] == histogram_color:
                        histogram_cells.append(row)
                        row += 1
                    
                    # Calculate how many cells to shift the histogram
                    if histogram_cells:
                        # Determine bottom row of grid
                        last_row = grid_size - 1
                        
                        # Calculate how many cells to shift
                        shift = last_row - histogram_cells[-1]
                        
                        # Clear original histogram positions
                        for row in histogram_cells:
                            result[row, col] = 0
                        
                        # Place histogram cells at the bottom edge, keeping their relative positions
                        for row in histogram_cells:
                            new_row = row + shift
                            if new_row <= last_row:  # Safety check
                                result[new_row, col] = histogram_color
                        
                        # Fill the gap between line color and histogram with line_color
                        # to ensure continuity
                        last_line_row = histogram_start - 1
                        first_hist_row = histogram_cells[0] + shift
                        for row in range(last_line_row + 1, first_hist_row):
                            result[row, col] = line_color
                
                else:  # Objects on bottom (not used in current constraints)
                    # Find topmost new_color cell in this column
                    start_row = np.min(new_color_indices)
                    
                    # Draw line from object to line with object_color
                    for row in range(start_row - 1, line_position, -1):
                        result[row, col] = object_color
                    
                    # Continue with line_color until finding histogram or reaching top edge
                    found_histogram = False
                    histogram_cells = []
                    histogram_start = None
                    
                    # Fill with line_color from line to histogram or edge
                    for row in range(line_position - 1, -1, -1):
                        if grid[row, col] == histogram_color:
                            histogram_start = row
                            found_histogram = True
                            break
                        result[row, col] = line_color
                    
                    # If no histogram found in this column, continue to next column
                    if not found_histogram:
                        continue
                    
                    # Find all histogram cells in this column
                    row = histogram_start
                    while row >= 0 and grid[row, col] == histogram_color:
                        histogram_cells.append(row)
                        row -= 1
                    
                    # Calculate how many cells to shift the histogram
                    if histogram_cells:
                        # Determine top row of grid
                        first_row = 0
                        
                        # Sort histogram cells in ascending order (top to bottom)
                        histogram_cells.sort()
                        
                        # Clear original histogram positions
                        for row in histogram_cells:
                            result[row, col] = 0
                        
                        # Calculate new positions with histogram at the top edge
                        shifted_positions = [first_row + i for i in range(len(histogram_cells))]
                        
                        # Place histogram cells at the edge
                        for i, row in enumerate(shifted_positions):
                            result[row, col] = histogram_color
                        
                        # Fill the gap between line color and histogram with line_color
                        # to ensure continuity
                        last_hist_row = shifted_positions[-1]
                        first_line_row = histogram_start + 1
                        for row in range(last_hist_row + 1, first_line_row):
                            result[row, col] = line_color
        
        return result

