from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry, Contiguity, random_cell_coloring

class Task0a1d4ef5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "Each grid contains several single-colored rectangular objects placed on a background made of two differently colored cells and empty cells.",
            "To construct the input grid, first add approximately {(vars['rows'] * vars['cols']) // 2} colored cells using two different colors, randomly distributed to create the grid background.",
            "Once the background is created, place single-colored rectangular objects so they are roughly aligned horizontally and vertically.",
            "This means: after placing the first object in the top-left quadrant, the second object should be placed almost horizontally aligned with the first, to its right. The third object can either continue the horizontal alignment or be placed vertically below the first—however, if placed below, there must be exactly one object to its right.",
            "The goal is to ensure that, within each grid, all row groups containing objects have the same number of objects, and all column groups containing objects also have the same number.",
            "The sizes of the rectangular objects can vary, and the objects must not be connected to each other.",
            "The colors used for the background and the rectangular blocks must be different.",
            "No two adjacent rectangular blocks should have the same color.",
            "Each single-colored rectangular object should be more than 2 cells wide and 2 cells long."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are always smaller than the input grids.",
            "They are constructed by copying the input grid and identifying all single-colored rectangular objects that are more than 2 cells wide and 2 cells long.",
            "Once identified, add a cell in the output grid for each of these rectangular objects. The color of the cell should match the top-left cell of the corresponding rectangular object in the input grid.",
            "Continue this process, filling the output grid with one cell per qualifying rectangular object, based on their top-left colors.",
            "The size of the output grid is equal to the total number of qualifying rectangular objects in the input grid.",
            "The arrangement of cells in the output grid matches the spatial layout of the identified rectangular objects in the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {}
        
        # Choose grid size (between 12 and 30 rows/columns)
        rows = random.randint(12, 30)
        cols = random.randint(12, 30)
        
        # Ensure at least one dimension is even
        if rows % 2 != 0 and cols % 2 != 0:
            if random.choice([True, False]):
                rows += 1
            else:
                cols += 1
        
        taskvars['rows'] = rows
        taskvars['cols'] = cols
        
        # Choose number of training examples (3-6)
        num_train = random.randint(3, 6)
        
        # Generate train and test data
        train_data = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Generate test data
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_non_rectangular_background(self, rows, cols, background_colors):
        """
        Create a background pattern that doesn't form rectangular patterns
        using a mixture of two colors and empty cells
        """
        grid = np.zeros((rows, cols), dtype=int)
        
        # First, randomly distribute background colors
        for r in range(rows):
            for c in range(cols):
                # With 2/3 probability, use a background color
                if random.random() < 0.6:
                    grid[r, c] = random.choice(background_colors)
        
        # Check for and break up rectangular patterns (4-connected cells of same color)
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                # Check if we have a 2x2 square of the same color
                current = grid[r, c]
                if current == 0:
                    continue
                    
                # Check if we're part of a potential rectangle pattern
                neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                same_count = sum(1 for nr, nc in neighbors if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == current)
                
                # If we have too many neighbors of same color, break the pattern
                if same_count >= 3:
                    # Either change to another background color or make empty
                    options = [0] + [color for color in background_colors if color != current]
                    grid[r, c] = random.choice(options)
        
        return grid
    
    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        
        # Choose colors for background and objects
        available_colors = list(range(1, 10))  # Colors 1-9
        random.shuffle(available_colors)
        
        # Choose 2 colors for background
        background_colors = available_colors[:2]
        
        # Choose 3-7 colors for rectangular objects
        num_object_colors = random.randint(3, min(7, len(available_colors) - 2))
        object_colors = available_colors[2:2+num_object_colors]
        
        # Create non-rectangular background with two different colors and empty cells
        grid = self.create_non_rectangular_background(rows, cols, background_colors)
        
        # Determine grid layout - use a clear structure
        if min(rows, cols) <= 20:
            # For smaller grids, use 2x2 layout (4 objects total)
            num_row_groups = 2
            num_col_groups = 2
        else:
            # For larger grids, use a 3x3 or 2x3 layout
            num_row_groups = random.randint(2, 3)
            num_col_groups = random.randint(2, 3)
        
        # Choose standard rectangle size to ensure uniformity
        # Pick a size that will comfortably fit within the grid divisions
        std_height = min(rows // (num_row_groups * 2), 5)
        std_width = min(cols // (num_col_groups * 2), 6)
        
        # Ensure standard size is at least 3x3
        std_height = max(3, std_height)
        std_width = max(3, std_width)
        
        # Allow minor variation in size while keeping dimensions consistent
        rectangle_heights = [std_height + random.randint(-1, 1) for _ in range(num_row_groups * num_col_groups)]
        rectangle_widths = [std_width + random.randint(-1, 1) for _ in range(num_row_groups * num_col_groups)]
        
        # Ensure all heights and widths are at least 3
        rectangle_heights = [max(3, h) for h in rectangle_heights]
        rectangle_widths = [max(3, w) for w in rectangle_widths]
        
        # Calculate equal spacing for objects
        row_spacing = rows // num_row_groups
        col_spacing = cols // num_col_groups
        
        # Create precise anchor points for alignment
        row_anchors = [i * row_spacing + (row_spacing - std_height) // 2 for i in range(num_row_groups)]
        col_anchors = [i * col_spacing + (col_spacing - std_width) // 2 for i in range(num_col_groups)]
        
        # Track color usage to ensure variety
        object_color_counts = {color: 0 for color in object_colors}
        
        # Place the rectangles
        used_colors = {}  # Track which colors are used in which positions
        
        for r_idx, row_anchor in enumerate(row_anchors):
            for c_idx, col_anchor in enumerate(col_anchors):
                # Get rectangle dimensions from the pre-calculated arrays
                rect_idx = r_idx * num_col_groups + c_idx
                height = rectangle_heights[rect_idx]
                width = rectangle_widths[rect_idx]
                
                # Calculate precise position to ensure alignment
                r_pos = row_anchor
                c_pos = col_anchor
                
                # Ensure we don't place rectangles too close to the edge
                r_pos = min(max(r_pos, 1), rows - height - 1)
                c_pos = min(max(c_pos, 1), cols - width - 1)
                
                # Choose color, ensuring adjacent rectangles don't have the same color
                adjacent_positions = []
                
                # Check for adjacent rectangles
                if c_idx > 0:
                    adjacent_positions.append((r_idx, c_idx-1))
                if r_idx > 0:
                    adjacent_positions.append((r_idx-1, c_idx))
                
                # Get colors of adjacent rectangles
                adjacent_colors = [used_colors.get(pos) for pos in adjacent_positions if pos in used_colors]
                
                # Choose a color not used by adjacent rectangles
                # Prioritize colors that have been used less frequently
                valid_colors = [c for c in object_colors if c not in adjacent_colors]
                
                if not valid_colors:  # Fallback if no valid colors
                    valid_colors = object_colors
                
                # Sort by usage count to prioritize less-used colors
                valid_colors.sort(key=lambda c: object_color_counts[c])
                
                # Choose from the least-used valid colors
                # Introduce randomness while still favoring less-used colors
                selection_pool = valid_colors[:max(1, len(valid_colors)//2)]
                selected_color = random.choice(selection_pool)
                
                # Update color usage count
                object_color_counts[selected_color] += 1
                used_colors[(r_idx, c_idx)] = selected_color
                
                # Create and place the rectangle
                rectangle = np.full((height, width), selected_color)
                grid[r_pos:r_pos+height, c_pos:c_pos+width] = rectangle
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """
        Transform input grid to output grid according to the transformation reasoning chain.
        Identifies rectangular objects and creates a grid where each cell represents one object,
        with its color taken from the top-left cell of the corresponding object.
        """
        # Identify large rectangular blocks directly by scanning the grid
        rows, cols = grid.shape
        rectangles = []
        
        # Track which cells have been processed
        processed = np.zeros_like(grid, dtype=bool)
        
        # Scan grid to find large, single-colored rectangular objects
        for r in range(rows):
            for c in range(cols):
                # Skip if already processed or empty
                if processed[r, c] or grid[r, c] == 0:
                    continue
                
                # Found potential start of rectangular object
                color = grid[r, c]
                
                # Check if it's a rectangular background color
                if r < rows - 2 and c < cols - 2:  # Ensure room for at least 3x3
                    # Try to find rectangle dimensions
                    height = 1
                    width = 1
                    
                    # Expand width as long as we find the same color
                    while c + width < cols and grid[r, c + width] == color:
                        width += 1
                    
                    # Expand height as long as rows are filled with the same color
                    is_rectangle = True
                    while r + height < rows and is_rectangle:
                        # Check if next row has same color across width
                        for cc in range(c, c + width):
                            if cc >= cols or grid[r + height, cc] != color:
                                is_rectangle = False
                                break
                        if is_rectangle:
                            height += 1
                    
                    # Check if this is a valid rectangular object (at least 3x3)
                    if width >= 3 and height >= 3:
                        # Mark all cells in this rectangle as processed
                        processed[r:r+height, c:c+width] = True
                        
                        # Store rectangle info: top-left position and color
                        rectangles.append((r, c, color))
        
        # If no valid rectangles found, return a 1x1 empty grid
        if not rectangles:
            return np.zeros((1, 1), dtype=int)
        
        # Determine the spatial layout by finding distinct row and column positions
        unique_rows = sorted(set(r for r, _, _ in rectangles))
        unique_cols = sorted(set(c for _, c, _ in rectangles))
        
        # Create row and column mappings
        row_map = {r: i for i, r in enumerate(unique_rows)}
        col_map = {c: i for i, c in enumerate(unique_cols)}
        
        # Create output grid with the correct size
        output_grid = np.zeros((len(unique_rows), len(unique_cols)), dtype=int)
        
        # Fill the output grid using the top-left cell colors
        for r, c, color in rectangles:
            output_row = row_map[r]
            output_col = col_map[c]
            output_grid[output_row, output_col] = color
        
        return output_grid

