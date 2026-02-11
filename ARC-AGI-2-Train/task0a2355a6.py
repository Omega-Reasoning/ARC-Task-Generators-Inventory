from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from scipy.ndimage import label
from input_library import create_object, retry, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task0a2355a6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains several {color('object_color')} objects, where each object is made of 4-way connected cells of {color('object_color')} color and the remaining cells are empty (0).",
            "The {color('object_color')} objects are shaped such that they form a one-cell wide boundary enclosing several empty regions, carved into a square or rectangle, or sometimes a combination of multiple 4-way connected square or rectangular loops, but the boundary remains one cell thick at all points.",
            "Each boundary encloses at least one internal empty (0) region.",
            "The enclosed empty region may consist of one or more empty (0) cells.",
            "The empty region is always rectangular or square in shape and is completely surrounded on all four sides by the {color('object_color')} frame.",
            "All {color('object_color')} objects are completely separated from one another, with no touching or overlapping between objects.",
            "As a result, the objects resemble boxed structures or hollow squares/rectangles with exactly one-cell wide frames."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the number of empty (0) regions inside each {color('object_color')} object.",
            "Once the number of enclosed empty regions has been determined, the color of each {color('object_color')} object is updated accordingly.",
            "All objects that contain exactly one empty (0) region are recolored to {color('change_color1')}.",
            "All objects that contain exactly two empty (0) regions are recolored to {color('change_color2')}.",
            "All objects that contain exactly three empty (0) regions are recolored to {color('change_color3')}.",
            "All objects that contain exactly four empty (0) regions are recolored to {color('change_color4')}.",
            "The overall shape of each object is preserved."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        taskvars = {}
        
        # Select distinct colors
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        taskvars['object_color'] = available_colors[0]
        taskvars['change_color1'] = available_colors[1]
        taskvars['change_color2'] = available_colors[2]
        taskvars['change_color3'] = available_colors[3]
        taskvars['change_color4'] = available_colors[4]
        
        # Create training examples
        num_train_examples = random.randint(3, 6)
        train_examples = []
        
        # Ensure we have one grid with specific requirements (1, 2, and 3 empty regions)
        mandatory_grid_created = False
        
        for i in range(num_train_examples):
            # If it's the first example and we haven't created our mandatory grid yet
            if i == 0 and not mandatory_grid_created:
                input_grid = self.create_input(taskvars, {'mandatory': True})
                mandatory_grid_created = True
            else:
                input_grid = self.create_input(taskvars, {})
            
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
        
        return taskvars, train_test_data
    
    def create_rectangular_frame(self, grid, row, col, width, height, color):
        """Create a simple rectangular frame with exactly one-cell wide border"""
        # Safety check
        if width < 3 or height < 3:
            return False
        
        # Check bounds to prevent index errors
        if row < 0 or col < 0 or row + height > grid.shape[0] or col + width > grid.shape[1]:
            return False
        
        # Draw top and bottom borders
        for c in range(col, col + width):
            grid[row, c] = color  # Top border
            grid[row + height - 1, c] = color  # Bottom border
            
        # Draw left and right borders
        for r in range(row + 1, row + height - 1):
            grid[r, col] = color  # Left border
            grid[r, col + width - 1] = color  # Right border
            
        return True
    
    def create_template_object(self, grid, template_type, start_row, start_col, color):
        """Create a simple template object (with no nested shapes)"""
        
        # Object 1: Simple rectangular frame (1 empty region)
        if template_type == 1:
            width = random.randint(5, 8)
            height = random.randint(5, 8)
            return self.create_rectangular_frame(grid, start_row, start_col, width, height, color), (start_row, start_col, width, height)
        
        # Object 2: Vertical object with 3 empty regions
        elif template_type == 2:
            pattern = [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1]
            ]
            height, width = len(pattern), len(pattern[0])
            
            # Check if pattern fits within grid bounds
            if start_row + height > grid.shape[0] or start_col + width > grid.shape[1]:
                return False, (0, 0, 0, 0)
            
            # Place the pattern at the specified position
            for r in range(height):
                for c in range(width):
                    if pattern[r][c] == 1:
                        grid[start_row + r, start_col + c] = color
            
            return True, (start_row, start_col, width, height)
        
        # Object 3: Adjacent rectangles (2 empty regions)
        elif template_type == 3:
            # Create rectangle with internal divider
            width = random.randint(8, 12)
            height = random.randint(5, 8)
            
            # Check if fits within grid bounds
            if start_row + height > grid.shape[0] or start_col + width > grid.shape[1]:
                return False, (0, 0, 0, 0)
            
            # Create outer frame
            if not self.create_rectangular_frame(grid, start_row, start_col, width, height, color):
                return False, (0, 0, 0, 0)
            
            # Add vertical divider
            div_col = start_col + width // 2
            for r in range(start_row + 1, start_row + height - 1):
                grid[r, div_col] = color
                
            return True, (start_row, start_col, width, height)
        
        # Object 4: Small hollow square (1 empty region)
        elif template_type == 4:
            width = height = 5
            return self.create_rectangular_frame(grid, start_row, start_col, width, height, color), (start_row, start_col, width, height)
            
        # Object 5: L-shape + square (2 empty regions)
        elif template_type == 5:
            pattern = [
                [0, 0, 1, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 0, 1],
                [0, 0, 1, 1, 1]
            ]
            height, width = len(pattern), len(pattern[0])
            
            # Check if pattern fits within grid bounds
            if start_row + height > grid.shape[0] or start_col + width > grid.shape[1]:
                return False, (0, 0, 0, 0)
            
            # Place the pattern at the specified position
            for r in range(height):
                for c in range(width):
                    if pattern[r][c] == 1:
                        grid[start_row + r, start_col + c] = color
            
            return True, (start_row, start_col, width, height)
            
        # Object 6: L-shape + square (2 empty regions)
        elif template_type == 6:
            pattern = [
                [0, 0, 0, 0, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1, 0, 1],
                [0, 0, 0, 0, 1, 1, 1]
            ]
            height, width = len(pattern), len(pattern[0])
            
            # Check if pattern fits within grid bounds
            if start_row + height > grid.shape[0] or start_col + width > grid.shape[1]:
                return False, (0, 0, 0, 0)
            
            # Place the pattern at the specified position
            for r in range(height):
                for c in range(width):
                    if pattern[r][c] == 1:
                        grid[start_row + r, start_col + c] = color
            
            return True, (start_row, start_col, width, height)
        
        # Object 7: Middle-left L-shape + square (2 empty regions)
        elif template_type == 7:
            pattern = [
                [0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1, 0],
                [0, 0, 1, 1, 1, 1, 0]
            ]
            height, width = len(pattern), len(pattern[0])
            
            # Check if pattern fits within grid bounds
            if start_row + height > grid.shape[0] or start_col + width > grid.shape[1]:
                return False, (0, 0, 0, 0)
            
            # Place the pattern at the specified position
            for r in range(height):
                for c in range(width):
                    if pattern[r][c] == 1:
                        grid[start_row + r, start_col + c] = color
            
            return True, (start_row, start_col, width, height)
        
        return False, (0, 0, 0, 0)
    
    def create_frame_with_internal_dividers(self, grid, row, col, width, height, num_regions, color):
        """Create a rectangular frame with internal dividers to form the specified number of enclosed regions"""
        
        # Safety checks
        if width < 5 or height < 5:
            return False
        
        # Check bounds to prevent index errors
        if row < 0 or col < 0 or row + height > grid.shape[0] or col + width > grid.shape[1]:
            return False
        
        # Create the outer frame
        if not self.create_rectangular_frame(grid, row, col, width, height, color):
            return False
            
        # Add internal walls to create the required number of regions
        if num_regions == 2:
            # Add one wall that divides the frame into two regions
            if random.choice([True, False]):
                # Horizontal divider
                # Ensure at least 2 cells on each side
                if height <= 4:  # Too small for proper division
                    return False
                
                div_row = row + max(2, min(height - 3, random.randint(2, height - 3)))
                for c in range(col + 1, col + width - 1):
                    grid[div_row, c] = color
            else:
                # Vertical divider
                # Ensure at least 2 cells on each side
                if width <= 4:  # Too small for proper division
                    return False
                
                div_col = col + max(2, min(width - 3, random.randint(2, width - 3)))
                for r in range(row + 1, row + height - 1):
                    grid[r, div_col] = color
        
        elif num_regions == 3:
            # T-shaped divider to create 3 regions
            # Ensure the frame is large enough
            if width <= 5 or height <= 5:
                return False
                
            if random.choice([True, False]):
                # |- shape
                v_div = col + max(2, min(width - 3, random.randint(width // 3, 2 * width // 3)))
                for r in range(row + 1, row + height - 1):
                    grid[r, v_div] = color
                
                # Add horizontal segment to top or bottom half
                if random.choice([True, False]):
                    h_div = row + max(2, min(height - 3, random.randint(2, height // 2)))
                else:
                    h_div = row + max(height // 2, min(height - 3, random.randint(height // 2, height - 3)))
                
                # Horizontal segment from vertical line to either left or right edge
                if random.choice([True, False]):
                    for c in range(col + 1, v_div):
                        grid[h_div, c] = color
                else:
                    for c in range(v_div + 1, col + width - 1):
                        grid[h_div, c] = color
            else:
                # -| shape
                h_div = row + max(2, min(height - 3, random.randint(height // 3, 2 * height // 3)))
                for c in range(col + 1, col + width - 1):
                    grid[h_div, c] = color
                
                # Add vertical segment to left or right half
                if random.choice([True, False]):
                    v_div = col + max(2, min(width - 3, random.randint(2, width // 2)))
                else:
                    v_div = col + max(width // 2, min(width - 3, random.randint(width // 2, width - 3)))
                
                # Vertical segment from horizontal line to either top or bottom edge
                if random.choice([True, False]):
                    for r in range(row + 1, h_div):
                        grid[r, v_div] = color
                else:
                    for r in range(h_div + 1, row + height - 1):
                        grid[r, v_div] = color
        
        elif num_regions == 4:
            # Cross-shaped divider to create 4 regions
            # Ensure the frame is large enough
            if width <= 5 or height <= 5:
                return False
                
            h_div = row + max(2, min(height - 3, random.randint(height // 3, 2 * height // 3)))
            v_div = col + max(2, min(width - 3, random.randint(width // 3, 2 * width // 3)))
            
            # Draw horizontal divider
            for c in range(col + 1, col + width - 1):
                grid[h_div, c] = color
            
            # Draw vertical divider
            for r in range(row + 1, row + height - 1):
                grid[r, v_div] = color
                
        return True
    
    def create_frame_with_regions(self, grid, num_regions):
        """Create a frame with specified number of enclosed regions"""
        rows, cols = grid.shape
        color = 1  # Temporary color value
        
        # Randomly choose between template objects and generated objects
        use_template = random.random() < 0.6  # 60% chance to use template
        
        if use_template:
            # Map number of regions to possible template types
            template_options = {
                1: [1, 4],             # Templates with 1 region
                2: [3, 5, 6, 7],       # Templates with 2 regions (including new requested shape)
                3: [2],                # Templates with 3 regions
            }
            
            # Get valid templates for this number of regions
            options = template_options.get(num_regions, [])
            
            if options:
                # Choose a random template
                template_type = random.choice(options)
                
                # Get template dimensions based on template type
                if template_type == 1:
                    height, width = random.randint(5, 8), random.randint(5, 8)
                elif template_type == 2:
                    height, width = 7, 3
                elif template_type == 3:
                    height, width = random.randint(5, 8), random.randint(8, 12)
                elif template_type == 4:
                    height, width = 5, 5
                elif template_type == 5:  # First L-shaped template
                    height, width = 5, 5
                elif template_type == 6:  # Second L-shaped template
                    height, width = 5, 7
                elif template_type == 7:  # Middle-left L-shape
                    height, width = 7, 7
                
                # Check if it fits in the grid
                if height < rows - 2 and width < cols - 2:
                    # Position within grid (with safety margins)
                    start_row = random.randint(1, rows - height - 1)
                    start_col = random.randint(1, cols - width - 1)
                    
                    # Create the template
                    success, area = self.create_template_object(grid, template_type, start_row, start_col, color)
                    if success:
                        return [area]
        
        # For single region, create a simple rectangular frame
        if num_regions == 1:
            width = min(cols - 4, random.randint(5, 10))
            height = min(rows - 4, random.randint(5, 10))
            
            # Ensure minimum size
            width = max(width, 5)
            height = max(height, 5)
            
            # Safety bounds check
            row = random.randint(1, max(1, rows - height - 1))
            col = random.randint(1, max(1, cols - width - 1))
            
            if self.create_rectangular_frame(grid, row, col, width, height, color):
                return [(row, col, width, height)]
            return []
        
        # For multiple regions, create a frame with internal dividers
        width = min(cols - 4, random.randint(8, 14))
        height = min(rows - 4, random.randint(8, 14))
        
        # Ensure minimum size based on number of regions
        width = max(width, 5 + num_regions)
        height = max(height, 5 + num_regions)
        
        # Make sure width and height are not too large
        width = min(width, cols - 4)
        height = min(height, rows - 4)
        
        # Position the frame (with safety bounds check)
        row = random.randint(1, max(1, rows - height - 1))
        col = random.randint(1, max(1, cols - width - 1))
        
        # Create the frame with internal dividers
        if self.create_frame_with_internal_dividers(grid, row, col, width, height, num_regions, color):
            return [(row, col, width, height)]
            
        # Fallback: create a single region if we couldn't create the requested number
        width = min(cols - 4, random.randint(5, 10))
        height = min(rows - 4, random.randint(5, 10))
        
        width = max(width, 5)
        height = max(height, 5)
        
        # Safety bounds check
        row = random.randint(1, max(1, rows - height - 1))
        col = random.randint(1, max(1, cols - width - 1))
        
        if self.create_rectangular_frame(grid, row, col, width, height, color):
            return [(row, col, width, height)]
            
        return []
    
    def count_enclosed_regions(self, grid, object_coords):
        """Count the number of enclosed empty regions within an object"""
        # Create a temporary grid with only this object
        rows, cols = grid.shape
        temp_grid = np.zeros((rows, cols), dtype=int)
        
        # Mark object cells
        for r, c in object_coords:
            temp_grid[r, c] = 1
        
        # Use flood fill from the outside to mark all reachable empty cells
        padded = np.pad(temp_grid, 1, mode='constant', constant_values=0)
        outside_visited = np.zeros_like(padded, dtype=bool)
        
        # Start from the boundaries
        outside_visited[0, :] = True
        outside_visited[-1, :] = True
        outside_visited[:, 0] = True
        outside_visited[:, -1] = True
        
        # Queue for BFS
        queue = [(r, c) for r in range(padded.shape[0]) for c in range(padded.shape[1]) 
                if outside_visited[r, c]]
        
        # BFS to mark all reachable empty cells
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-way connectivity
                nr, nc = r + dr, c + dc
                if (0 <= nr < padded.shape[0] and 0 <= nc < padded.shape[1] and
                    not outside_visited[nr, nc] and padded[nr, nc] == 0):
                    outside_visited[nr, nc] = True
                    queue.append((nr, nc))
        
        # Remove padding from outside_visited
        outside_visited = outside_visited[1:-1, 1:-1]
        
        # Find cells that are empty in the original grid and not reachable from outside
        # These are the enclosed regions
        enclosed_mask = (grid == 0) & ~outside_visited
        
        # Label connected components to count distinct regions
        labeled, num_regions = label(enclosed_mask, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        
        return num_regions
    
    def create_input(self, taskvars, gridvars):
        # Determine grid size - make it large enough to fit multiple objects
        # Ensure minimum size based on requirements
        if gridvars.get('mandatory', False):
            # For mandatory example with 1, 2, and 3 regions, we need a larger grid
            rows = random.randint(20, 30)
            cols = random.randint(20, 30)
        else:
            rows = random.randint(15, 30)
            cols = random.randint(15, 30)
            
        grid = np.zeros((rows, cols), dtype=int)
        
        # Determine object color
        color = taskvars['object_color']
        
        # Determine objects to create
        if gridvars.get('mandatory', False):
            # Create exactly one object for each of 1, 2, and 3 regions
            region_counts = [1, 2, 3]
            num_objects = 3
        else:
            # Random number of objects with 1-4 regions
            num_objects = random.randint(2, 5)
            region_counts = []
            
            # Ensure variety in region counts
            for _ in range(num_objects):
                region_count = random.randint(1, min(4, (rows*cols) // 100))  # Limit based on grid size
                region_counts.append(region_count)
        
        # Randomize the order of objects (for diversity)
        random.shuffle(region_counts)
        
        # Keep track of used positions
        used_areas = []  # List of (row, col, width, height) for occupied areas
        
        # Create each object and place it
        for region_count in region_counts:
            # Create a temporary grid for this object
            temp_grid = np.zeros_like(grid)
            
            # Create the frame with the specified number of regions
            frame_areas = self.create_frame_with_regions(temp_grid, region_count)
            
            if not frame_areas:
                continue  # Skip if we couldn't create a valid frame
                
            # Get the main frame area
            main_area = frame_areas[0]
            frame_row, frame_col, frame_width, frame_height = main_area
            
            # Try to find a spot for this object
            placed = False
            max_attempts = 30
            
            for attempt in range(max_attempts):
                # Bounds check for placing the object
                if rows <= frame_height + 2 or cols <= frame_width + 2:
                    break  # Grid too small for this object
                
                # Randomly position this object within the grid (with safety margins)
                new_row = random.randint(1, max(1, rows - frame_height - 1))
                new_col = random.randint(1, max(1, cols - frame_width - 1))
                
                # Check if this position overlaps with any existing object
                overlap = False
                padding = 2  # Minimum space between objects
                
                for used_row, used_col, used_width, used_height in used_areas:
                    if (new_row - padding < used_row + used_height and 
                        new_row + frame_height + padding > used_row and
                        new_col - padding < used_col + used_width and
                        new_col + frame_width + padding > used_col):
                        overlap = True
                        break
                
                if not overlap:
                    # Place the object by adding all cells from the temp grid
                    # but with the correct offset and color
                    row_offset = new_row - frame_row
                    col_offset = new_col - frame_col
                    
                    for r in range(rows):
                        for c in range(cols):
                            if (0 <= r < temp_grid.shape[0] and 
                                0 <= c < temp_grid.shape[1] and 
                                temp_grid[r, c] > 0):
                                new_r = r + row_offset
                                new_c = c + col_offset
                                if 0 <= new_r < rows and 0 <= new_c < cols:
                                    grid[new_r, new_c] = color
                    
                    # Mark this area as used
                    used_areas.append((new_row, new_col, frame_width, frame_height))
                    placed = True
                    break
            
            # If we couldn't place this object after several attempts, try with a simpler one
            if not placed and len(used_areas) < 1:
                # Try with a simple frame (1 region) in any available space
                temp_grid = np.zeros_like(grid)
                frame_width = min(5, (cols - 2) // 2)
                frame_height = min(5, (rows - 2) // 2)
                
                # Ensure minimum sizes
                frame_width = max(frame_width, 3)
                frame_height = max(frame_height, 3)
                
                for attempt in range(max_attempts):
                    # Safety bounds check
                    new_row = random.randint(1, max(1, rows - frame_height - 1))
                    new_col = random.randint(1, max(1, cols - frame_width - 1))
                    
                    # Check for overlap
                    overlap = False
                    for used_row, used_col, used_width, used_height in used_areas:
                        if (new_row < used_row + used_height + 1 and 
                            new_row + frame_height + 1 > used_row and
                            new_col < used_col + used_width + 1 and
                            new_col + frame_width + 1 > used_col):
                            overlap = True
                            break
                    
                    if not overlap:
                        self.create_rectangular_frame(grid, new_row, new_col, frame_width, frame_height, color)
                        used_areas.append((new_row, new_col, frame_width, frame_height))
                        break
        
        # Verify we have at least one object in our grid
        if not used_areas:
            # Fallback: Create at least a simple frame in the center
            frame_width = min(6, cols - 2)
            frame_height = min(6, rows - 2)
            
            # Ensure minimum sizes
            frame_width = max(frame_width, 3)
            frame_height = max(frame_height, 3)
            
            # Safety bounds check
            row = max(1, (rows - frame_height) // 2)
            col = max(1, (cols - frame_width) // 2)
            
            # Final bounds check
            if row + frame_height < rows and col + frame_width < cols:
                self.create_rectangular_frame(grid, row, col, frame_width, frame_height, color)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        object_color = taskvars['object_color']
        change_color1 = taskvars['change_color1']
        change_color2 = taskvars['change_color2']
        change_color3 = taskvars['change_color3']
        change_color4 = taskvars['change_color4']
        
        # Find all connected objects of object_color
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Filter objects of the target color
        color_objects = objects.filter(lambda obj: object_color in obj.colors)
        
        # Process each object
        for obj in color_objects:
            # Extract object coordinates
            obj_coords = [(r, c) for r, c, _ in obj.cells]
            
            # Count enclosed regions
            num_regions = self.count_enclosed_regions(grid, obj_coords)
            
            # Recolor the object based on the number of enclosed regions
            if num_regions == 1:
                new_color = change_color1
            elif num_regions == 2:
                new_color = change_color2
            elif num_regions == 3:
                new_color = change_color3
            elif num_regions == 4:
                new_color = change_color4
            else:
                # Keep original color if more than 4 regions or 0 regions (shouldn't happen)
                new_color = object_color
                
            # Apply the new color
            for r, c, _ in obj.cells:
                output_grid[r, c] = new_color
                
        return output_grid

