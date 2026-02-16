from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import create_object, Contiguity, random_cell_coloring

class Taskaedd82e4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have varying sizes.",
            "The grid consists of multiple patterns in {color('pattern_color')} color and scattered single cells in the same color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid.",
            "The patterns remain in {color('pattern_color')} color",
            "Only the scattered single cells change to {color('cell_color')} color",
            "Individual cells that are connected diagonally are also identified and colored with {color('cell_color')}",
            "Within patterns, individual cells that act as diagonal attachments or extensions are also identified and colored with {color('cell_color')}"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_solid_pattern(self, height: int, width: int, pattern_color: int) -> np.ndarray:
        """Create a solid, obvious pattern with strong orthogonal connectivity."""
        pattern = np.zeros((height, width), dtype=int)
        
        # Create different types of substantial patterns
        pattern_type = random.choice(['solid_rect', 'L_shape', 'T_shape', 'plus', 'thick_line', 'corner', 'zigzag'])
        
        if pattern_type == 'solid_rect':
            # Solid rectangle - always fill completely
            pattern[:, :] = pattern_color
            
        elif pattern_type == 'L_shape':
            # L-shaped pattern
            if height >= 2 and width >= 2:
                pattern[0, :] = pattern_color  # Top row
                pattern[:, 0] = pattern_color  # Left column
            else:
                pattern[:, :] = pattern_color
                
        elif pattern_type == 'T_shape':
            # T-shaped pattern
            if height >= 2 and width >= 3:
                pattern[0, :] = pattern_color  # Top horizontal bar
                mid_col = width // 2
                pattern[:, mid_col] = pattern_color  # Vertical stem
            elif height >= 3 and width >= 2:
                # Rotated T
                pattern[:, 0] = pattern_color  # Left vertical bar
                mid_row = height // 2
                pattern[mid_row, :] = pattern_color  # Horizontal stem
            else:
                pattern[:, :] = pattern_color
                
        elif pattern_type == 'plus':
            # Plus/cross pattern
            if height >= 3 and width >= 3:
                mid_row = height // 2
                mid_col = width // 2
                pattern[mid_row, :] = pattern_color  # Horizontal line
                pattern[:, mid_col] = pattern_color  # Vertical line
            else:
                pattern[:, :] = pattern_color
                
        elif pattern_type == 'thick_line':
            # Thick line pattern
            if width > height:
                # Horizontal thick line
                for r in range(min(2, height)):
                    pattern[r, :] = pattern_color
            else:
                # Vertical thick line
                for c in range(min(2, width)):
                    pattern[:, c] = pattern_color
                    
        elif pattern_type == 'corner':
            # Corner pattern (reverse L)
            if height >= 2 and width >= 2:
                pattern[-1, :] = pattern_color  # Bottom row
                pattern[:, -1] = pattern_color  # Right column
            else:
                pattern[:, :] = pattern_color
                
        else:  # zigzag
            # Zigzag pattern
            if height >= 2 and width >= 3:
                pattern[0, 0] = pattern_color
                pattern[0, 1] = pattern_color
                pattern[1, 1] = pattern_color
                pattern[1, 2] = pattern_color
                if width > 3:
                    pattern[0, 3] = pattern_color
            else:
                pattern[:, :] = pattern_color
        
        return pattern

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a grid with multiple obvious solid patterns and scattered cells"""
        pattern_color = taskvars['pattern_color']
        grid_height = gridvars['grid_height']
        grid_width = gridvars['grid_width']
        num_patterns = gridvars.get('num_patterns', 2)  # Default to 2 patterns
            
        max_attempts = 25
        for attempt in range(max_attempts):
            # Initialize empty grid
            grid = np.zeros((grid_height, grid_width), dtype=int)
            
            # Keep track of occupied areas to avoid overlaps
            occupied = np.zeros_like(grid, dtype=bool)
            patterns_placed = 0
            
            # Create multiple patterns
            for pattern_idx in range(num_patterns):
                # Pattern sizes - scale with grid size
                max_pattern_height = min(max(3, grid_height // 4), grid_height - 1)
                max_pattern_width = min(max(3, grid_width // 4), grid_width - 1)
                
                # Ensure patterns are at least 2x2 to be substantial
                pattern_height = random.randint(2, max_pattern_height)
                pattern_width = random.randint(2, max_pattern_width)
                
                # Create the pattern
                pattern = self.create_solid_pattern(pattern_height, pattern_width, pattern_color)
                
                # Try to place the pattern
                max_placement_attempts = 40
                placed = False
                
                for placement_attempt in range(max_placement_attempts):
                    # Calculate valid placement positions
                    max_pos_r = grid_height - pattern_height
                    max_pos_c = grid_width - pattern_width
                    
                    if max_pos_r < 0 or max_pos_c < 0:
                        break  # Pattern too big
                    
                    pos_r = random.randint(0, max_pos_r)
                    pos_c = random.randint(0, max_pos_c)
                    
                    # Check for overlap with existing patterns (with 1-cell margin)
                    check_r_start = max(0, pos_r - 1)
                    check_r_end = min(grid_height, pos_r + pattern_height + 1)
                    check_c_start = max(0, pos_c - 1)
                    check_c_end = min(grid_width, pos_c + pattern_width + 1)
                    
                    if not np.any(occupied[check_r_start:check_r_end, check_c_start:check_c_end]):
                        # Place the pattern
                        for r in range(pattern_height):
                            for c in range(pattern_width):
                                if pattern[r, c] == pattern_color:
                                    grid[pos_r + r, pos_c + c] = pattern_color
                                    occupied[pos_r + r, pos_c + c] = True
                        placed = True
                        patterns_placed += 1
                        break
                
                if not placed and patterns_placed == 0:
                    # Force place at least one pattern
                    # Find the largest empty space
                    best_pos = None
                    best_size = 0
                    
                    for r in range(grid_height - 1):
                        for c in range(grid_width - 1):
                            if not occupied[r, c]:
                                # Try different sizes
                                max_h = min(max(4, grid_height // 3), grid_height - r)
                                max_w = min(max(4, grid_width // 3), grid_width - c)
                                for h in range(2, max_h + 1):
                                    for w in range(2, max_w + 1):
                                        if not np.any(occupied[r:r+h, c:c+w]):
                                            size = h * w
                                            if size > best_size:
                                                best_size = size
                                                best_pos = (r, c, h, w)
                    
                    if best_pos:
                        r, c, h, w = best_pos
                        grid[r:r+h, c:c+w] = pattern_color
                        occupied[r:r+h, c:c+w] = True
                        patterns_placed += 1
            
            # Add diagonal extensions occasionally
            if random.random() < 0.3:
                # Find positions diagonally adjacent to patterns
                extension_positions = []
                for r in range(grid_height):
                    for c in range(grid_width):
                        if grid[r, c] == 0:
                            has_diagonal = False
                            has_orthogonal = False
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = r + dr, c + dc
                                    if (0 <= nr < grid_height and 0 <= nc < grid_width and 
                                        grid[nr, nc] == pattern_color):
                                        if abs(dr) + abs(dc) == 1:
                                            has_orthogonal = True
                                        else:
                                            has_diagonal = True
                            
                            if has_diagonal and not has_orthogonal:
                                extension_positions.append((r, c))
                
                # Add 1-2 diagonal extensions
                if extension_positions:
                    num_extensions = min(2, len(extension_positions))
                    selected = random.sample(extension_positions, num_extensions)
                    for r, c in selected:
                        grid[r, c] = pattern_color
            
            # Add scattered individual cells - scale with grid size
            individual_cells_added = 0
            target_individual_cells = max(2, min(6, (grid_height * grid_width) // 15))  # Scale with grid area
            
            # Find empty positions
            empty_positions = [(r, c) for r in range(grid_height) for c in range(grid_width) if grid[r, c] == 0]
            
            # Add individual cells with some spacing
            attempts = 0
            while individual_cells_added < target_individual_cells and attempts < 50 and empty_positions:
                attempts += 1
                r, c = random.choice(empty_positions)
                empty_positions.remove((r, c))
                
                # Check if it's reasonably isolated - scale spacing with grid size
                too_close = False
                min_distance = max(1, min(2, grid_height // 8))  # Scale with grid size
                
                for dr in range(-min_distance, min_distance + 1):
                    for dc in range(-min_distance, min_distance + 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < grid_height and 0 <= nc < grid_width and 
                            grid[nr, nc] == pattern_color):
                            too_close = True
                            break
                    if too_close:
                        break
                
                if not too_close or individual_cells_added == 0:  # Force at least one
                    grid[r, c] = pattern_color
                    individual_cells_added += 1
            
            # Verify we have both patterns and individual cells
            all_objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
            individual_cells = [obj for obj in all_objects if len(obj) == 1]
            patterns = [obj for obj in all_objects if len(obj) >= 3]
            
            
            
            if len(individual_cells) > 0 and len(patterns) > 0:
                return grid
        
        # Enhanced fallback
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Create patterns that scale with grid size
        pattern_size = max(2, min(4, grid_height // 3, grid_width // 3))
        
        if grid_height >= pattern_size and grid_width >= pattern_size:
            # First pattern: L-shape in top-left
            grid[0:pattern_size, 0:2] = pattern_color
            grid[0:2, 0:pattern_size] = pattern_color
            
            # Second pattern in bottom-right if space allows
            if grid_height >= pattern_size * 2 and grid_width >= pattern_size * 2:
                end_r = grid_height - pattern_size
                end_c = grid_width - pattern_size
                grid[end_r:end_r+pattern_size, end_c:end_c+pattern_size] = pattern_color
            
        elif grid_height >= 2 and grid_width >= 3:
            # Two small patterns
            grid[0, 0:2] = pattern_color  # Top line
            grid[-1, -2:] = pattern_color  # Bottom line
            
        else:
            # Minimal patterns
            grid[0, 0] = pattern_color
            if grid.size > 1:
                grid[-1, -1] = pattern_color
        
        # Add individual cells - scale with grid size
        empty_positions = [(r, c) for r in range(grid_height) for c in range(grid_width) if grid[r, c] == 0]
        if empty_positions:
            num_individual = min(max(2, grid_height * grid_width // 20), len(empty_positions))
            selected = random.sample(empty_positions, num_individual)
            for r, c in selected:
                grid[r, c] = pattern_color
        
        return grid
    
    def is_individual_cell_group(self, obj: GridObject, grid: np.ndarray) -> bool:
        """Check if an object is an individual cell group."""
        if len(obj) == 1:
            return True
        
        if len(obj) > 2:
            return False
            
        # Check if cells are only connected diagonally
        cells = list(obj.cells)
        for i, (r1, c1, _) in enumerate(cells):
            for j, (r2, c2, _) in enumerate(cells):
                if i >= j:
                    continue
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    return False
        
        return True
    
    def find_diagonal_attachments_in_object(self, obj: GridObject, grid: np.ndarray) -> set:
        """Find cells within an object that are only diagonally connected."""
        if len(obj) <= 3:
            return set()
        
        diagonal_attachments = set()
        cells = list(obj.cells)
        
        for r, c, color in cells:
            orthogonal_connections = 0
            diagonal_connections = 0
            
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    
                    if (nr, nc, color) in obj.cells:
                        if abs(dr) + abs(dc) == 1:
                            orthogonal_connections += 1
                        else:
                            diagonal_connections += 1
            
            if orthogonal_connections == 0 and diagonal_connections > 0:
                diagonal_attachments.add((r, c))
        
        return diagonal_attachments
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input grid."""
        pattern_color = taskvars['pattern_color']
        cell_color = taskvars['cell_color']
        
        output_grid = grid.copy()
        all_objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        for obj in all_objects:
            if self.is_individual_cell_group(obj, grid):
                for r, c, _ in obj.cells:
                    output_grid[r, c] = cell_color
            else:
                diagonal_attachments = self.find_diagonal_attachments_in_object(obj, grid)
                for r, c in diagonal_attachments:
                    output_grid[r, c] = cell_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        pattern_color = random.randint(1, 9)
        cell_color = random.choice([c for c in range(1, 10) if c != pattern_color])
        
        taskvars = {
            'pattern_color': pattern_color,
            'cell_color': cell_color,
        }
        
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Create more diverse grid dimensions - expanded range
        grid_dimensions = []
        for _ in range(num_train_examples + 1):
            # Allow full range from 5 to 30 for more variety
            height = random.randint(5, 30)
            width = random.randint(5, 30)
            grid_dimensions.append((height, width))
        
        for i in range(num_train_examples):
            height, width = grid_dimensions[i]
            # Scale number of patterns with grid size
            num_patterns = max(2, min(6, (height * width) // 50))
            gridvars = {
                'grid_height': height,
                'grid_width': width,
                'num_patterns': num_patterns,
            }
            
            #print(f"Creating training grid {i+1}: {height}x{width}")
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Test example
        test_height, test_width = grid_dimensions[-1]
        test_num_patterns = max(2, min(6, (test_height * test_width) // 50))
        test_gridvars = {
            'grid_height': test_height,
            'grid_width': test_width,
            'num_patterns': test_num_patterns,
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