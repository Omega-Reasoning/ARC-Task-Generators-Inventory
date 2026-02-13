from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import create_object, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Set

class Task17829a00Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "Each input grid has a completely filled background with the {color('background_color')} color. The first row is filled with color a, and the last row is filled with color b.",
            "Within the grid, there are several objects colored with color a and color b.",
            "These objects are arranged in groups of columns.",
            "Each color a object is positioned above a color b object.",
            "There is at most one color a and one color b object in same group of columns",
            "Not all columns are required to contain color a or color b objects."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all objects colored with color a and color b.",
            "Move all color a objects to the first row, and all color b objects to the last row, such that each object becomes connected to their respective row.",
            "Do not change the shape or composition of the objects during this movement.",
            "After moving the objects, the original positions of the color a and color b objects are filled with the {color('background_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_object_shape(self, max_width: int, max_height: int, color: int) -> List[Tuple[int, int]]:
        """Create various object shapes with diversity"""
        # Ensure we have valid dimensions
        if max_width < 1 or max_height < 1:
            return [(0, 0)]
        
        shapes = [
            # Single cell
            lambda w, h: [(0, 0)],
            
            # Horizontal line
            lambda w, h: [(0, c) for c in range(min(w, random.randint(2, min(4, w))))],
            
            # Vertical line
            lambda w, h: [(r, 0) for r in range(min(h, random.randint(2, min(3, h))))],
            
            # Rectangle/square
            lambda w, h: [(r, c) for r in range(min(h, random.randint(1, min(2, h)))) 
                         for c in range(min(w, random.randint(1, min(3, w))))],
            
            # L-shape
            lambda w, h: [(0, 0), (0, 1), (1, 0)] if w >= 2 and h >= 2 else [(0, 0)],
            
            # T-shape
            lambda w, h: [(0, 0), (0, 1), (0, 2), (1, 1)] if w >= 3 and h >= 2 else [(0, 0), (0, 1)],
            
            # Cross/plus shape
            lambda w, h: [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)] if w >= 3 and h >= 3 else [(0, 0), (0, 1)],
            
            # Diagonal line
            lambda w, h: [(i, i) for i in range(min(w, h, 3))],
            
            # Corner shape
            lambda w, h: [(0, 0), (0, 1), (1, 0)] if w >= 2 and h >= 2 else [(0, 0)],
            
            # Random connected blob
            lambda w, h: self.create_random_blob(min(w, 3), min(h, 3)),
            
            # Zigzag pattern
            lambda w, h: [(0, 0), (0, 1), (1, 1), (1, 2)] if w >= 3 and h >= 2 else [(0, 0), (0, 1)],
        ]
        
        # Choose a random shape
        shape_func = random.choice(shapes)
        coords = shape_func(max_width, max_height)
        
        # Remove duplicates and ensure we have at least one cell
        coords = list(set(coords))
        if not coords:
            coords = [(0, 0)]
            
        return coords
    
    def create_random_blob(self, max_width: int, max_height: int) -> List[Tuple[int, int]]:
        """Create a random connected blob shape"""
        if max_width < 1 or max_height < 1:
            return [(0, 0)]
            
        # Start with center cell
        blob = {(0, 0)}
        
        # Calculate maximum possible expansions
        max_cells = max_width * max_height
        if max_cells <= 1:
            return [(0, 0)]
        
        # Safe range for expansion - ensure we don't exceed available cells
        max_expansions = min(6, max_cells - 1)
        if max_expansions <= 0:
            return [(0, 0)]
        
        # Add random neighbors to create connected blob
        num_expansions = random.randint(1, max_expansions)
        for _ in range(num_expansions):
            # Get all possible expansion points
            expansion_points = []
            for r, c in blob:
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    new_r, new_c = r + dr, c + dc
                    if (0 <= new_r < max_height and 0 <= new_c < max_width and 
                        (new_r, new_c) not in blob):
                        expansion_points.append((new_r, new_c))
            
            if expansion_points:
                blob.add(random.choice(expansion_points))
            else:
                break  # No more expansion possible
        
        return list(blob)
    
    def get_object_columns(self, coords: List[Tuple[int, int]]) -> Set[int]:
        """Get the set of columns that an object occupies"""
        return {c for r, c in coords}
    
    def check_column_conflicts(self, new_coords: List[Tuple[int, int]], 
                              existing_objects_coords: List[List[Tuple[int, int]]]) -> bool:
        """Check if new object shares columns with any existing objects of the same color"""
        new_columns = self.get_object_columns(new_coords)
        
        for existing_coords in existing_objects_coords:
            existing_columns = self.get_object_columns(existing_coords)
            if not new_columns.isdisjoint(existing_columns):  # They share at least one column
                return True  # Conflict found
        
        return False  # No conflicts
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        background_color = taskvars['background_color']
        color_a = taskvars['color_a']
        color_b = taskvars['color_b']
        
        # Create grid filled with background color
        grid = np.full((grid_size, grid_size), background_color, dtype=int)
        
        # Fill first row with color a and last row with color b
        grid[0, :] = color_a
        grid[grid_size - 1, :] = color_b
        
        # Determine minimum number of objects based on grid size
        if grid_size > 7:
            min_objects_each = 2
            max_objects_each = min(4, grid_size // 3)  # Reduced to allow for column separation
        else:
            min_objects_each = 1
            max_objects_each = min(2, grid_size // 3)
        
        # Ensure we have valid ranges
        if max_objects_each < min_objects_each:
            max_objects_each = min_objects_each
        
        num_color_a_objects = random.randint(min_objects_each, max_objects_each)
        num_color_b_objects = random.randint(min_objects_each, max_objects_each)
        
        # Track placed objects and their columns
        placed_color_a_objects = []  # List of coordinate lists
        placed_color_b_objects = []  # List of coordinate lists
        
        attempts = 0
        max_attempts = 300
        
        # Place color a objects first
        while len(placed_color_a_objects) < num_color_a_objects and attempts < max_attempts:
            attempts += 1
            
            # Available area for color a (not adjacent to first row, upper half)
            min_row_a = 2
            max_row_a = max(3, grid_size // 2)
            
            if max_row_a > min_row_a:
                if max_row_a - 1 < min_row_a:
                    continue
                    
                start_row = random.randint(min_row_a, max_row_a - 1)
                start_col = random.randint(0, grid_size - 1)
                
                # Create object shape with safe bounds
                max_width = max(1, min(3, grid_size - start_col))
                max_height = max(1, min(3, max_row_a - start_row))
                
                shape_coords = self.create_object_shape(max_width, max_height, color_a)
                
                # Convert to absolute coordinates
                test_coords = []
                valid_placement = True
                for dr, dc in shape_coords:
                    r, c = start_row + dr, start_col + dc
                    if 0 < r < grid_size - 1 and 0 <= c < grid_size:
                        test_coords.append((r, c))
                    else:
                        valid_placement = False
                        break
                
                if (valid_placement and test_coords and 
                    not self.check_column_conflicts(test_coords, placed_color_a_objects)):
                    
                    # Place the object
                    for r, c in test_coords:
                        grid[r, c] = color_a
                    placed_color_a_objects.append(test_coords)
        
        # Place color b objects, ensuring they can share columns with color a objects
        # but not with other color b objects
        attempts = 0
        while len(placed_color_b_objects) < num_color_b_objects and attempts < max_attempts:
            attempts += 1
            
            # Available area for color b (not adjacent to last row, lower half)
            min_row_b = max(grid_size // 2, 2)
            max_row_b = grid_size - 2
            
            if max_row_b > min_row_b:
                if max_row_b - 1 < min_row_b:
                    continue
                    
                start_row = random.randint(min_row_b, max_row_b - 1)
                start_col = random.randint(0, grid_size - 1)
                
                # Create object shape with safe bounds
                max_width = max(1, min(3, grid_size - start_col))
                max_height = max(1, min(3, max_row_b - start_row))
                
                shape_coords = self.create_object_shape(max_width, max_height, color_b)
                
                # Convert to absolute coordinates
                test_coords = []
                valid_placement = True
                for dr, dc in shape_coords:
                    r, c = start_row + dr, start_col + dc
                    if 0 < r < grid_size - 1 and 0 <= c < grid_size:
                        test_coords.append((r, c))
                    else:
                        valid_placement = False
                        break
                
                if (valid_placement and test_coords and 
                    not self.check_column_conflicts(test_coords, placed_color_b_objects)):
                    
                    # Place the object
                    for r, c in test_coords:
                        grid[r, c] = color_b
                    placed_color_b_objects.append(test_coords)
        
        return grid
    
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        grid_size = taskvars['grid_size']
        background_color = taskvars['background_color']
        color_a = taskvars['color_a']
        color_b = taskvars['color_b']
        
        # Find all objects in the grid using 8-way connectivity
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=background_color)
        
        # Separate color a and color b objects (excluding the filled rows)
        color_a_objects = []
        color_b_objects = []
        
        for obj in objects:
            # Skip objects that are part of the first or last row fill
            obj_coords = obj.coords
            is_first_row = any(r == 0 for r, c in obj_coords)
            is_last_row = any(r == grid_size - 1 for r, c in obj_coords)
            
            # Only consider objects that are not connected to the boundary rows
            if not is_first_row and not is_last_row:
                if obj.has_color(color_a):
                    color_a_objects.append(obj)
                elif obj.has_color(color_b):
                    color_b_objects.append(obj)
        
        # Remove original objects from their positions
        for obj in color_a_objects + color_b_objects:
            obj.cut(output_grid, background_color)
        
        # Move color a objects to connect with first row
        for obj in color_a_objects:
            # Find the topmost row of the object
            min_row = min(r for r, c, _ in obj.cells)
            # Calculate how much to move up to connect with first row (move to row 1)
            move_up = min_row - 1
            obj.translate(-move_up, 0)
            obj.paste(output_grid)
        
        # Move color b objects to connect with last row  
        for obj in color_b_objects:
            # Find the bottommost row of the object
            max_row = max(r for r, c, _ in obj.cells)
            # Calculate how much to move down to connect with last row (move so bottom touches row grid_size-2)
            move_down = (grid_size - 2) - max_row
            obj.translate(move_down, 0)
            obj.paste(output_grid)
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'grid_size': random.randint(8, 15),  # Increased minimum to allow for column separation
            'background_color': random.randint(1, 9)
        }
        
        # Choose two different colors for a and b, different from background
        available_colors = [c for c in range(1, 10) if c != taskvars['background_color']]
        colors_ab = random.sample(available_colors, 2)
        taskvars['color_a'] = colors_ab[0]
        taskvars['color_b'] = colors_ab[1]
        
        # Generate train and test examples with retry to ensure valid grids
        def create_valid_example():
            max_retries = 20
            for _ in range(max_retries):
                try:
                    input_grid = self.create_input(taskvars, {})
                    
                    # Verify that we have objects and check column constraints
                    objects = find_connected_objects(input_grid, diagonal_connectivity=True, 
                                                   background=taskvars['background_color'])
                    
                    color_a_objects = [obj for obj in objects if obj.has_color(taskvars['color_a']) 
                                     and not any(r in [0, taskvars['grid_size']-1] for r, c in obj.coords)]
                    color_b_objects = [obj for obj in objects if obj.has_color(taskvars['color_b']) 
                                     and not any(r in [0, taskvars['grid_size']-1] for r, c in obj.coords)]
                    
                    # Check column separation for same-color objects
                    def check_same_color_separation(objs):
                        for i, obj1 in enumerate(objs):
                            cols1 = {c for r, c in obj1.coords}
                            for j, obj2 in enumerate(objs[i+1:], i+1):
                                cols2 = {c for r, c in obj2.coords}
                                if not cols1.isdisjoint(cols2):  # They share columns
                                    return False
                        return True
                    
                    valid_separation = (check_same_color_separation(color_a_objects) and 
                                      check_same_color_separation(color_b_objects))
                    
                    min_required = 2 if taskvars['grid_size'] > 7 else 1
                    has_enough_objects = (len(color_a_objects) >= min_required and 
                                        len(color_b_objects) >= min_required)
                    
                    if valid_separation and has_enough_objects:
                        output_grid = self.transform_input(input_grid, taskvars)
                        return {'input': input_grid, 'output': output_grid}
                        
                except (ValueError, IndexError) as e:
                    # Skip this attempt if there's a range error or other issue
                    continue
            
            # Fallback if we can't create a valid example
            try:
                input_grid = self.create_input(taskvars, {})
                output_grid = self.transform_input(input_grid, taskvars)
                return {'input': input_grid, 'output': output_grid}
            except Exception:
                # Final fallback - create minimal valid grid
                grid = np.full((taskvars['grid_size'], taskvars['grid_size']), taskvars['background_color'], dtype=int)
                grid[0, :] = taskvars['color_a']
                grid[-1, :] = taskvars['color_b']
                # Add single objects
                if taskvars['grid_size'] > 4:
                    grid[2, 1] = taskvars['color_a']  # Simple color a object
                    grid[taskvars['grid_size']-3, 3] = taskvars['color_b']  # Simple color b object
                return {'input': grid, 'output': grid}
        
        # Generate train and test examples
        train_examples = [create_valid_example() for _ in range(3)]
        test_examples = [create_valid_example() for _ in range(1)]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

