from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import retry
import numpy as np
import random

class Task1478ab18Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}×{vars['grid_size']}.",
            "Each grid contains 4 {color('corner')} cells and a completely filled background of {color('background')} color.",
            "The 4 {color('corner')} cells are formed by first sketching the 4 corners of a square i.e is bigger than 2x2, then moving exactly one of the {color('corner')} corner cells one unit inward diagonally, so that it now occupies an interior position.",
            "The original corner cell is then filled with the {color('background')} color.",
            "The size of the sketched squares varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the 4 {color('corner')} cells. Out of these, 3 form the corners of a square that has {color('background')} interior cells and exactly one {color('corner')} cell.",
            "First, fill in the missing corner cell with the {color('fill')} color.",
            "Next, fill one vertical and one horizontal side of the square that are connected to the {color('fill')} cell using the {color('fill')} color.",
            "Then, identify the two {color('corner')} cells that are now connected by this {color('fill')} path.",
            "Connect the same two {color('corner')} cells again using an alternate path by diagonally filling all the cells between them with the {color('fill')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        corner_color = taskvars['corner']
        background_color = taskvars['background']
        
        # Create background grid
        grid = np.full((grid_size, grid_size), background_color, dtype=int)
        
        def generate_valid_grid():
            # Random square size (at least 3x3 to be bigger than 2x2)
            min_size = 4
            max_size = min(grid_size - 2, 12)  # Leave some margin
            square_size = random.randint(min_size, max_size)
            
            # Random position for the square (ensuring it fits)
            max_top = grid_size - square_size - 1
            max_left = grid_size - square_size - 1
            
            if max_top < 1 or max_left < 1:
                return None
                
            top = random.randint(1, max_top)
            left = random.randint(1, max_left)
            
            # Calculate the 4 corners of the square
            corners = [
                (top, left),                           # top-left
                (top, left + square_size - 1),         # top-right
                (top + square_size - 1, left),         # bottom-left
                (top + square_size - 1, left + square_size - 1)  # bottom-right
            ]
            
            # Choose one corner to move inward diagonally
            corner_to_move = random.randint(0, 3)
            
            # Calculate the inward diagonal direction for each corner
            inward_directions = [
                (1, 1),   # top-left moves down-right
                (1, -1),  # top-right moves down-left
                (-1, 1),  # bottom-left moves up-right
                (-1, -1)  # bottom-right moves up-left
            ]
            
            moved_corner = corners[corner_to_move]
            dr, dc = inward_directions[corner_to_move]
            new_pos = (moved_corner[0] + dr, moved_corner[1] + dc)
            
            # Create the grid with corner cells
            temp_grid = np.full((grid_size, grid_size), background_color, dtype=int)
            
            # Place the 3 unmoved corners
            for i, (r, c) in enumerate(corners):
                if i != corner_to_move:
                    temp_grid[r, c] = corner_color
            
            # Place the moved corner at its new position
            temp_grid[new_pos[0], new_pos[1]] = corner_color
            
            return temp_grid
        
        # Generate until we get a valid grid
        result = retry(generate_valid_grid, lambda x: x is not None, max_attempts=100)
        return result
    
    def _identify_corner_cells_without_opposite(self, grid, corner_color):
        """
        First function: Identify two corner cells that do not have a corner cell on the other end.
        Returns the two corner cells that will be connected by the L-shape and diagonal paths.
        """
        # Find all corner cells
        corner_cells = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == corner_color:
                    corner_cells.append((r, c))
        
        # Find bounding box
        rows = [r for r, c in corner_cells]
        cols = [c for r, c in corner_cells]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        # Expected corners of the complete square
        expected_corners = [
            (min_r, min_c),     # top-left
            (min_r, max_c),     # top-right
            (max_r, min_c),     # bottom-left
            (max_r, max_c)      # bottom-right
        ]
        
        # Find the missing corner
        missing_corner = None
        for corner in expected_corners:
            if corner not in corner_cells:
                missing_corner = corner
                break
        
        # Find the two corner cells that will be connected
        # These are the corners that share a row or column with the missing corner
        if missing_corner:
            r, c = missing_corner
            connected_corners = []
            for corner_pos in corner_cells:
                cr, cc = corner_pos
                if (cr == r and min_c <= cc <= max_c) or (cc == c and min_r <= cr <= max_r):
                    connected_corners.append(corner_pos)
            
            return connected_corners, missing_corner, (min_r, max_r, min_c, max_c)
        
        return [], None, (min_r, max_r, min_c, max_c)
    
    def _make_l_shape_path(self, output_grid, connected_corners, missing_corner, bounds, corner_color, fill_color):
        """
        Second function: Make L-shape path between the two corner cells (horizontal and vertical).
        """
        if len(connected_corners) != 2 or missing_corner is None:
            return
        
        min_r, max_r, min_c, max_c = bounds
        r, c = missing_corner
        
        # Fill the missing corner first
        output_grid[r, c] = fill_color
        
        # Create set of corner cells to avoid overwriting them
        corner_cells_set = set()
        for row in range(output_grid.shape[0]):
            for col in range(output_grid.shape[1]):
                if output_grid[row, col] == corner_color:
                    corner_cells_set.add((row, col))
        
        # Fill horizontal side (from missing corner to adjacent corner)
        corner_on_same_row = None
        for corner_pos in connected_corners:
            cr, cc = corner_pos
            if cr == r:
                corner_on_same_row = corner_pos
                break
        
        if corner_on_same_row:
            start_c = min(min_c, max_c)
            end_c = max(min_c, max_c)
            for col in range(start_c, end_c + 1):
                if (r, col) not in corner_cells_set:
                    output_grid[r, col] = fill_color
        
        # Fill vertical side (from missing corner to adjacent corner)
        corner_on_same_col = None
        for corner_pos in connected_corners:
            cr, cc = corner_pos
            if cc == c:
                corner_on_same_col = corner_pos
                break
        
        if corner_on_same_col:
            start_r = min(min_r, max_r)
            end_r = max(min_r, max_r)
            for row in range(start_r, end_r + 1):
                if (row, c) not in corner_cells_set:
                    output_grid[row, c] = fill_color
    
    def _make_diagonal_path(self, output_grid, connected_corners, corner_color, fill_color):
        """
        Third function: Make diagonal path between the two corner cells.
        """
        if len(connected_corners) != 2:
            return
        
        # Create set of corner cells to avoid overwriting them
        corner_cells_set = set()
        for row in range(output_grid.shape[0]):
            for col in range(output_grid.shape[1]):
                if output_grid[row, col] == corner_color:
                    corner_cells_set.add((row, col))
        
        # Fill diagonal path between the two connected corners
        (r1, c1), (r2, c2) = connected_corners
        steps = max(abs(r2 - r1), abs(c2 - c1))
        
        if steps > 0:
            for i in range(steps + 1):
                dr = r1 + i * (r2 - r1) // steps
                dc = c1 + i * (c2 - c1) // steps
                if (dr, dc) not in corner_cells_set:
                    output_grid[dr, dc] = fill_color
    
    def transform_input(self, grid, taskvars):
        corner_color = taskvars['corner']
        background_color = taskvars['background']
        fill_color = taskvars['fill']
        
        # Copy the input grid
        output_grid = grid.copy()
        
        # Step 1: Identify two corner cells that do not have a corner cell on the other end
        connected_corners, missing_corner, bounds = self._identify_corner_cells_without_opposite(grid, corner_color)
        
        # Step 2: Make L-shape path between them (horizontal and vertical)
        self._make_l_shape_path(output_grid, connected_corners, missing_corner, bounds, corner_color, fill_color)
        
        # Step 3: Make diagonal path between them
        self._make_diagonal_path(output_grid, connected_corners, corner_color, fill_color)
        
        return output_grid
    
    def create_grids(self):
        # Generate task variables
        taskvars = {
            'grid_size': random.randint(8, 30),
            'corner': random.randint(1, 9),
            'background': random.randint(1, 9),
            'fill': random.randint(1, 9)
        }
        
        # Ensure all colors are different
        while taskvars['corner'] == taskvars['background']:
            taskvars['background'] = random.randint(1, 9)
        
        while taskvars['fill'] == taskvars['corner'] or taskvars['fill'] == taskvars['background']:
            taskvars['fill'] = random.randint(1, 9)
        
        # Create train and test data
        train_data = []
        for _ in range(3):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Create test data
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}

