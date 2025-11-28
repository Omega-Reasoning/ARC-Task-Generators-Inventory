import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, retry
from scipy import ndimage

class ARCTask239be575Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "Two squares each of size {vars['subgrd']} x {vars['subgrd']} of color {color('in_color')} are in the input grid.",
            "Random cells are placed at positions which are not occupied by the above squares of color {color('cell_color')}.",
            "Either there exists a 4-connectivity path between the two squares using cells of color {color('cell_color')} or there is no path at all (not even 8-connectivity).",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "First identify the two squares in the input grid of color {color('in_color')}.",
            "If there is a 4-connectivity path connecting the two squares via the cells of color {color('cell_color')}, then the output grid is 1x1 of color {color('cell_color')} else it is 1x1 empty grid cell(0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def check_path_connectivity(self, grid, square1_coords, square2_coords, cell_color, in_color, connectivity):
        """
        Check if there's a path between two squares with given connectivity.
        connectivity: 4 or 8
        """
        # Create a mask of valid path cells
        path_mask = np.logical_or(grid == cell_color, grid == in_color)
        
        # Define directions based on connectivity
        if connectivity == 4:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:  # 8-connectivity
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                         (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        # Start BFS from first square
        for start_r, start_c in square1_coords:
            visited = set()
            to_visit = [(start_r, start_c)]
            
            while to_visit:
                r, c = to_visit.pop(0)
                if (r, c) in visited:
                    continue
                    
                visited.add((r, c))
                
                # If we reached any cell of the second square
                if (r, c) in square2_coords:
                    return True
                
                # Check neighbors based on connectivity
                for dr, dc in directions:
                    new_r, new_c = r + dr, c + dc
                    if (0 <= new_r < grid.shape[0] and 
                        0 <= new_c < grid.shape[1] and 
                        path_mask[new_r, new_c] and
                        (new_r, new_c) not in visited):
                        to_visit.append((new_r, new_c))
        
        return False

    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        subgrd = taskvars['subgrd']
        in_color = taskvars['in_color']
        cell_color = taskvars['cell_color']

        max_attempts = 100
        for attempt in range(max_attempts):
            try:
                grid = np.zeros((rows, cols), dtype=int)
                
                square_positions = []
                available_positions = [(r, c) for r in range(rows - subgrd) for c in range(cols - subgrd)]
                
                if len(available_positions) == 0:
                    continue  # Grid too small for this subgrid size
                
                # Place first square
                pos1 = random.choice(available_positions)
                square_positions.append(pos1)
                
                # Remove positions that would cause overlap or touching with first square
                r1, c1 = pos1
                available_positions = [(r, c) for r, c in available_positions 
                                     if not (r1-subgrd <= r <= r1 + subgrd and 
                                            c1-subgrd <= c <= c1 + subgrd)]
                
                if not available_positions:
                    continue  # Try again if no valid positions for second square
                    
                # Place second square
                pos2 = random.choice(available_positions)
                square_positions.append(pos2)
                
                for r, c in square_positions:
                    grid[r:r+subgrd, c:c+subgrd] = in_color
                
                # Decide randomly whether to create a path or no path
                should_have_path = random.choice([True, False])
                
                if should_have_path:
                    # Create a guaranteed 4-connectivity path
                    r1, c1 = square_positions[0]
                    r2, c2 = square_positions[1]
                    
                    # Start from center of first square
                    curr_r = r1 + subgrd // 2
                    curr_c = c1 + subgrd // 2
                    
                    # End at center of second square
                    end_r = r2 + subgrd // 2
                    end_c = c2 + subgrd // 2
                    
                    # Create a path (Manhattan-style)
                    while curr_r != end_r or curr_c != end_c:
                        if grid[curr_r, curr_c] == 0:
                            grid[curr_r, curr_c] = cell_color
                        
                        # Move towards target
                        if curr_r < end_r:
                            curr_r += 1
                        elif curr_r > end_r:
                            curr_r -= 1
                        elif curr_c < end_c:
                            curr_c += 1
                        elif curr_c > end_c:
                            curr_c -= 1
                    
                    # Place additional random cells
                    for _ in range(int(0.3 * rows * cols)):
                        r, c = random.randint(0, rows-1), random.randint(0, cols-1)
                        if grid[r, c] == 0:
                            grid[r, c] = cell_color
                else:
                    # Create no path - place random cells but ensure no path at all
                    # Place fewer random cells to reduce chance of accidental path
                    for _ in range(int(0.15 * rows * cols)):
                        r, c = random.randint(0, rows-1), random.randint(0, cols-1)
                        if grid[r, c] == 0:
                            grid[r, c] = cell_color
                
                # Get square coordinates
                square1_coords = set()
                square2_coords = set()
                r1, c1 = square_positions[0]
                r2, c2 = square_positions[1]
                
                for r in range(r1, r1 + subgrd):
                    for c in range(c1, c1 + subgrd):
                        square1_coords.add((r, c))
                
                for r in range(r2, r2 + subgrd):
                    for c in range(c2, c2 + subgrd):
                        square2_coords.add((r, c))
                
                # Verify the path connectivity matches our intention
                has_4_path = self.check_path_connectivity(grid, square1_coords, square2_coords, 
                                                         cell_color, in_color, 4)
                has_8_path = self.check_path_connectivity(grid, square1_coords, square2_coords, 
                                                         cell_color, in_color, 8)
                
                # Valid if: (has 4-path) OR (has no 8-path)
                # Invalid if: (no 4-path) AND (has 8-path) - only diagonal connections exist
                if has_4_path or not has_8_path:
                    return grid  # Success! Return the grid
                else:
                    continue  # Has only 8-connectivity path, try again
                
            except (ValueError, IndexError):
                continue  # Try again if any error occurs
        
        # Fallback: create a simple grid with no path
        grid = np.zeros((rows, cols), dtype=int)
        if rows >= subgrd and cols >= subgrd:
            grid[0:subgrd, 0:subgrd] = in_color
            if rows >= 2*subgrd and cols >= 2*subgrd:
                grid[rows-subgrd:rows, cols-subgrd:cols] = in_color
        return grid

    def transform_input(self, grid, taskvars):
        cell_color = taskvars['cell_color']
        
        # Find square objects
        def find_squares(grid):
            rows, cols = grid.shape
            squares = []
            visited = set()
            
            # Define the cross-shaped mask for 4-connectivity
            structure = np.array([[0,1,0],
                            [1,1,1],
                            [0,1,0]], dtype=bool)
            
            # Process each color separately
            unique_colors = np.unique(grid[grid != 0])  # Exclude 0 (empty cells)
            for color in unique_colors:
                # Create mask for current color
                color_mask = grid == color
                # Label connected components for this color
                labeled_array, num_features = ndimage.label(color_mask, structure=structure)
                # Process each labeled component of this color
                for label in range(1, num_features + 1):
                    component = (labeled_array == label)
                    if not any((r,c) in visited for r,c in zip(*np.where(component))):
                        # Get bounding box
                        rows_idx, cols_idx = np.where(component)
                        min_r, max_r = min(rows_idx), max(rows_idx)
                        min_c, max_c = min(cols_idx), max(cols_idx)
                        height = max_r - min_r + 1
                        width = max_c - min_c + 1
                        
                        # Check if it's a square
                        if height == width:
                            # Verify all cells in the region are the same color
                            region = grid[min_r:max_r+1, min_c:max_c+1]
                            if np.all(region == color):
                                square_coords = set((r,c) for r,c in zip(*np.where(component)))
                                squares.append({
                                    'color': color,
                                    'coords': square_coords,
                                    'size': width
                                })
                                visited.update(square_coords)
            
            return squares
        
        squares = find_squares(grid)
        # Find the two squares of the same color
        color_squares = {}
        for square in squares:
            color = square['color']
            if color not in color_squares:
                color_squares[color] = []
            color_squares[color].append(square)
        
        # Find the color that has exactly two squares
        target_squares = None
        for color, sq_list in color_squares.items():
            if len(sq_list) == 2:
                target_squares = sq_list
                break
        
        if target_squares:
            square1_coords = target_squares[0]['coords']
            square2_coords = target_squares[1]['coords']
            
            # Create a mask of valid path cells
            path_mask = np.logical_or(grid == cell_color, grid == target_squares[0]['color'])
            
            # Start from each cell in the first square
            for start_r, start_c in square1_coords:
                visited = set()
                to_visit = [(start_r, start_c)]
                while to_visit:
                    r, c = to_visit.pop(0)  # Use pop(0) for BFS
                    if (r, c) in visited:
                        continue
                        
                    visited.add((r, c))
                    
                    # If we reached any cell of the second square
                    if (r, c) in square2_coords:
                        return np.array([[cell_color]])
                    
                    # Check only 4 directions (4-connectivity)
                    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        new_r, new_c = r + dr, c + dc
                        if (0 <= new_r < grid.shape[0] and 
                            0 <= new_c < grid.shape[1] and 
                            path_mask[new_r, new_c] and
                            (new_r, new_c) not in visited):
                            to_visit.append((new_r, new_c))
                        
        return np.array([[0]])

    def create_grids(self):
        # First choose grid size
        rows = random.randint(10, 30)
        cols = random.randint(10, 30)
        
        # Then choose subgrid size that fits
        # We need space for 2 squares that don't touch, so:
        # - Each square needs subgrd x subgrd
        # - They need at least 1 cell gap between them
        # - Conservative estimate: we need at least 3*subgrd space in one dimension
        max_subgrd = min(rows, cols) // 3
        max_subgrd = max(2, min(max_subgrd, 5))  # Ensure it's between 2 and 5
        
        subgrd = random.randint(2, max_subgrd)
        
        taskvars = {
            'rows': rows,
            'cols': cols,
            'subgrd': subgrd,
            'in_color': random.choice(range(1, 10)),
            'cell_color': random.choice(range(1, 10))
        }
        while taskvars['cell_color'] == taskvars['in_color']:
            taskvars['cell_color'] = random.choice(range(1, 10))
        
        train_test_data = {
            'train': [
                {
                    'input': (input_grid := self.create_input(taskvars, {})),
                    'output': self.transform_input(input_grid, taskvars)
                }
                for _ in range(random.randint(5, 6))
            ],
            'test': [
                {
                    'input': (test_grid := self.create_input(taskvars, {})),
                    'output': self.transform_input(test_grid, taskvars)
                }
            ]
        }
        
        return taskvars, train_test_data