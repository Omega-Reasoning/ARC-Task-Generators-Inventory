from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Set

class task15663ba9Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each input contains 2-3 single-colored objects with the same color, with all remaining cells being empty (0).",
            "Objects are completely separated from each other (minimum 1 cell gap).",
            "Each object forms a closed boundary that encloses a region of empty cells.",
            "Both the interior and exterior of each object must be completely empty (0); only the boundary should be colored.",
            "Each grid has at least one irregular polygon shape object.",
            "The object color must vary across examples.",
            "The boundary is composed of vertical and horizontal lines of varying lengths, connected in a 4-way manner (up, down, left, right).",
            "Each boundary cell must be connected to exactly two other boundary cells via 4-way connectivity."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all single-colored boundaries that enclose empty (0) regions.",
            "For each boundary object, locate all the corner cells of that boundary.",
            "Recolor all corner cells lying on the exterior (outer side) of each boundary to {color('exterior_border')}.",
            "Recolor all corner cells lying on the interior side of each boundary to {color('interior_border')}.",
            "All other boundary cells remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def _generate_orthogonal_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate a path using only horizontal and vertical moves between two points."""
        r1, c1 = start
        r2, c2 = end
        path = [start]
        
        # Randomly choose whether to move horizontally or vertically first
        if random.choice([True, False]):
            # Horizontal first
            step = 1 if c1 < c2 else -1
            for c in range(c1 + step, c2 + step, step):
                path.append((r1, c))
            # Then vertical
            step = 1 if r1 < r2 else -1
            for r in range(r1 + step, r2 + step, step):
                path.append((r, c2))
        else:
            # Vertical first
            step = 1 if r1 < r2 else -1
            for r in range(r1 + step, r2 + step, step):
                path.append((r, c1))
            # Then horizontal
            step = 1 if c1 < c2 else -1
            for c in range(c1 + step, c2 + step, step):
                path.append((r2, c))
        
        return path

    def _add_random_bump(self, boundary_path: List[Tuple[int, int]], grid_size: int):
        """Add a random bump to the boundary path while maintaining connectivity."""
        if len(boundary_path) < 4:
            return
        
        # Choose a random segment to add a bump to
        segment_start = random.randint(1, len(boundary_path) - 3)
        p1 = boundary_path[segment_start]
        p2 = boundary_path[segment_start + 1]
        
        # Determine direction of the segment
        dr = p2[0] - p1[0]
        dc = p2[1] - p1[1]
        
        # Calculate perpendicular direction for the bump
        if dr == 0:  # Horizontal segment
            bump_dr = random.choice([-1, 1])
            bump_dc = 0
        elif dc == 0:  # Vertical segment
            bump_dr = 0
            bump_dc = random.choice([-1, 1])
        else:
            return  # Skip diagonal segments
        
        # Create bump point
        bump_r = p1[0] + bump_dr
        bump_c = p1[1] + bump_dc
        
        # Check if bump point is within bounds and not already in path
        if (0 <= bump_r < grid_size and 0 <= bump_c < grid_size and
            (bump_r, bump_c) not in boundary_path):
            boundary_path.insert(segment_start + 1, (bump_r, bump_c))

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create input grid with 2-3 separated closed boundary objects, at least one irregular."""
        grid_size = taskvars['grid_size']
        boundary_color = gridvars.get('boundary_color', random.randint(1, 9))
        
        def generate_multiple_boundaries():
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Decide number of objects (ensure some have 3)
            num_objects = 3 if random.random() < 0.4 else 2  # 40% chance of 3 objects
            
            has_irregular = False
            placed_objects = []
            
            for obj_idx in range(num_objects):
                for attempt in range(50):  # More attempts per object
                    temp_grid = grid.copy()
                    
                    # Ensure at least one irregular: force first object to be irregular, others random
                    is_irregular = False
                    if obj_idx == 0:  # First object MUST be irregular
                        boundary_path = self._generate_irregular_polygon_path(grid_size)
                        is_irregular = True
                    elif not has_irregular:  # If we don't have irregular yet, force it
                        boundary_path = self._generate_irregular_polygon_path(grid_size)
                        is_irregular = True
                    else:  # We already have irregular, so this can be either
                        if random.random() < 0.6:  # 60% chance for irregular to have variety
                            boundary_path = self._generate_irregular_polygon_path(grid_size)
                            is_irregular = True
                        else:
                            boundary_path = self._generate_rectangular_path(grid_size)
                            is_irregular = False
                    
                    if not boundary_path:
                        continue
                    
                    # Check minimum distance from existing objects
                    min_distance = 1  # At least 1 cell separation
                    too_close = False
                    for (r, c) in boundary_path:
                        # Check 3x3 area around each point
                        for dr in range(-min_distance, min_distance+1):
                            for dc in range(-min_distance, min_distance+1):
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < grid_size and 0 <= nc < grid_size and 
                                    temp_grid[nr, nc] != 0):
                                    too_close = True
                                    break
                            if too_close:
                                break
                        if too_close:
                            break
                    
                    if too_close:
                        continue
                    
                    # Draw the boundary
                    for r, c in boundary_path:
                        if 0 <= r < grid_size and 0 <= c < grid_size:
                            temp_grid[r, c] = boundary_color
                    
                    # Validate the boundary
                    if self._is_valid_single_boundary(temp_grid, boundary_path, boundary_color):
                        grid = temp_grid
                        placed_objects.append((boundary_path, is_irregular))
                        if is_irregular:
                            has_irregular = True
                        break
            
            # Ensure we have the required number of objects and at least one irregular
            if len(placed_objects) >= 2 and has_irregular:
                # Extract just the boundary paths for separation verification
                object_paths = [obj[0] for obj in placed_objects]
                # Ensure objects are separated by at least 1 cell
                if self._verify_minimum_separation(grid, object_paths, min_distance=1):
                    return grid
            
            return np.zeros((grid_size, grid_size), dtype=int)
        
        def is_valid_multiple_boundaries(grid):
            objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
            if len(objects) < 2 or len(objects) > 3:
                return False
            
            has_irregular = False
            for obj in objects:
                coords = obj.coords
                if len(coords) < 4:  # Minimum size
                    return False
                
                # Check boundary connectivity
                valid_cells = 0
                for r, c in coords:
                    neighbors = 0
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if (nr, nc) in coords:
                            neighbors += 1
                    if neighbors == 2:
                        valid_cells += 1
                
                if valid_cells < len(coords) * 0.7:  # 70% should have 2 neighbors
                    return False
                
                # Check if object is irregular
                if not self._is_rectangular(coords):
                    has_irregular = True
            
            return has_irregular
        
        return retry(generate_multiple_boundaries, is_valid_multiple_boundaries, max_attempts=300)

    def _verify_minimum_separation(self, grid: np.ndarray, objects: List[List[Tuple[int, int]]], min_distance: int = 1) -> bool:
        """Verify all objects are separated by at least min_distance cells."""
        occupied = np.zeros_like(grid, dtype=bool)
        
        # Mark occupied cells and their neighbors
        for obj in objects:
            for r, c in obj:
                for dr in range(-min_distance, min_distance+1):
                    for dc in range(-min_distance, min_distance+1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                            occupied[nr, nc] = True
        
        # Check that only one object exists in each connected component
        visited = np.zeros_like(grid, dtype=bool)
        components = 0
        
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 0 and not visited[r, c]:
                    components += 1
                    stack = [(r, c)]
                    visited[r, c] = True
                    
                    while stack:
                        curr_r, curr_c = stack.pop()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and 
                                grid[nr, nc] != 0 and not visited[nr, nc]):
                                visited[nr, nc] = True
                                stack.append((nr, nc))
        
        return components == len(objects)

    def _generate_rectangular_path(self, grid_size: int) -> List[Tuple[int, int]]:
        """Generate a rectangular boundary path with optional bumps."""
        min_size = max(3, grid_size // 8)
        max_size = max(min_size + 1, min(grid_size // 4, grid_size - 6))
        
        height = random.randint(min_size, max_size)
        width = random.randint(min_size, max_size)
        
        max_start_r = max(2, grid_size - height - 2)
        max_start_c = max(2, grid_size - width - 2)
        
        if max_start_r < 2 or max_start_c < 2:
            return []
        
        start_r = random.randint(2, max_start_r)
        start_c = random.randint(2, max_start_c)
        
        boundary_path = []
        
        # Top edge
        for c in range(start_c, start_c + width):
            boundary_path.append((start_r, c))
        
        # Right edge (skip corner)
        for r in range(start_r + 1, start_r + height):
            boundary_path.append((r, start_c + width - 1))
        
        # Bottom edge (skip corner)
        for c in range(start_c + width - 2, start_c - 1, -1):
            boundary_path.append((start_r + height - 1, c))
        
        # Left edge (skip corners)
        for r in range(start_r + height - 2, start_r, -1):
            boundary_path.append((r, start_c))
        
        return boundary_path

    def _generate_irregular_polygon_path(self, grid_size: int) -> List[Tuple[int, int]]:
        """Generate an irregular polygon boundary path."""
        margin = 3
        min_center = margin + 3
        max_center = grid_size - margin - 3
        
        if max_center <= min_center:
            return []
        
        center_r = random.randint(min_center, max_center)
        center_c = random.randint(min_center, max_center)
        
        num_vertices = random.randint(4, 6)
        angles = sorted([random.uniform(0, 2 * 3.14159) for _ in range(num_vertices)])
        
        max_radius = min(grid_size // 6, min(center_r - margin, center_c - margin, 
                                       grid_size - center_r - margin, grid_size - center_c - margin))
        min_radius = max(2, max_radius // 2)
        
        if max_radius <= min_radius:
            return []
        
        vertices = []
        for angle in angles:
            radius = random.uniform(min_radius, max_radius)
            r = int(center_r + radius * np.cos(angle))
            c = int(center_c + radius * np.sin(angle))
            r = max(margin, min(grid_size - margin - 1, r))
            c = max(margin, min(grid_size - margin - 1, c))
            vertices.append((r, c))
        
        boundary_path = []
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            path_segment = self._generate_orthogonal_path(start, end)
            for point in path_segment:
                if not boundary_path or point != boundary_path[-1]:
                    boundary_path.append(point)
        
        # Add some random bumps to make it more irregular
        for _ in range(random.randint(1, 2)):
            self._add_random_bump(boundary_path, grid_size)
        
        return boundary_path

    def _is_rectangular(self, coords: List[Tuple[int, int]]) -> bool:
        """Check if coordinates form a rectangular boundary (not filled rectangle)."""
        if not coords:
            return False
        
        coord_set = set(coords)
        min_r = min(r for r, c in coords)
        max_r = max(r for r, c in coords)
        min_c = min(c for r, c in coords)
        max_c = max(c for r, c in coords)
        
        # Check if it forms a rectangular boundary
        expected_boundary = set()
        
        # Top and bottom edges
        for c in range(min_c, max_c + 1):
            expected_boundary.add((min_r, c))
            expected_boundary.add((max_r, c))
        
        # Left and right edges
        for r in range(min_r + 1, max_r):
            expected_boundary.add((r, min_c))
            expected_boundary.add((r, max_c))
        
        return coord_set == expected_boundary

    def _is_valid_single_boundary(self, grid: np.ndarray, boundary_path: List[Tuple[int, int]], boundary_color: int) -> bool:
        """Validate a single boundary object."""
        coords = set(boundary_path)
        
        # Check connectivity
        for r, c in boundary_path:
            neighbors = 0
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if (r + dr, c + dc) in coords:
                    neighbors += 1
            if neighbors != 2:
                return False
        
        # Check if encloses area
        for r in range(1, grid.shape[0] - 1):
            for c in range(1, grid.shape[1] - 1):
                if grid[r, c] == 0 and self._is_interior_cell(grid, r, c, boundary_color):
                    return True
        return False

    def _is_interior_cell(self, grid, r, c, boundary_color):
        """Check if a cell is interior using flood fill."""
        if grid[r, c] != 0:
            return False
        
        visited = set()
        stack = [(r, c)]
        
        while stack:
            curr_r, curr_c = stack.pop()
            if (curr_r, curr_c) in visited:
                continue
            visited.add((curr_r, curr_c))
            
            if (curr_r == 0 or curr_r == grid.shape[0] - 1 or 
                curr_c == 0 or curr_c == grid.shape[1] - 1):
                return False
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and 
                    grid[nr, nc] == 0 and (nr, nc) not in visited):
                    stack.append((nr, nc))
        
        return True

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by recoloring corner cells."""
        output_grid = grid.copy()
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        for boundary_obj in objects:
            coords = boundary_obj.coords
            corner_cells = []
            
            for r, c in coords:
                neighbors = []
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in coords:
                        neighbors.append((dr, dc))
                
                if len(neighbors) == 2:
                    (dr1, dc1), (dr2, dc2) = neighbors
                    if dr1 * dr2 + dc1 * dc2 == 0:
                        corner_cells.append((r, c))
            
            for r, c in corner_cells:
                if self._is_exterior_corner(grid, r, c):
                    output_grid[r, c] = taskvars['exterior_border']
                else:
                    output_grid[r, c] = taskvars['interior_border']
        
        return output_grid

    def _is_exterior_corner(self, grid, r, c):
        """Determine if a corner is on the exterior side."""
        boundary_color = grid[r, c]
        neighbors = []
        
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and 
                grid[nr, nc] == boundary_color):
                neighbors.append((dr, dc))
        
        if len(neighbors) != 2:
            return True
        
        (dr1, dc1), (dr2, dc2) = neighbors
        diag_r, diag_c = r - dr1 - dr2, c - dc1 - dc2
        
        if not (0 <= diag_r < grid.shape[0] and 0 <= diag_c < grid.shape[1]):
            return True
        
        if grid[diag_r, diag_c] == 0:
            return not self._is_interior_cell(grid, diag_r, diag_c, grid[r, c])
        
        return True

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and generate train/test grids."""
        for attempt in range(10):  # Try multiple times to generate valid grids
            try:
                taskvars = {
                    'grid_size': random.randint(15, 30),  # Larger grids for multiple objects
                    'interior_border': random.randint(1, 9),
                    'exterior_border': random.randint(1, 9)
                }
                
                # Ensure distinct colors
                while taskvars['interior_border'] == taskvars['exterior_border']:
                    taskvars['exterior_border'] = random.randint(1, 9)
                
                train_examples = []
                used_colors = set()
                
                for i in range(3):
                    boundary_color = random.randint(1, 9)
                    while (boundary_color == taskvars['interior_border'] or 
                           boundary_color == taskvars['exterior_border'] or
                           boundary_color in used_colors):
                        boundary_color = random.randint(1, 9)
                    
                    used_colors.add(boundary_color)
                    gridvars = {'boundary_color': boundary_color}
                    input_grid = self.create_input(taskvars, gridvars)
                    output_grid = self.transform_input(input_grid, taskvars)
                    
                    # Ensure we have at least one 3-object grid in training
                    if i == 2 and len(find_connected_objects(input_grid, diagonal_connectivity=False, background=0)) < 3:
                        # Retry this example to get a 3-object case
                        for _ in range(5):
                            input_grid = self.create_input(taskvars, gridvars)
                            if len(find_connected_objects(input_grid, diagonal_connectivity=False, background=0)) == 3:
                                output_grid = self.transform_input(input_grid, taskvars)
                                break
                    
                    train_examples.append({'input': input_grid, 'output': output_grid})
                
                # Test example - force 3 objects
                boundary_color = random.randint(1, 9)
                while (boundary_color == taskvars['interior_border'] or 
                       boundary_color == taskvars['exterior_border'] or
                       boundary_color in used_colors):
                    boundary_color = random.randint(1, 9)
                
                gridvars = {'boundary_color': boundary_color}
                test_input = self.create_input(taskvars, gridvars)
                
                # Ensure test has 3 objects
                for _ in range(5):
                    if len(find_connected_objects(test_input, diagonal_connectivity=False, background=0)) == 3:
                        break
                    test_input = self.create_input(taskvars, gridvars)
                
                test_output = self.transform_input(test_input, taskvars)
                
                return taskvars, {
                    'train': train_examples,
                    'test': [{'input': test_input, 'output': test_output}]
                }
            except ValueError:
                continue
        
        raise ValueError("Failed to generate valid grids after multiple attempts")

