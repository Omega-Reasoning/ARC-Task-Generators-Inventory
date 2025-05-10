from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Taskaf902bf9Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of varying sizes.",
            "The grid contains 1-3 sets of patterns. Each pattern consists of 4 corner cells in {{color(\"corner_color\")}}.",
            "Patterns can be of three types:",
            "1. Square pattern: 4 corner cells forming a square",
            "2. Rectangle pattern: 4 corner cells forming a wider rectangle",
            "3. Cross pattern: 4 corner cells with exactly one empty cell between them"
        ]
        
        transformation_reasoning_chain = [
            "The output grid copies all corner cells from the input grid.",
            "For square and rectangle patterns: Fill all cells between the corners with {{color(\"quad_color\")}}",
            "For cross patterns: Fill only the center cell between the corners with {{color(\"quad_color\")}}",
            "Each pattern is transformed independently while maintaining its type and position"
        ]
        
        taskvars_definitions = {
            "grid_size": "Size of the square grid",
            "corner_color": "Color of the corner cells in all patterns (between 1 and 9)",
            "quad_color": "Color used to fill spaces between corners (between 1 and 9, different from corner_color)"
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self):
        grid_size = self.taskvars["grid_size"]  # Remove minimum size constraint
        corner_color = self.taskvars["corner_color"]
        grid = np.zeros((grid_size, grid_size), dtype=int)
        self.quad_points = []

        # Randomly choose how many and which patterns to include
        available_patterns = ['square', 'rectangle', 'single'] * 2  # Allow up to 2 of each type
        num_patterns = random.randint(1, 3)  # Include 1-3 patterns
        selected_patterns = random.sample(available_patterns, num_patterns)
        
        # Create zones dynamically based on number of patterns
        zones = []
        for i in range(num_patterns):
            row = (i // 2) * (grid_size // 2)  # Alternate between top and bottom half
            col = (i % 2) * (grid_size // 2)   # Alternate between left and right half
            zones.append((row, col))

        for (zone_r, zone_c), pattern_type in zip(zones, selected_patterns):
            if pattern_type == 'square':
                size = 3  # Fixed size for predictability
                corners = [
                    (zone_r + 1, zone_c + 1),
                    (zone_r + 1, zone_c + 1 + size),
                    (zone_r + 1 + size, zone_c + 1),
                    (zone_r + 1 + size, zone_c + 1 + size)
                ]
            
            elif pattern_type == 'rectangle':
                width = 4  # Fixed width
                height = 2  # Fixed height
                corners = [
                    (zone_r + 1, zone_c + 1),
                    (zone_r + 1, zone_c + 1 + width),
                    (zone_r + 1 + height, zone_c + 1),
                    (zone_r + 1 + height, zone_c + 1 + width)
                ]
            
            elif pattern_type == 'single':
                # Create cross-like pattern with corners
                corners = [
                    (zone_r + 1, zone_c + 1),      # Top-left
                    (zone_r + 1, zone_c + 3),      # Top-right
                    (zone_r + 3, zone_c + 1),      # Bottom-left
                    (zone_r + 3, zone_c + 3)       # Bottom-right
                ]
                
            # Verify and place corners
            if all(r < grid_size and c < grid_size for r, c in corners):
                for r, c in corners:
                    grid[r, c] = corner_color
                
                # Add pattern info with calculated center for single type
                if pattern_type == 'single':
                    min_r = min(r for r, _ in corners)
                    max_r = max(r for r, _ in corners)
                    min_c = min(c for _, c in corners)
                    max_c = max(c for _, c in corners)
                    center = ((min_r + max_r) // 2, (min_c + max_c) // 2)
                    self.quad_points.append({
                        'type': pattern_type,
                        'corners': corners,
                        'center': center
                    })
                else:
                    self.quad_points.append({
                        'type': pattern_type,
                        'corners': corners
                    })

        return grid

    def transform_input(self, input_grid):
        output_grid = np.copy(input_grid)
        quad_color = self.taskvars["quad_color"]
        
        # Fill each pattern according to its type
        for pattern in self.quad_points:
            pattern_type = pattern['type']
            corners = pattern['corners']
            
            if pattern_type == 'single':
                # Fill only center cell for cross pattern
                center_r, center_c = pattern['center']
                output_grid[center_r, center_c] = quad_color
            else:
                # Fill inner area for squares and rectangles
                min_r = min(r for r, _ in corners)
                max_r = max(r for r, _ in corners)
                min_c = min(c for _, c in corners)
                max_c = max(c for _, c in corners)
                for r in range(min_r + 1, max_r):
                    for c in range(min_c + 1, max_c):
                        output_grid[r, c] = quad_color
        
        return output_grid
    
    def _is_region_clear(self, grid, start_r, start_c, size):
        """Check if region has enough space for a new quadrangle"""
        # Add buffer of 1 cell around the quadrangle
        buffer = 1
        min_r = max(0, start_r - buffer)
        max_r = min(grid.shape[0], start_r + size + buffer + 1)
        min_c = max(0, start_c - buffer)
        max_c = min(grid.shape[1], start_c + size + buffer + 1)
        
        return np.all(grid[min_r:max_r, min_c:max_c] == 0)
    
    def _find_quadrangle_corners(self, corners):
        """Find 4 corners that form a quadrangle"""
        for i, (r1, c1) in enumerate(corners):
            for j, (r2, c2) in enumerate(corners[i+1:], i+1):
                if r1 == r2 or c1 == c2:  # Same row or column
                    for k, (r3, c3) in enumerate(corners[j+1:], j+1):
                        if (r3 == r1 and c3 != c2) or (r3 == r2 and c3 != c1):
                            for l, (r4, c4) in enumerate(corners[k+1:], k+1):
                                if self._is_valid_quadrangle(r1, c1, r2, c2, r3, c3, r4, c4):
                                    return [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        return None
    
    def _is_valid_quadrangle(self, r1, c1, r2, c2, r3, c3, r4, c4):
        """Check if 4 corners form a valid quadrangle"""
        corners = [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
        rows = set(r for r, _ in corners)
        cols = set(c for _, c in corners)
        return len(rows) == 2 and len(cols) == 2
    
    def create_grids(self):
        # Generate random grid parameters (minimum size 6)
        grid_size = random.randint(6, 20)
        
        # Choose distinct colors for corner cells and quadrangle
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)
            
        # Store the parameters
        self.taskvars = {
            "grid_size": grid_size,
            "corner_color": color1,
            "quad_color": color2
        }
        
        # Generate training pairs
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            input_grid = self.create_input()
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_input = self.create_input()
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return self.taskvars, TrainTestData(train=train_pairs, test=test_pairs)

# Test the generator