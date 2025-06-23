from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Taskaf902bf9Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of varying sizes.",
            "The grid contains 1-3 sets of patterns. Each pattern consists of 4 corner cells in {color('corner_color')}.",
            "Patterns can be of three types:",
            "1. Square pattern: 4 corner cells forming a square",
            "2. Rectangle pattern: 4 corner cells forming a wider rectangle",
            "3. Cross pattern: 4 corner cells with exactly one empty cell between them"
        ]
        
        transformation_reasoning_chain = [
            "The output grid copies all corner cells from the input grid.",
            "For square and rectangle patterns: Fill all cells between the corners with {color('quad_color')}",
            "For cross patterns: Fill only the center cell between the corners with {color('quad_color')}",
            "Each pattern is transformed independently while maintaining its type and position"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with corner patterns."""
        grid_size = gridvars["grid_size"]
        corner_color = taskvars["corner_color"]
        grid = np.zeros((grid_size, grid_size), dtype=int)

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

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by detecting patterns and filling them according to their type."""
        output_grid = np.copy(grid)
        quad_color = taskvars["quad_color"]
        corner_color = taskvars["corner_color"]
        
        # Find all corner cells
        corner_positions = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == corner_color:
                    corner_positions.append((r, c))
        
        # Group corners into patterns (sets of 4 corners that form rectangles)
        patterns = self._find_rectangular_patterns(corner_positions)
        
        # Fill each pattern according to its type
        for pattern_corners in patterns:
            pattern_type = self._classify_pattern(pattern_corners)
            
            if pattern_type == 'cross':
                # Fill only center cell for cross pattern
                min_r = min(r for r, _ in pattern_corners)
                max_r = max(r for r, _ in pattern_corners)
                min_c = min(c for _, c in pattern_corners)
                max_c = max(c for _, c in pattern_corners)
                center_r = (min_r + max_r) // 2
                center_c = (min_c + max_c) // 2
                if output_grid[center_r, center_c] == 0:
                    output_grid[center_r, center_c] = quad_color
            else:
                # Fill inner area for squares and rectangles
                min_r = min(r for r, _ in pattern_corners)
                max_r = max(r for r, _ in pattern_corners)
                min_c = min(c for _, c in pattern_corners)
                max_c = max(c for _, c in pattern_corners)
                for r in range(min_r + 1, max_r):
                    for c in range(min_c + 1, max_c):
                        if output_grid[r, c] == 0:
                            output_grid[r, c] = quad_color
        
        return output_grid
    
    def _find_rectangular_patterns(self, corner_positions):
        """Find groups of 4 corners that form rectangles."""
        patterns = []
        used_corners = set()
        
        for i, corner1 in enumerate(corner_positions):
            if corner1 in used_corners:
                continue
                
            r1, c1 = corner1
            
            # Find other corners that could form a rectangle with corner1
            for j, corner2 in enumerate(corner_positions[i+1:], i+1):
                if corner2 in used_corners:
                    continue
                    
                r2, c2 = corner2
                
                # Check if corner1 and corner2 can be opposite corners or adjacent corners
                for k, corner3 in enumerate(corner_positions[j+1:], j+1):
                    if corner3 in used_corners:
                        continue
                        
                    r3, c3 = corner3
                    
                    for l, corner4 in enumerate(corner_positions[k+1:], k+1):
                        if corner4 in used_corners:
                            continue
                            
                        r4, c4 = corner4
                        
                        # Check if these 4 points form a rectangle
                        if self._is_rectangle([corner1, corner2, corner3, corner4]):
                            patterns.append([corner1, corner2, corner3, corner4])
                            used_corners.update([corner1, corner2, corner3, corner4])
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
        
        return patterns
    
    def _is_rectangle(self, corners):
        """Check if 4 corners form a rectangle."""
        if len(corners) != 4:
            return False
            
        rows = sorted(set(r for r, _ in corners))
        cols = sorted(set(c for _, c in corners))
        
        # Must have exactly 2 unique rows and 2 unique columns
        if len(rows) != 2 or len(cols) != 2:
            return False
            
        # Check that we have corners at all 4 combinations
        expected_corners = {(r, c) for r in rows for c in cols}
        actual_corners = set(corners)
        
        return expected_corners == actual_corners
    
    def _classify_pattern(self, corners):
        """Classify pattern as square, rectangle, or cross based on corner positions."""
        min_r = min(r for r, _ in corners)
        max_r = max(r for r, _ in corners)
        min_c = min(c for _, c in corners)
        max_c = max(c for _, c in corners)
        
        height = max_r - min_r
        width = max_c - min_c
        
        # Cross pattern: corners with exactly 2 cells between them
        if height == 2 and width == 2:
            return 'cross'
        # Square pattern: equal height and width
        elif height == width:
            return 'square'
        # Rectangle pattern: different height and width
        else:
            return 'rectangle'
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Generate random grid parameters (minimum size 6)
        grid_size = random.randint(6, 20)
        
        # Choose distinct colors for corner cells and quadrangle
        corner_color = random.randint(1, 9)
        quad_color = random.choice([c for c in range(1, 10) if c != corner_color])
            
        # Store the task variables
        taskvars = {
            "corner_color": corner_color,
            "quad_color": quad_color
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {"grid_size": grid_size}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {"grid_size": grid_size}
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