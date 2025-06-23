from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random

class Taska87f7484Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids should follow a consistent structure:",
            "Horizontal grids must have 3 rows.",
            "Vertical grids must have 3 columns.", 
            "Example sizes: 3x9, 12x3, 15x3, 3x12.",
            "Each grid is composed of repeating 3x3 patterns.",
            "All patterns are visually similar except one, which is different.",
            "Each pattern (including the different pattern) is visually distinguishable by color."
        ]
        
        transformation_reasoning_chain = [
            "The output is a 3x3 grid.",
            "Identify the one unique/different 3x3 pattern in the input grid.", 
            "Return that different 3x3 pattern as the output."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        """Create input grid with repeating 3x3 patterns where one is different."""
        # Decide if horizontal (3 rows) or vertical (3 columns)
        is_horizontal = random.choice([True, False])
        
        # Determine number of patterns (3-6 patterns total)
        num_patterns = random.randint(3, 6)
        
        if is_horizontal:
            grid_height = 3
            grid_width = num_patterns * 3
        else:
            grid_height = num_patterns * 3  
            grid_width = 3
            
        # Ensure grid stays within bounds
        if grid_height > 20:
            grid_height = 20
            num_patterns = grid_height // 3
        if grid_width > 20:
            grid_width = 20  
            num_patterns = grid_width // 3
            
        # Create base pattern that will be repeated
        base_pattern = self._create_base_pattern()
        
        # Create the unique/different pattern
        unique_pattern = self._create_unique_pattern(base_pattern)
        
        # Choose position for unique pattern
        unique_position = random.randint(0, num_patterns - 1)
        
        # Get available colors (excluding background 0)
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        pattern_colors = available_colors[:num_patterns]
        
        # Create the full grid
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Store the unique pattern with its color for later use
        self.unique_pattern_output = unique_pattern.copy()
        self.unique_pattern_output[self.unique_pattern_output != 0] = pattern_colors[unique_position]
        
        for i in range(num_patterns):
            color = pattern_colors[i]
            
            if i == unique_position:
                pattern = unique_pattern.copy()
            else:
                pattern = base_pattern.copy()
                
            # Color the pattern
            pattern[pattern != 0] = color
            
            # Place pattern in grid
            if is_horizontal:
                grid[0:3, i*3:(i+1)*3] = pattern
            else:
                grid[i*3:(i+1)*3, 0:3] = pattern
                
        return grid
    
    def transform_input(self, input_grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        """Extract the unique 3x3 pattern."""
        return self.unique_pattern_output
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # No task variables needed for this task
        taskvars = {}
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {}  # No grid variables needed
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {}
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
    
    # Helper methods for pattern creation
    def _create_base_pattern(self):
        """Create a base 3x3 pattern that fills most of the space"""
        pattern_types = [
            self._create_checkerboard_pattern,
            self._create_border_pattern,
            self._create_diagonal_pattern,
            self._create_filled_shape_pattern,
            self._create_stripes_pattern,
            self._create_corner_fill_pattern
        ]
        
        pattern_creator = random.choice(pattern_types)
        return pattern_creator()
    
    def _create_unique_pattern(self, base_pattern):
        """Create a unique pattern that's different from the base"""
        unique_types = [
            self._create_inverse_pattern,
            self._create_rotated_pattern,
            self._create_different_full_pattern
        ]
        
        pattern_creator = random.choice(unique_types)
        return pattern_creator(base_pattern)
    
    def _create_checkerboard_pattern(self):
        """Create a checkerboard pattern"""
        pattern = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                if (i + j) % 2 == 0:
                    pattern[i, j] = 1
        return pattern
    
    def _create_border_pattern(self):
        """Create a border/frame pattern"""
        pattern = np.zeros((3, 3), dtype=int)
        # Fill border
        pattern[0, :] = 1  # Top row
        pattern[2, :] = 1  # Bottom row
        pattern[:, 0] = 1  # Left column
        pattern[:, 2] = 1  # Right column
        # Randomly fill or leave center empty
        if random.choice([True, False]):
            pattern[1, 1] = 1
        return pattern
    
    def _create_diagonal_pattern(self):
        """Create diagonal patterns"""
        pattern = np.zeros((3, 3), dtype=int)
        diagonal_type = random.choice(['main', 'anti', 'both', 'thick_main', 'thick_anti'])
        
        if diagonal_type == 'main':
            for i in range(3):
                pattern[i, i] = 1
        elif diagonal_type == 'anti':
            for i in range(3):
                pattern[i, 2-i] = 1
        elif diagonal_type == 'both':
            for i in range(3):
                pattern[i, i] = 1
                pattern[i, 2-i] = 1
        elif diagonal_type == 'thick_main':
            # Main diagonal plus adjacent cells
            for i in range(3):
                pattern[i, i] = 1
                if i > 0:
                    pattern[i, i-1] = 1
                if i < 2:
                    pattern[i, i+1] = 1
        elif diagonal_type == 'thick_anti':
            # Anti-diagonal plus adjacent cells
            for i in range(3):
                pattern[i, 2-i] = 1
                if i > 0 and 2-i+1 < 3:
                    pattern[i, 2-i+1] = 1
                if i < 2 and 2-i-1 >= 0:
                    pattern[i, 2-i-1] = 1
        
        return pattern
    
    def _create_filled_shape_pattern(self):
        """Create filled geometric shapes"""
        pattern = np.zeros((3, 3), dtype=int)
        shape_type = random.choice(['L', 'T', 'plus', 'corners', 'center_cross'])
        
        if shape_type == 'L':
            pattern[0, 0] = pattern[1, 0] = pattern[2, 0] = 1
            pattern[2, 1] = pattern[2, 2] = 1
        elif shape_type == 'T':
            pattern[0, :] = 1
            pattern[1, 1] = pattern[2, 1] = 1
        elif shape_type == 'plus':
            pattern[1, :] = 1
            pattern[:, 1] = 1
        elif shape_type == 'corners':
            pattern[0, 0] = pattern[0, 2] = 1
            pattern[2, 0] = pattern[2, 2] = 1
            pattern[1, 0] = pattern[1, 2] = 1
            pattern[0, 1] = pattern[2, 1] = 1
        elif shape_type == 'center_cross':
            pattern[1, 1] = 1
            pattern[0, 1] = pattern[2, 1] = 1
            pattern[1, 0] = pattern[1, 2] = 1
        
        return pattern
    
    def _create_stripes_pattern(self):
        """Create stripe patterns"""
        pattern = np.zeros((3, 3), dtype=int)
        stripe_type = random.choice(['horizontal', 'vertical', 'alternating_h', 'alternating_v'])
        
        if stripe_type == 'horizontal':
            rows_to_fill = random.choice([[0, 2], [1], [0, 1], [1, 2]])
            for row in rows_to_fill:
                pattern[row, :] = 1
        elif stripe_type == 'vertical':
            cols_to_fill = random.choice([[0, 2], [1], [0, 1], [1, 2]])
            for col in cols_to_fill:
                pattern[:, col] = 1
        elif stripe_type == 'alternating_h':
            pattern[0, :] = 1
            pattern[2, :] = 1
        elif stripe_type == 'alternating_v':
            pattern[:, 0] = 1
            pattern[:, 2] = 1
        
        return pattern
    
    def _create_corner_fill_pattern(self):
        """Create patterns that fill corners and surrounding areas"""
        pattern = np.zeros((3, 3), dtype=int)
        corner_type = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right', 'opposite_corners'])
        
        if corner_type == 'top_left':
            pattern[0, 0] = pattern[0, 1] = pattern[1, 0] = 1
            pattern[1, 1] = pattern[0, 2] = pattern[2, 0] = 1
        elif corner_type == 'top_right':
            pattern[0, 2] = pattern[0, 1] = pattern[1, 2] = 1
            pattern[1, 1] = pattern[0, 0] = pattern[2, 2] = 1
        elif corner_type == 'bottom_left':
            pattern[2, 0] = pattern[2, 1] = pattern[1, 0] = 1
            pattern[1, 1] = pattern[2, 2] = pattern[0, 0] = 1
        elif corner_type == 'bottom_right':
            pattern[2, 2] = pattern[2, 1] = pattern[1, 2] = 1
            pattern[1, 1] = pattern[2, 0] = pattern[0, 2] = 1
        elif corner_type == 'opposite_corners':
            pattern[0, 0] = pattern[0, 1] = pattern[1, 0] = 1
            pattern[2, 2] = pattern[2, 1] = pattern[1, 2] = 1
        
        return pattern
    
    def _create_inverse_pattern(self, base_pattern):
        """Create inverse of base pattern"""
        pattern = np.ones((3, 3), dtype=int)
        pattern[base_pattern != 0] = 0
        return pattern
    
    def _create_rotated_pattern(self, base_pattern):
        """Create rotated version of base pattern"""
        rotations = random.choice([1, 2, 3])
        return np.rot90(base_pattern, rotations)
    
    def _create_different_full_pattern(self, base_pattern):
        """Create completely different full pattern"""
        new_pattern = self._create_base_pattern()
        attempts = 0
        while np.array_equal(new_pattern, base_pattern) and attempts < 10:
            new_pattern = self._create_base_pattern()
            attempts += 1
        return new_pattern

