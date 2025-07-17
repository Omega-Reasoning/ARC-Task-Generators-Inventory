from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject

class Task88a10436Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "The grid consists of one single cell randomly placed on the grid of a distinct color.",
            "There exists a small unique pattern (within 3x3 area) on the grid of another distinct color.",
            "The pattern is a subtle, unique shape - not a basic geometric form.",
            "The single cell and pattern are placed with sufficient distance between them."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The random single cell is removed from the output grid.",
            "The same pattern shape is recreated at the position where the single cell was located.",
            "The recreated pattern is centered around the single cell position.",
            "The recreated pattern uses the same color as the original pattern."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        taskvars = {}
        
        grid_size = random.randint(10, 15)  # Increased size to accommodate spacing
        taskvars['grid_size'] = grid_size
        
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1
        
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data

    def create_small_unique_pattern(self, color):
        """Create a small unique pattern within 3x3 area."""
        patterns = [
            # L-shape
            np.array([[color, 0, 0],
                     [color, 0, 0],
                     [color, color, 0]]),
            
            # T-shape
            np.array([[color, color, color],
                     [0, color, 0],
                     [0, 0, 0]]),
            
            # Plus shape
            np.array([[0, color, 0],
                     [color, color, color],
                     [0, color, 0]]),
            
            # Z-shape
            np.array([[color, color, 0],
                     [0, color, 0],
                     [0, color, color]]),
            
            # Corner shape
            np.array([[color, color, 0],
                     [color, 0, 0],
                     [0, 0, 0]]),
            
            # Arrow shape
            np.array([[0, color, 0],
                     [color, color, 0],
                     [0, color, 0]]),
            
            # Cross shape
            np.array([[color, 0, color],
                     [0, color, 0],
                     [color, 0, color]]),
            
            # Step shape
            np.array([[color, 0, 0],
                     [color, color, 0],
                     [0, color, 0]]),
        ]
        
        return random.choice(patterns)

    def get_pattern_bounds(self, grid, pattern_color):
        """Get the bounding box of the pattern in the grid."""
        pattern_cells = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == pattern_color:
                    pattern_cells.append((r, c))
        
        if not pattern_cells:
            return None
        
        min_r = min(cell[0] for cell in pattern_cells)
        max_r = max(cell[0] for cell in pattern_cells)
        min_c = min(cell[1] for cell in pattern_cells)
        max_c = max(cell[1] for cell in pattern_cells)
        
        return (min_r, max_r, min_c, max_c)

    def is_far_enough(self, r, c, pattern_bounds, min_distance=3):
        """Check if position (r, c) is far enough from the pattern."""
        if pattern_bounds is None:
            return True
        
        min_r, max_r, min_c, max_c = pattern_bounds
        
        # Calculate minimum distance from point to pattern bounding box
        if r < min_r:
            dr = min_r - r
        elif r > max_r:
            dr = r - max_r
        else:
            dr = 0
        
        if c < min_c:
            dc = min_c - c
        elif c > max_c:
            dc = c - max_c
        else:
            dc = 0
        
        # Use Manhattan distance
        distance = dr + dc
        return distance >= min_distance

    def extract_pattern_from_grid(self, grid):
        """Extract the pattern from the grid by finding the multi-cell object."""
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        # Find the pattern object (multi-cell object)
        pattern_obj = None
        for obj in objects.objects:
            if len(obj.cells) > 1:  # This is the pattern, not the single cell
                pattern_obj = obj
                break
        
        if pattern_obj is None:
            return None, None
        
        # Get bounding box
        r_slice, c_slice = pattern_obj.bounding_box
        
        # Extract pattern as a 2D array
        pattern_height = r_slice.stop - r_slice.start
        pattern_width = c_slice.stop - c_slice.start
        pattern = np.zeros((pattern_height, pattern_width), dtype=int)
        
        # Fill the pattern array
        for cell in pattern_obj.cells:
            r, c = cell[:2]
            pattern[r - r_slice.start, c - c_slice.start] = grid[r, c]
        
        return pattern, grid[r_slice.start, c_slice.start]  # Return pattern and its color

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with a small unique pattern and a single colored cell."""
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Randomly select 2 different colors for each grid
        available_colors = list(range(1, 10))
        single_cell_color, pattern_color = random.sample(available_colors, 2)
        
        # Create a small unique pattern (3x3)
        pattern = self.create_small_unique_pattern(pattern_color)
        pattern_height, pattern_width = pattern.shape
        
        # Place the pattern at a random position (ensuring it fits)
        r_pos = random.randint(0, grid_size - pattern_height)
        c_pos = random.randint(0, grid_size - pattern_width)
        
        for r in range(pattern_height):
            for c in range(pattern_width):
                if pattern[r, c] != 0:
                    grid[r + r_pos, c + c_pos] = pattern[r, c]
        
        # Get pattern bounds for distance checking
        pattern_bounds = self.get_pattern_bounds(grid, pattern_color)
        
        # Place a single cell of single_cell_color at a position far from the pattern
        # Make sure there's enough space around it to center the pattern
        margin = 2  # At least 2 cells margin from edges for centering
        min_distance = 3  # Minimum distance from pattern
        
        # Find all valid positions that are far enough from the pattern
        valid_positions = []
        for r in range(margin, grid_size - margin):
            for c in range(margin, grid_size - margin):
                if (grid[r, c] == 0 and 
                    self.is_far_enough(r, c, pattern_bounds, min_distance)):
                    valid_positions.append((r, c))
        
        if valid_positions:
            r_main, c_main = random.choice(valid_positions)
            grid[r_main, c_main] = single_cell_color
        else:
            # Fallback: find any empty position far from edges
            empty_cells = []
            for r in range(margin, grid_size - margin):
                for c in range(margin, grid_size - margin):
                    if grid[r, c] == 0:
                        empty_cells.append((r, c))
            
            if empty_cells:
                r_main, c_main = random.choice(empty_cells)
                grid[r_main, c_main] = single_cell_color
            else:
                # Last resort
                r_main, c_main = random.randint(margin, grid_size-1-margin), random.randint(margin, grid_size-1-margin)
                grid[r_main, c_main] = single_cell_color
            
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by removing single cell and recreating pattern at that position."""
        # Copy the input grid
        output_grid = grid.copy()
        
        # Extract the pattern from the original grid before removing the single cell
        pattern, pattern_color = self.extract_pattern_from_grid(grid)
        
        if pattern is None:
            return output_grid
        
        # Get all objects from the grid
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        # Find the single-cell object and its position
        single_cell_pos = None
        for obj in objects.objects:
            if len(obj.cells) == 1:
                r, c = next(iter(obj.cells))[:2]
                single_cell_pos = (r, c)
                # Remove the single cell from output grid
                output_grid[r, c] = 0
                break
        
        if single_cell_pos is None:
            return output_grid
        
        # Get pattern dimensions
        pattern_height, pattern_width = pattern.shape
        
        # Center the pattern around the single cell's position
        r_center, c_center = single_cell_pos
        r_start = r_center - pattern_height // 2
        c_start = c_center - pattern_width // 2
        
        # Place the pattern centered around the single cell position
        for r in range(pattern_height):
            for c in range(pattern_width):
                if pattern[r, c] != 0:
                    new_r = r_start + r
                    new_c = c_start + c
                    # Check bounds
                    if 0 <= new_r < output_grid.shape[0] and 0 <= new_c < output_grid.shape[1]:
                        output_grid[new_r, new_c] = pattern_color

        return output_grid

    def create_grids_default(self, nr_train_examples: int, nr_test_examples: int, taskvars: dict) -> dict:
        """Create training and test grids."""
        train_pairs = []
        test_pairs = []
        
        # Generate training examples
        for _ in range(nr_train_examples):
            gridvars = {}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test examples
        for _ in range(nr_test_examples):
            gridvars = {}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            test_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        return TrainTestData(train=train_pairs, test=test_pairs)

# Test the generator
if __name__ == "__main__":
    generator = Task88a10436Generator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:")
    print(f"Grid size: {taskvars['grid_size']}")
    
    generator.visualize_train_test_data(train_test_data)