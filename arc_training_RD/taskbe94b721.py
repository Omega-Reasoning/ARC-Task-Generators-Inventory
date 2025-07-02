from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry, Contiguity, random_cell_coloring

class Taskbe94b721Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_rows']} x {vars['grid_cols']}.",
            "The grid consists of multiple patterns of different sizes. No two patterns share the same size.",
            "Each pattern is of a different color and color range between 1-9."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is formed by identifying that one single pattern which is big and covers many cells.",
            "Only that pattern must be displayed in the output grid with its color.",
            "The output grid size is exactly the bounding box of the largest pattern."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        # Use the grid size from gridvars (set once in create_grids)
        height = gridvars['grid_rows']
        width = gridvars['grid_cols']
        
        # Start with an empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Generate a random number of patterns (between 3 and 5)
        num_patterns = random.randint(3, 5)
        
        # Generate different colors for patterns (1-9)
        colors = random.sample(range(1, 10), num_patterns)
        
        # Keep track of pattern sizes to ensure they're unique
        pattern_sizes = set()
        
        # Define minimum spacing between patterns
        min_spacing = 1
        
        # Keep track of occupied areas
        occupied_cells = set()
        
        # Pre-calculate target sizes to avoid overlaps
        # Create a spread of pattern sizes with forced difference
        target_size_ranges = []
        
        # First pattern (largest) gets a size range between 15-25
        target_size_ranges.append((15, min(25, (height * width) // 8)))
        
        # Other patterns get progressively smaller size ranges
        for i in range(1, num_patterns):
            min_size = max(4, 15 - i * 3)
            max_size = min(14 - (i-1) * 2, (height * width) // 10)
            if min_size < max_size:
                target_size_ranges.append((min_size, max_size))
            else:
                target_size_ranges.append((4, max(5, max_size)))
        
        # Shuffle colors and size ranges together
        color_size_pairs = list(zip(colors, target_size_ranges))
        random.shuffle(color_size_pairs)
        
        # Create patterns with predefined size ranges
        for color, (min_size, max_size) in color_size_pairs:
            # Function to create a pattern with a size in the target range
            def create_pattern_with_target_size():
                # Choose height and width for the pattern
                pattern_height = random.randint(2, min(6, height // 2))
                pattern_width = random.randint(2, min(6, width // 2))
                
                # Create the pattern
                pattern = create_object(
                    height=pattern_height,
                    width=pattern_width,
                    color_palette=color,
                    contiguity=Contiguity.EIGHT
                )
                
                # Get the actual size
                actual_size = np.count_nonzero(pattern != 0)
                
                # Target met?
                return pattern if min_size <= actual_size <= max_size else None
            
            # Try to create a pattern with a size in the target range
            max_attempts = 50
            pattern = None
            attempts = 0
            
            while attempts < max_attempts:
                attempts += 1
                pattern_candidate = create_pattern_with_target_size()
                if pattern_candidate is None:
                    continue
                
                # Check if size is unique
                candidate_size = np.count_nonzero(pattern_candidate != 0)
                if candidate_size not in pattern_sizes:
                    pattern = pattern_candidate
                    pattern_sizes.add(candidate_size)
                    break
            
            # If failed to create a unique pattern, skip this color
            if pattern is None:
                continue
            
            # Try to find a position for the pattern with proper spacing
            placed = False
            max_position_attempts = 50
            position_attempts = 0
            
            while position_attempts < max_position_attempts and not placed:
                position_attempts += 1
                
                # Choose a random position
                row_pos = random.randint(0, height - pattern.shape[0])
                col_pos = random.randint(0, width - pattern.shape[1])
                
                # Check if position respects minimum spacing
                position_valid = True
                
                # Check for overlaps with existing patterns plus minimum spacing
                for r in range(max(0, row_pos - min_spacing), 
                             min(height, row_pos + pattern.shape[0] + min_spacing)):
                    for c in range(max(0, col_pos - min_spacing), 
                                 min(width, col_pos + pattern.shape[1] + min_spacing)):
                        if (r, c) in occupied_cells:
                            position_valid = False
                            break
                    if not position_valid:
                        break
                
                if position_valid:
                    # Place the pattern
                    for r in range(pattern.shape[0]):
                        for c in range(pattern.shape[1]):
                            if pattern[r, c] != 0:
                                grid[row_pos + r, col_pos + c] = pattern[r, c]
                                occupied_cells.add((row_pos + r, col_pos + c))
                    placed = True
        
        # Make sure we have at least 2 patterns
        # Count actual patterns in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
        if len(objects.objects) < 2:
            return self.create_input(taskvars, gridvars)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Find all connected objects in the input grid
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
        
        # Sort objects by size (largest first)
        sorted_objects = objects.sort_by_size(reverse=True)
        
        if len(sorted_objects.objects) == 0:
            # Edge case: no objects found
            return np.zeros((1, 1), dtype=int)
        
        # Get the largest object
        largest_object = sorted_objects[0]
        
        # Get the bounding box of the largest object
        bounding_box = largest_object.bounding_box
        
        # Extract the pattern as a minimal array
        pattern_array = largest_object.to_array()
        
        return pattern_array
    
    def create_grids(self):
        # Generate a single grid size that will be used for all grids
        # Make it non-square by ensuring height != width
        height = random.randint(10, 18)
        width = random.randint(12, 20)
        
        # Ensure it's not square
        while height == width:
            width = random.randint(12, 20)
        
        # Task variables including grid dimensions
        taskvars = {
            'grid_rows': height,
            'grid_cols': width
        }
        
        # Grid variables to pass to create_input
        gridvars = {
            'grid_rows': height,
            'grid_cols': width
        }
        
        # Number of train pairs (3-5)
        num_train_pairs = random.randint(3, 5)
        
        # Generate train pairs (all with the same grid size)
        train_pairs = []
        for _ in range(num_train_pairs):
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair (with the same grid size)
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)

