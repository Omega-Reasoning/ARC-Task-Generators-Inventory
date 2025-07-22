from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, random_cell_coloring, retry
import numpy as np
import random

class LargestShapeColorTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares of size {vars['rows']} x {vars['columns']}.",
            "Each input grid consists of a huge chunk of shapes of distinct color. No two shapes can be of the same size.",
            "Each input grid must contain at least 3 shapes.",
            "For the grid background, it is a disorganised and incomplete checkerboard pattern comprising two colors only.",
            "The background colors and shape colors are unique and not the same."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is a 3x3 grid.",
            "Check for the largest shape from the input grid, and fill the output grid with that shape color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def _create_incomplete_disorganized_checkerboard(self, size: int, bg_color1: int, bg_color2: int) -> np.ndarray:
        """Create an incomplete and disorganized checkerboard with gaps and irregular patterns."""
        grid = np.zeros((size, size), dtype=int)  # Start with all zeros (empty)
        
        # Create scattered checkerboard-like chunks
        for i in range(size):
            for j in range(size):
                # Only fill some cells to make it incomplete
                if random.random() < 0.7:  # 70% chance to place a background cell
                    # Create irregular checkerboard pattern with some randomness
                    if random.random() < 0.6:  # 60% follow checkerboard logic
                        if (i + j) % 2 == 0:
                            grid[i, j] = bg_color1
                        else:
                            grid[i, j] = bg_color2
                    else:  # 40% random choice to break the pattern
                        grid[i, j] = random.choice([bg_color1, bg_color2])
                # else: leave as 0 (empty/gap)
        
        # Create some small empty chunks by clearing random rectangular areas
        num_gaps = random.randint(3, 6)
        for _ in range(num_gaps):
            gap_size = random.randint(1, 3)
            start_row = random.randint(0, size - gap_size)
            start_col = random.randint(0, size - gap_size)
            
            # Clear a small rectangular area
            for r in range(start_row, min(start_row + gap_size, size)):
                for c in range(start_col, min(start_col + gap_size, size)):
                    grid[r, c] = 0
        
        # Add some scattered single cells to break uniformity
        num_scattered = random.randint(5, 10)
        for _ in range(num_scattered):
            r = random.randint(0, size - 1)
            c = random.randint(0, size - 1)
            if random.random() < 0.5:
                grid[r, c] = random.choice([bg_color1, bg_color2])
            else:
                grid[r, c] = 0  # Create gaps
        
        return grid

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with at least 3 distinct colored shapes on incomplete disorganized checkerboard background."""
        size = gridvars['grid_size']
        
        # Generate unique colors for this specific grid
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        
        bg_color1 = all_colors[0]
        bg_color2 = all_colors[1]
        shape_colors = all_colors[2:8]  # Take 6 shape colors
        
        # Create incomplete disorganized checkerboard pattern
        grid = self._create_incomplete_disorganized_checkerboard(size, bg_color1, bg_color2)
        
        # Add distinct colored shapes with different sizes
        shapes_added = []
        
        # Ensure we try to place at least 3 shapes
        min_shapes_required = 3
        max_attempts_per_shape = 50
        total_attempts = 0
        max_total_attempts = 200
        
        for color_idx, color in enumerate(shape_colors):
            if total_attempts > max_total_attempts:
                break
                
            placed = False
            
            for attempt in range(max_attempts_per_shape):
                total_attempts += 1
                
                # Random shape size (ensuring uniqueness)
                # Make shapes smaller for smaller grids to fit more shapes
                max_shape_size = max(3, size // 4)
                min_shape_size = 2
                shape_width = random.randint(min_shape_size, max_shape_size)
                shape_height = random.randint(min_shape_size, max_shape_size)
                shape_area = shape_width * shape_height
                
                # Check if this size is already used
                if any(len(shape['coords']) == shape_area for shape in shapes_added):
                    continue
                
                # Try to place shape
                max_row = size - shape_height
                max_col = size - shape_width
                
                if max_row > 0 and max_col > 0:
                    start_row = random.randint(0, max_row)
                    start_col = random.randint(0, max_col)
                    
                    # Check if area is available (maintain distance from other shapes)
                    can_place = True
                    buffer = 1  # Reduced buffer for smaller grids
                    
                    # Check for conflicts with existing shapes
                    for existing_shape in shapes_added:
                        for (er, ec) in existing_shape['coords']:
                            for r in range(max(0, start_row - buffer), 
                                         min(size, start_row + shape_height + buffer)):
                                for c in range(max(0, start_col - buffer), 
                                             min(size, start_col + shape_width + buffer)):
                                    if (er, ec) == (r, c):
                                        can_place = False
                                        break
                                if not can_place:
                                    break
                            if not can_place:
                                break
                        if not can_place:
                            break
                    
                    if can_place:
                        # Create rectangular shape
                        shape_coords = []
                        for r in range(start_row, start_row + shape_height):
                            for c in range(start_col, start_col + shape_width):
                                if r < size and c < size:
                                    grid[r, c] = color
                                    shape_coords.append((r, c))
                        
                        if shape_coords:  # Only add if we actually placed something
                            shapes_added.append({
                                'color': color,
                                'coords': shape_coords
                            })
                            placed = True
                            break
                
                if placed:
                    break
        
        # Verify we have at least 3 shapes, if not, try again with relaxed constraints
        if len(shapes_added) < min_shapes_required:
            return self._create_input_with_relaxed_constraints(size, bg_color1, bg_color2, shape_colors, min_shapes_required)
        
        return grid

    def _create_input_with_relaxed_constraints(self, size: int, bg_color1: int, bg_color2: int, 
                                             shape_colors: list, min_shapes: int) -> np.ndarray:
        """Create input with more relaxed constraints to ensure minimum shape count."""
        
        # Create incomplete disorganized checkerboard background
        grid = self._create_incomplete_disorganized_checkerboard(size, bg_color1, bg_color2)
        
        # Place shapes with relaxed distance constraints
        shapes_added = []
        used_sizes = set()
        
        for i in range(min(len(shape_colors), min_shapes + 2)):  # Try to place a few more than minimum
            color = shape_colors[i]
            placed = False
            
            for attempt in range(100):
                # Generate unique size
                shape_size = random.randint(1, max(2, size // 5))
                if shape_size in used_sizes:
                    continue
                
                # Try different positions
                for pos_attempt in range(50):
                    start_row = random.randint(0, size - shape_size)
                    start_col = random.randint(0, size - shape_size)
                    
                    # Check if we can place a square shape here
                    can_place = True
                    for existing_shape in shapes_added:
                        for (er, ec) in existing_shape['coords']:
                            if (start_row <= er < start_row + shape_size and 
                                start_col <= ec < start_col + shape_size):
                                can_place = False
                                break
                        if not can_place:
                            break
                    
                    if can_place:
                        # Place square shape
                        shape_coords = []
                        for r in range(start_row, start_row + shape_size):
                            for c in range(start_col, start_col + shape_size):
                                grid[r, c] = color
                                shape_coords.append((r, c))
                        
                        shapes_added.append({
                            'color': color,
                            'coords': shape_coords
                        })
                        used_sizes.add(shape_size)
                        placed = True
                        break
                
                if placed:
                    break
            
            if len(shapes_added) >= min_shapes:
                break
        
        return grid

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        """Transform input to 3x3 output filled with largest shape color."""
        # Find all unique colors in the grid
        unique_colors = np.unique(grid)
        
        # Remove 0 (empty cells) from unique colors for background detection
        unique_colors = unique_colors[unique_colors != 0]
        
        if len(unique_colors) < 3:  # Not enough colors
            return np.full((3, 3), unique_colors[0] if len(unique_colors) > 0 else 1, dtype=int)
        
        # We need to identify background colors vs shape colors
        # Background colors appear most frequently in checkerboard pattern
        color_counts = {color: np.sum(grid == color) for color in unique_colors}
        
        # Get the two most frequent colors (likely background colors)
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        bg_colors = {sorted_colors[0][0], sorted_colors[1][0]}
        
        # Create a mask for non-background pixels
        mask = np.ones_like(grid, dtype=bool)
        for bg_color in bg_colors:
            mask &= (grid != bg_color)
        
        # Also exclude empty cells (0)
        mask &= (grid != 0)
        
        # Find connected components
        objects = find_connected_objects(
            np.where(mask, grid, 0), 
            diagonal_connectivity=False,
            background=0
        )
        
        # Find the largest object
        largest_color = unique_colors[2] if len(unique_colors) > 2 else unique_colors[0]  # Default fallback
        max_size = 0
        
        for obj in objects:
            if len(obj) > max_size:
                max_size = len(obj)
                largest_color = list(obj.colors)[0]  # Get the color of this object
        
        # Create 3x3 output grid filled with largest shape color
        output_grid = np.full((3, 3), largest_color, dtype=int)
        
        return output_grid

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # Store task variables (only grid dimensions now)
        taskvars = {
            'rows': None,  # Will be set per grid
            'columns': None,  # Will be set per grid
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate grid sizes - make them larger to accommodate at least 3 shapes
        min_size = 10
        max_size = 20
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            size = all_sizes[i]
            taskvars['rows'] = size
            taskvars['columns'] = size
            
            gridvars = {'grid_size': size}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_size = all_sizes[-1]
        taskvars['rows'] = test_size
        taskvars['columns'] = test_size
        test_gridvars = {'grid_size': test_size}
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

# Test code
if __name__ == "__main__":
    generator = LargestShapeColorTaskGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    print(f"Number of train examples: {len(train_test_data['train'])}")
    print(f"Number of test examples: {len(train_test_data['test'])}")
    
    # Visualize if the visualization method is available
    try:
        ARCTaskGenerator.visualize_train_test_data(train_test_data)
    except:
        print("Visualization not available, but grids created successfully!")