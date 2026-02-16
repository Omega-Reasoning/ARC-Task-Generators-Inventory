from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity
from Framework.transformation_library import find_connected_objects

class Task539a4f51Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a colored square block positioned in the top-left corner, followed by one or more inverted corner paths.",
            "Each inverted corner path is a same-colored path that starts from (0, n), extends downward to (n, n), and then moves left to (n, 0).",
            "Each inverted corner path has a unique color, distinct from the top-left square block, with colors varying across examples.",
            "The last inverted corner path at index {vars['grid_size']-1} may sometimes be empty (0).",
            "The square-colored block can range in size from 1x1 to 3x3, and the first inverted corner path begins only after the square block is completed."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {2*vars['grid_size']}x{2*vars['grid_size']}.",
            "They are constructed by identifying the colored square block positioned in the top-left corner, followed by one or more inverted corner paths.",
            "An inverted corner path is a same-colored path that starts from (0, n), extends downward to (n, n), and then moves left to (n, 0).",
            "Output grids are constructed by copying all colored cells from the input grid and placing them in the top-left corner of the output grid.",
            "Next, since the output grids are larger than the input, the remaining cells are filled by continuing the inverted corner paths, starting from the very next column after the input ends, using colors repeated from the beginning of the grid.",
            "The first inverted corner shape that appears after copying the colored cells from the input grid has a color and thickness matching the width of the square block in the top-left corner, followed by one-cell-wide inverted corner paths.",
            "In case the input grid has the last inverted corner shape as empty (0), the last two inverted corner shapes in the output grid will also be empty (0); in such cases, fill them with the color of the top-left square block."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        
        grid_size = random.randint(4, 15)
        
        # Create task variables dictionary
        taskvars = {'grid_size': grid_size}
        
        # Generate 3-4 training examples
        num_train_examples = random.randint(3, 4)
        
        # Prepare the required configurations for diversity
        # 1. Example with empty last corner path
        # 2. Example with 3x3 square block
        # 3. Example with another square size (2x2)

        # Set up configurations for training examples
        train_configs = []
        
        # First example: Empty last corner path with random square size
        train_configs.append({
            'empty_last_corner': True,
            'square_size': random.choice([1, 2, 3]),
            'colors': self._generate_unique_colors(grid_size)
        })
        
        # Second example: 3x3 square block, may or may not have empty last corner
        train_configs.append({
            'empty_last_corner': False,  # Not empty to ensure diversity
            'square_size': 3,  # Ensure 3x3 block
            'colors': self._generate_unique_colors(grid_size)
        })
        
        # Third example: 2x2 square block (not 1x1 or 3x3), not empty last corner
        train_configs.append({
            'empty_last_corner': False,
            'square_size': 2,  # Ensure 2x2 block
            'colors': self._generate_unique_colors(grid_size)
        })
        
        # If we need 4 examples, add a fourth with 1x1 square
        if num_train_examples > 3:
            train_configs.append({
                'empty_last_corner': False,
                'square_size': 1,  # Ensure 1x1 block
                'colors': self._generate_unique_colors(grid_size)
            })
        
        # Shuffle the configs to avoid predictable ordering
        random.shuffle(train_configs)
        
        # Generate train examples based on configurations
        train_examples = []
        for config in train_configs:
            input_grid = self.create_input(taskvars, config)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test examples (2)
        test_examples = []
        for _ in range(2):
            gridvars = {
                'empty_last_corner': random.choice([True, False]),
                'square_size': random.randint(1, 3),
                'colors': self._generate_unique_colors(grid_size)
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            test_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def _generate_unique_colors(self, grid_size):
        """Generate unique colors for the square block and inverted corner paths"""
        # We need at least grid_size unique colors (1 for square, grid_size-1 for paths)
        colors = random.sample(range(1, 10), min(9, grid_size + 1))
        return colors
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        square_size = gridvars['square_size']
        colors = gridvars['colors']
        empty_last_corner = gridvars['empty_last_corner']
        
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create colored square in top-left corner
        square_color = colors[0]
        for r in range(square_size):
            for c in range(square_size):
                grid[r, c] = square_color
        
        # Create inverted corner paths
        for i in range(square_size, grid_size):  # Start after the square block
            # Skip the last corner if it should be empty
            if i == grid_size - 1 and empty_last_corner:
                continue
                
            # Select color for this path (cycling through available colors if needed)
            color_idx = (i - square_size) % len(colors[1:]) + 1  # Skip the square color
            path_color = colors[color_idx]
            
            # Create vertical part (top to bottom)
            for r in range(i + 1):
                grid[r, i] = path_color
                
            # Create horizontal part (right to left)
            for c in range(i, -1, -1):
                grid[i, c] = path_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        output_size = 2 * grid_size
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Find the colored cells in the input grid
        # Copy only colored cells to the output grid
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] != 0:
                    output_grid[r, c] = grid[r, c]
        
        # Find square block size and color
        square_objects = find_connected_objects(grid, diagonal_connectivity=True)
        top_left_square = None
        
        for obj in square_objects:
            coords = obj.coords
            if (0, 0) in coords:
                top_left_square = obj
                break
        
        square_color = None
        square_width = 0
        
        if top_left_square:
            square_color = list(top_left_square.colors)[0]
            # Calculate square width by finding max column
            square_width = max(c for r, c, _ in top_left_square.cells) + 1
        
        # Extract colors from inverted corner paths
        path_colors = []
        for col in range(square_width, grid_size):
            if grid[0, col] != 0:
                path_colors.append(grid[0, col])
        
        # Check if the last corner path is empty
        last_corner_empty = grid[grid_size-1, grid_size-1] == 0
        
        # Find the next available column to start new inverted corners
        # This is the first column after all colored cells
        next_col = 0
        for c in range(grid_size):
            if np.all(grid[:, c] == 0):
                next_col = c
                break
        else:
            # If all columns have at least one colored cell
            next_col = grid_size
        
        # First new inverted corner should have thickness matching square width
        for i in range(square_width):
            col = next_col + i
            # Create thick inverted L matching square width
            for r in range(col + 1):
                output_grid[r, col] = square_color
            for c in range(col, -1, -1):
                output_grid[col, c] = square_color
        
        next_col += square_width
        
        # Continue with one-cell-wide inverted corner paths
        color_index = 0
        while next_col < output_size:
            # Choose color (cycling through path colors)
            if color_index < len(path_colors):
                corner_color = path_colors[color_index]
                color_index += 1
            else:
                # If we run out of colors, cycle back to the beginning
                color_index = 0
                if path_colors:
                    corner_color = path_colors[color_index]
                    color_index += 1
                else:
                    corner_color = square_color
            
            # Special case: If last corner was empty, fill the last two corners with square color
            if last_corner_empty and next_col >= output_size - 2:
                corner_color = square_color
            
            # Create vertical part (top to bottom)
            for r in range(next_col + 1):
                output_grid[r, next_col] = corner_color
                
            # Create horizontal part (right to left)
            for c in range(next_col, -1, -1):
                output_grid[next_col, c] = corner_color
            
            next_col += 1
        
        return output_grid

