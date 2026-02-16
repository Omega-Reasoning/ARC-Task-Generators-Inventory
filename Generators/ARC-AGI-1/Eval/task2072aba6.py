from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random

class Task2072aba6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square of size {vars['rows']} x {vars['columns']}.",
            "Each input grid consists of a simple pattern which covers most of the grid and is of color {color('pattern_color')}."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is double the size of the input grid.",
            "The pattern now gets a checker board color filling in each of its cell namely {color('color1')} color and {color('color2')} color"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_t_shape(self, grid, pattern_color):
        """Create a T-shape pattern that covers the entire grid."""
        rows, cols = grid.shape
        
        # Vertical line of T (middle column)
        mid_col = cols // 2
        grid[:, mid_col] = pattern_color
        
        # Horizontal line of T (top row)
        grid[0, :] = pattern_color
        
        return grid

    def create_l_shape(self, grid, pattern_color):
        """Create an L-shape pattern."""
        rows, cols = grid.shape
        
        # Vertical line of L (leftmost column)
        grid[:, 0] = pattern_color
        
        # Horizontal line of L (bottom row)
        grid[rows - 1, :] = pattern_color
        
        return grid

    def create_plus_shape(self, grid, pattern_color):
        """Create a plus/cross shape pattern."""
        rows, cols = grid.shape
        
        # Vertical line (middle column)
        mid_col = cols // 2
        grid[:, mid_col] = pattern_color
        
        # Horizontal line (middle row)
        mid_row = rows // 2
        grid[mid_row, :] = pattern_color
        
        return grid

    def create_corner_l(self, grid, pattern_color):
        """Create an L-shape in a corner."""
        rows, cols = grid.shape
        corner = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
        
        if corner == 'top-left':
            grid[0, :] = pattern_color  # Top row
            grid[:, 0] = pattern_color  # Left column
        elif corner == 'top-right':
            grid[0, :] = pattern_color  # Top row
            grid[:, cols - 1] = pattern_color  # Right column
        elif corner == 'bottom-left':
            grid[rows - 1, :] = pattern_color  # Bottom row
            grid[:, 0] = pattern_color  # Left column
        else:  # bottom-right
            grid[rows - 1, :] = pattern_color  # Bottom row
            grid[:, cols - 1] = pattern_color  # Right column
        
        return grid

    def create_border_pattern(self, grid, pattern_color):
        """Create a border/frame pattern."""
        rows, cols = grid.shape
        
        # Top and bottom borders
        grid[0, :] = pattern_color
        grid[rows - 1, :] = pattern_color
        
        # Left and right borders
        grid[:, 0] = pattern_color
        grid[:, cols - 1] = pattern_color
        
        return grid

    def create_diagonal_cross(self, grid, pattern_color):
        """Create diagonal cross pattern."""
        rows, cols = grid.shape
        
        # Main diagonal (top-left to bottom-right)
        for i in range(min(rows, cols)):
            grid[i, i] = pattern_color
        
        # Anti-diagonal (top-right to bottom-left)
        for i in range(min(rows, cols)):
            grid[i, cols - 1 - i] = pattern_color
        
        return grid

    def create_h_shape(self, grid, pattern_color):
        """Create an H-shape pattern."""
        rows, cols = grid.shape
        
        # Left vertical line
        grid[:, 0] = pattern_color
        
        # Right vertical line
        grid[:, cols - 1] = pattern_color
        
        # Horizontal middle line
        mid_row = rows // 2
        grid[mid_row, :] = pattern_color
        
        return grid

    def create_z_shape(self, grid, pattern_color):
        """Create a Z-shape pattern."""
        rows, cols = grid.shape
        
        # Top horizontal line
        grid[0, :] = pattern_color
        
        # Bottom horizontal line
        grid[rows - 1, :] = pattern_color
        
        # Diagonal line (anti-diagonal)
        for i in range(min(rows, cols)):
            grid[i, cols - 1 - i] = pattern_color
        
        return grid

    def create_input(self, taskvars, gridvars):
        """Create an input grid with various pattern shapes."""
        rows = taskvars['rows']
        columns = taskvars['columns']
        pattern_color = taskvars['pattern_color']
        
        # Create base grid
        grid = np.zeros((rows, columns), dtype=int)
        
        # Choose pattern type randomly
        pattern_types = [
            self.create_t_shape,
            self.create_l_shape,
            self.create_plus_shape,
            self.create_corner_l,
            self.create_border_pattern,
            self.create_diagonal_cross,
            self.create_h_shape,
            self.create_z_shape
        ]
        
        # Select a random pattern
        pattern_func = random.choice(pattern_types)
        grid = pattern_func(grid, pattern_color)
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input to double size with checkerboard pattern."""
        rows = taskvars['rows']
        columns = taskvars['columns']
        color1 = taskvars['color1']
        color2 = taskvars['color2']
        pattern_color = taskvars['pattern_color']
        
        # Create output grid with double size
        output_grid = np.zeros((rows * 2, columns * 2), dtype=int)
        
        # For each cell in the input grid
        for r in range(rows):
            for c in range(columns):
                if grid[r, c] == pattern_color:
                    # This cell has the pattern, so we fill the corresponding 2x2 area
                    # in the output with checkerboard pattern
                    
                    # Calculate the 2x2 area in output grid
                    out_r_start = r * 2
                    out_c_start = c * 2
                    
                    # Create checkerboard pattern in this 2x2 area
                    # Top-left and bottom-right get color1
                    # Top-right and bottom-left get color2
                    output_grid[out_r_start, out_c_start] = color1           # Top-left
                    output_grid[out_r_start, out_c_start + 1] = color2       # Top-right
                    output_grid[out_r_start + 1, out_c_start] = color2       # Bottom-left
                    output_grid[out_r_start + 1, out_c_start + 1] = color1   # Bottom-right
                # If grid[r, c] == 0, the corresponding 2x2 area remains 0 (background)
        
        return output_grid

    def create_grids(self):
        """Create train and test grids with consistent variables."""
        # Generate random colors ensuring they're all different
        pattern_color = random.randint(1, 9)
        color1 = random.choice([c for c in range(1, 10) if c != pattern_color])
        color2 = random.choice([c for c in range(1, 10) if c != pattern_color and c != color1])
        
        # Generate grid size (smaller as suggested: 3-5)
        grid_size = random.randint(3, 5)  # Smaller grids for better pattern visibility
        
        # Store task variables
        taskvars = {
            'rows': grid_size,
            'columns': grid_size,
            'pattern_color': pattern_color,
            'color1': color1,
            'color2': color2,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Create training examples with different pattern types
        for i in range(num_train_examples):
            gridvars = {}  # No additional grid variables needed
            
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

