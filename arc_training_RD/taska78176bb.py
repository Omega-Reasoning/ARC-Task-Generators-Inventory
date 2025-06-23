from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, retry

class Taska78176bbGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "The input grid has a diagonal line of {color('diagonal_color')} colored cells running from the top-left to bottom-right (or another consistent diagonal).",
            "A region is of {color('region_color')} color,overlapping the diagonal line .",
            "The overlapping region can be of shape inverted right-angle, with its right angle at the top-left or it can grow downward and to the right forming a staircase-like shape or it just overlaps the diagonal line(but may not be centered on it)."
        ]
        
        transformation_reasoning_chain = [
            "It copies the original diagonal line of {color('diagonal_color')}color.",
            "The overlapping regions disappear and are filled with empty (0) cells.",
            "For each diagonal cell under the right-angle triangle, copy it one row below in the same column."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with diagonal line and overlapping region."""
        grid_size = gridvars['grid_size']
        diagonal_color = taskvars['diagonal_color']
        region_color = taskvars['region_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Draw diagonal line from top-left to bottom-right
        for i in range(grid_size):
            grid[i, i] = diagonal_color
        
        # Create overlapping region (right-angle triangle or staircase)
        region_type = random.choice(["right_angle", "staircase", "offset_overlap"])
        region_size = random.randint(2, min(5, grid_size - 2))  # Size of the region (2-5 or up to grid size)
        
        # Starting point on the diagonal
        start_row = random.randint(0, grid_size - region_size - 1)
        start_col = start_row  # Since we're on the diagonal
        
        if region_type == "right_angle":
            # Inverted right-angle triangle growing downward and to the right
            for i in range(region_size):
                for j in range(i + 1):
                    r = start_row + i
                    c = start_col + j
                    if 0 <= r < grid_size and 0 <= c < grid_size:
                        grid[r, c] = region_color
        
        elif region_type == "staircase":
            # Staircase shape growing downward and to the right
            for i in range(region_size):
                r = start_row + i
                for j in range(start_col, start_col + i + 1):
                    if 0 <= r < grid_size and 0 <= j < grid_size:
                        grid[r, j] = region_color
        
        else:  # offset_overlap
            # Simple rectangle overlapping the diagonal but offset
            offset = random.choice([-1, 0, 1])  # Offset from diagonal
            width = random.randint(2, 4)  # Width of the rectangle
            height = random.randint(2, 4)  # Height of the rectangle
            
            for i in range(height):
                for j in range(width):
                    r = start_row + i
                    c = start_col + j + offset
                    if 0 <= r < grid_size and 0 <= c < grid_size:
                        grid[r, c] = region_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by copying diagonal and shifting covered diagonal cells down."""
        # Create a copy of the input grid for the output
        output_grid = np.zeros_like(grid)
        
        # Get colors from taskvars
        diagonal_color = taskvars['diagonal_color']
        region_color = taskvars['region_color']
        
        # Identify unique colors in the input
        unique_colors = set(np.unique(grid)) - {0}
        if len(unique_colors) != 2:
            # Fallback if we don't have exactly two colors
            return grid
        
        # Find the diagonal cells
        grid_size = grid.shape[0]
        diagonal_cells = [(i, i) for i in range(grid_size) if grid[i, i] != 0]
        
        # Determine which color is the diagonal color (the one that appears most on the diagonal)
        color_counts = {}
        for r, c in diagonal_cells:
            color = grid[r, c]
            color_counts[color] = color_counts.get(color, 0) + 1
        
        if color_counts:
            actual_diagonal_color = max(color_counts.items(), key=lambda x: x[1])[0]
        else:
            actual_diagonal_color = diagonal_color
        
        # 1. Copy the original diagonal line
        for i in range(grid_size):
            if grid[i, i] == actual_diagonal_color:
                output_grid[i, i] = actual_diagonal_color
        
        # 2. Identify overlapping cells (diagonal cells that have the region color)
        overlapping_cells = []
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] == region_color and grid[r, r] == actual_diagonal_color and r <= c:
                    # This is a cell in the region that covers a diagonal cell
                    overlapping_cells.append((r, c))
        
        # 3. For each diagonal cell under the overlapping region, copy it one row below
        for r, c in diagonal_cells:
            # Check if this diagonal cell is under the overlapping region
            is_under_region = False
            for or_row, or_col in overlapping_cells:
                if r == or_row and c == or_col:
                    is_under_region = True
                    break
            
            if is_under_region and r + 1 < grid_size:
                # Copy this diagonal cell one row below
                output_grid[r + 1, c] = actual_diagonal_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Choose colors between 1-9
        diagonal_color = random.randint(1, 9)
        region_color = random.randint(1, 9)
        while region_color == diagonal_color:
            region_color = random.randint(1, 9)
        
        # Store task variables
        taskvars = {
            'diagonal_color': diagonal_color,
            'region_color': region_color
        }
        
        # Create 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate different grid sizes for each example
        grid_sizes = [random.randint(5, 20) for _ in range(num_train_examples + 1)]
        
        for i in range(num_train_examples):
            gridvars = {'grid_size': grid_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': grid_sizes[-1]}
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