from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random

class Task19bb5febGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each input grid contains a single rectangular or square region filled with a background color {color('bg_color')}.",
            "Within that region you will find between 1 and 4 non-overlapping, well-spaced 2×2 sub-blocks, each in a unique color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is of size 2x2.",
            "Imagine dividing the input grid colored region into four quadrants.",
            "For each quadrant, if it contains one of the 2×2 sub-blocks, paint the corresponding cell in the output grid with that sub-block color.",
            "If a quadrant has no sub-block, leave its output cell empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        """Create an input grid with a colored region containing randomly placed 2x2 sub-blocks."""
        grid_size = gridvars['grid_size']
        bg_color = taskvars['bg_color']
        
        # Create base grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create rectangular colored region (must be large enough for quadrants and spacing)
        min_region_size = 8  # At least 8x8 to have proper quadrants with spacing
        max_region_size = min(grid_size - 2, 12)  # Leave some border
        
        region_width = random.randint(min_region_size, max_region_size)
        region_height = random.randint(min_region_size, max_region_size)
        
        # Center the region
        start_row = (grid_size - region_height) // 2
        start_col = (grid_size - region_width) // 2
        
        # Fill the region with background color
        grid[start_row:start_row + region_height, start_col:start_col + region_width] = bg_color
        
        # Define quadrants within the region with proper boundaries
        mid_row = start_row + region_height // 2
        mid_col = start_col + region_width // 2
        
        quadrants = [
            (start_row, mid_row, start_col, mid_col),           # Top-left
            (start_row, mid_row, mid_col, start_col + region_width),     # Top-right
            (mid_row, start_row + region_height, start_col, mid_col),    # Bottom-left
            (mid_row, start_row + region_height, mid_col, start_col + region_width)  # Bottom-right
        ]
        
        # Randomly choose 1-4 quadrants to place 2x2 sub-blocks
        num_blocks = random.randint(1, 4)
        selected_quadrants = random.sample(range(4), num_blocks)
        
        # Available colors (excluding background and 0)
        available_colors = [c for c in range(1, 10) if c != bg_color]
        block_colors = random.sample(available_colors, num_blocks)
        
        # Place 2x2 blocks randomly within selected quadrants with spacing
        placed_blocks = []
        
        for i, quad_idx in enumerate(selected_quadrants):
            r1, r2, c1, c2 = quadrants[quad_idx]
            
            # Ensure quadrant is large enough for a 2x2 block
            quad_height = r2 - r1
            quad_width = c2 - c1
            
            if quad_height >= 2 and quad_width >= 2:
                # Find valid positions for 2x2 block within this quadrant
                max_r = r2 - 2
                max_c = c2 - 2
                
                # Try to place block with spacing from other blocks
                attempts = 0
                placed = False
                
                while attempts < 20 and not placed:
                    block_r = random.randint(r1, max_r)
                    block_c = random.randint(c1, max_c)
                    
                    # Check if this position has enough spacing from other blocks
                    has_spacing = True
                    for prev_r, prev_c in placed_blocks:
                        # Ensure at least 1 cell spacing between blocks
                        if (abs(block_r - prev_r) < 3 or abs(block_c - prev_c) < 3) and \
                           not (abs(block_r - prev_r) >= 3 or abs(block_c - prev_c) >= 3):
                            has_spacing = False
                            break
                    
                    if has_spacing:
                        # Place 2x2 block
                        grid[block_r:block_r+2, block_c:block_c+2] = block_colors[i]
                        placed_blocks.append((block_r, block_c))
                        placed = True
                    
                    attempts += 1
                
                # If we couldn't place with spacing, place anyway (fallback)
                if not placed:
                    block_r = random.randint(r1, max_r)
                    block_c = random.randint(c1, max_c)
                    grid[block_r:block_r+2, block_c:block_c+2] = block_colors[i]
                    placed_blocks.append((block_r, block_c))
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input to 2x2 output based on quadrant contents."""
        bg_color = taskvars['bg_color']
        
        # Find the colored region (background colored area)
        bg_mask = grid == bg_color
        if not np.any(bg_mask):
            return np.zeros((2, 2), dtype=int)
        
        # Get bounding box of colored region
        rows, cols = np.where(bg_mask)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        # Calculate quadrant boundaries
        region_height = max_row - min_row + 1
        region_width = max_col - min_col + 1
        mid_row = min_row + region_height // 2
        mid_col = min_col + region_width // 2
        
        quadrants = [
            (min_row, mid_row, min_col, mid_col),                    # Top-left -> output[0,0]
            (min_row, mid_row, mid_col, max_col + 1),               # Top-right -> output[0,1]
            (mid_row, max_row + 1, min_col, mid_col),               # Bottom-left -> output[1,0]
            (mid_row, max_row + 1, mid_col, max_col + 1)            # Bottom-right -> output[1,1]
        ]
        
        output = np.zeros((2, 2), dtype=int)
        
        # Check each quadrant for 2x2 blocks
        for quad_idx, (r1, r2, c1, c2) in enumerate(quadrants):
            # Scan this quadrant for 2x2 blocks
            for r in range(r1, r2 - 1):
                for c in range(c1, c2 - 1):
                    if r + 2 <= r2 and c + 2 <= c2:
                        block_region = grid[r:r+2, c:c+2]
                        unique_colors = np.unique(block_region)
                        
                        # Check if this is a 2x2 block of the same non-background color
                        if len(unique_colors) == 1 and unique_colors[0] != bg_color and unique_colors[0] != 0:
                            output_row = quad_idx // 2
                            output_col = quad_idx % 2
                            output[output_row, output_col] = unique_colors[0]
                            break
                else:
                    continue
                break
        
        return output

    def create_grids(self):
        """Create train and test grids with consistent variables."""
        # Generate background color
        bg_color = random.randint(1, 9)
        
        # Store task variables
        taskvars = {
            'bg_color': bg_color
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate grid sizes - need to be large enough for proper quadrants
        min_size = 10
        max_size = 16
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': all_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': all_sizes[-1]}
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

# Create generator instance
generator = Task19bb5febGenerator()