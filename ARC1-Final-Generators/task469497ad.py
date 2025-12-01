from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring, retry
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task469497adGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a 2x2 square block and a completely filled last row and last column.",
            "The last row and last column have identical color arrangements, with all same-colored cells forming continuous connections along both the last row and last column.",
            "The color of the square block is different from the colors used in the last row and last column.",
            "The last row and column can contain up to three different colors.",
            "The square block is positioned such that it is never connected to the last row or last column."]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by identifying the number of different colors used in the input grid, denoted as n.",
            "The output grids are of size (nx{vars['grid_size']})x(nx{vars['grid_size']}).",
            "Each cell in the input grid is expanded in both x and y dimensions, preserving the original shape and pattern of the input grid.",
            "This results in n number of last rows and n number of last columns being colored, while the 2x2 block expands into an (nx2)x(nx2) block.",
            "Lastly, {color('fill_color')} cells are added to all available corners of the (nx2)x(nx2) block and extended diagonally until they encounter another colored cell or reach the matrix edge."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(5, 7),
            'fill_color': random.randint(1, 9)
        }
        
        # Generate 3-4 training examples and 1 test example
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        # Special case: 2x2 block in center with monochromatic border
        center_example = self._create_center_example(taskvars)
        train_examples.append(center_example)
        
        # Generate remaining training examples
        for _ in range(num_train_examples - 1):
            gridvars = self._create_random_gridvars(taskvars)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_gridvars = self._create_random_gridvars(taskvars)
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def _create_center_example(self, taskvars):
        """Create the special example with the 2x2 block in the center and monochromatic border"""
        grid_size = taskvars['grid_size']
        
        # Choose distinct colors for block and border
        all_colors = list(range(1, 10))
        all_colors.remove(taskvars['fill_color'])  # Don't use fill_color for borders
        border_color = random.choice(all_colors)
        
        remaining_colors = all_colors.copy()
        remaining_colors.remove(border_color)
        block_color = random.choice(remaining_colors)
        
        # For the center example, place the block in the exact center
        center_row = (grid_size - 2) // 2
        center_col = (grid_size - 2) // 2
        
        gridvars = {
            'block_color': block_color,
            'border_colors': [border_color],
            'block_position': (center_row, center_col)
        }
        
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        
        return {'input': input_grid, 'output': output_grid}
    
    def _create_random_gridvars(self, taskvars):
        """Create random grid variables for diverse examples with balanced block positioning"""
        grid_size = taskvars['grid_size']
        
        # Choose 1-3 border colors and a different block color
        all_colors = list(range(1, 10))
        all_colors.remove(taskvars['fill_color'])  # Don't use fill_color for borders
        
        num_border_colors = random.randint(1, min(3, len(all_colors)-1))  # Ensure at least one color left for block
        border_colors = random.sample(all_colors, num_border_colors)
        
        remaining_colors = [c for c in all_colors if c not in border_colors]
        block_color = random.choice(remaining_colors)
        
        # Determine block position with equal spacing constraint
        # Now we follow these rules:
        # 1. For equal rows above/below: start from first column (col=0)
        # 2. For equal columns left/right: start from first row (row=0)
        # 3. Random chance to have both criteria (place at (0,0))
        
        positioning_type = random.choice(["vertical", "horizontal", "both"])
        
        if positioning_type == "vertical" or positioning_type == "both":
            # Equal rows above and below (vertically centered)
            row = (grid_size - 2) // 2  # Center vertically
            col = 0  # Start from first column when vertically balanced
        elif positioning_type == "horizontal":
            # Equal columns left and right (horizontally centered)
            row = 0  # Start from first row when horizontally balanced
            col = (grid_size - 2) // 2  # Center horizontally
        else:
            # This shouldn't be reached but as a fallback
            row = random.randint(0, grid_size - 3)
            col = random.randint(0, grid_size - 3)
        
        # Ensure we're not touching the last row or column
        if row + 2 >= grid_size - 1:
            row = grid_size - 3
        if col + 2 >= grid_size - 1:
            col = grid_size - 3
        
        return {
            'block_color': block_color,
            'border_colors': border_colors,
            'block_position': (row, col)
        }
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        
        # Initialize empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Add 2x2 square block
        block_color = gridvars['block_color']
        row, col = gridvars['block_position']
        grid[row:row+2, col:col+2] = block_color
        
        # Create border pattern for last row and column
        border_colors = gridvars['border_colors']
        
        # If multiple border colors, create segments with continuous same-colored cells
        if len(border_colors) > 1:
            # Create a pattern that ensures continuity of same colors
            border_pattern = self._create_continuous_color_pattern(grid_size, border_colors)
            
            # Apply pattern to last row and column
            grid[grid_size-1, :] = border_pattern
            grid[:, grid_size-1] = border_pattern
        else:
            # Simple case: one color for both last row and column
            grid[grid_size-1, :] = border_colors[0]
            grid[:, grid_size-1] = border_colors[0]
        
        return grid
    
    def _create_continuous_color_pattern(self, length, colors):
        """Create a pattern where same colors form continuous segments"""
        pattern = np.zeros(length, dtype=int)
        
        # Determine minimum segment length to ensure continuity
        min_segment = 2
        max_segments = length // min_segment
        
        # Fix for the error: ensure we have a valid range for num_segments
        if len(colors) >= max_segments:
            num_segments = max_segments
        else:
            num_segments = random.randint(len(colors), max_segments)
        
        # Create segments with random lengths
        segment_boundaries = sorted(random.sample(range(1, length), num_segments-1))
        segment_boundaries = [0] + segment_boundaries + [length]
        
        segments = []
        for i in range(len(segment_boundaries)-1):
            segments.append(segment_boundaries[i+1] - segment_boundaries[i])
        
        # Assign colors to segments ensuring adjacent segments have different colors
        color_assignments = []
        
        for i in range(len(segments)):
            if i == 0:
                # For the first segment, just pick a random color
                color = random.choice(colors)
            else:
                # For subsequent segments, avoid the previous color if possible
                available_colors = [c for c in colors if (len(colors) == 1 or c != color_assignments[-1])]
                if not available_colors:  # Fallback if only one color is available
                    available_colors = colors
                color = random.choice(available_colors)
            
            color_assignments.append(color)
        
        # Fill the pattern
        pos = 0
        for segment_length, color in zip(segments, color_assignments):
            pattern[pos:pos + segment_length] = color
            pos += segment_length
        
        return pattern
    
    def transform_input(self, grid, taskvars):
        # Identify the number of unique colors in the input grid
        unique_colors = np.unique(grid)
        unique_colors = unique_colors[unique_colors != 0]  # Exclude background
        n = len(unique_colors)
        
        grid_size = taskvars['grid_size']
        fill_color = taskvars['fill_color']
        
        # Create expanded output grid
        output_size = n * grid_size
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Expand each cell from input to an nxn block in output
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] != 0:
                    output_grid[n*r:n*(r+1), n*c:n*(c+1)] = grid[r, c]
        
        # Find the expanded 2x2 block (now n*2 x n*2)
        # First identify where the block was in the input grid
        for r in range(grid_size-1):
            for c in range(grid_size-1):
                # Check for a 2x2 block
                if (grid[r,c] == grid[r,c+1] == grid[r+1,c] == grid[r+1,c+1] and 
                    grid[r,c] != 0 and
                    grid[r,c] not in grid[grid_size-1,:] and  # Not in last row
                    grid[r,c] not in grid[:,grid_size-1]):    # Not in last column
                    
                    # Found the block, now identify its expanded boundaries
                    block_top = n * r
                    block_left = n * c
                    block_bottom = n * (r + 2) - 1
                    block_right = n * (c + 2) - 1
                    
                    # Add diagonal lines from the four corners of the expanded block
                    
                    # Define the four corners of the expanded block
                    corners = [
                        (block_top, block_left),       # Top-left
                        (block_top, block_right),      # Top-right
                        (block_bottom, block_left),    # Bottom-left
                        (block_bottom, block_right)    # Bottom-right
                    ]
                    
                    # For each corner, draw a diagonal line
                    for corner_r, corner_c in corners:
                        # Determine diagonal direction (away from the block)
                        dr = -1 if corner_r == block_top else 1
                        dc = -1 if corner_c == block_left else 1
                        
                        # Start at the corner (already part of the block)
                        r_pos, c_pos = corner_r, corner_c
                        
                        # First step away from the corner (the first cell of the diagonal line)
                        r_pos += dr
                        c_pos += dc
                        
                        # Continue extending diagonally until reaching a non-empty cell or the grid boundary
                        while 0 <= r_pos < output_size and 0 <= c_pos < output_size:
                            if output_grid[r_pos, c_pos] == 0:  # Only fill empty cells
                                output_grid[r_pos, c_pos] = fill_color
                                # Move to next diagonal position
                                r_pos += dr
                                c_pos += dc
                            else:
                                # Stop when encountering a non-empty cell
                                break
                    
                    # Only process the first 2x2 block found
                    return output_grid
        
        # If we reach here, something went wrong - no block was found
        # This shouldn't happen with proper input generation
        return output_grid

