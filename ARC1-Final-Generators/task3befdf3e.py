from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import create_object, Contiguity, retry

class Task3befdf3eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain between 1 and 4 colored square blocks, each with a frame and an interior.",
            "Each square block consists of two unique colors: the first color forms the interior block, while the second color creates a one-cell-wide frame around it.",
            "The square blocks can be either 3x3 or 4x4 in size; for a 3x3 block, the interior is 1x1, while for a 4x4 block, the interior is a 2x2 square.",
            "If there are multiple colored blocks in a grid, they share the same two colors (interior and frame) but may vary in size; when there are exactly two blocks they will have different sizes.",
            "The square blocks are positioned so that if the block is 3x3, it has a one-cell wide frame made of empty (0) cells around it, and if it is 4x4, it has a two-cell wide frame made of empty (0) cells around it."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and swapping the two colors in each square block, replacing the interior block color with the original frame color from the input and the frame color with the original interior block color from the input grids.",
            "Next, depending on the size of the interior block, each existing square block is expanded by adding one additional rectangular extension to each of its four sides.",
            "The additional extensions are sized 1x3 if the interior block is 1x1, and 2x4 if the interior block is 2x2.",
            "The newly added rectangular extensions are colored with the original frame color from the input grids of their respective blocks.",
            "The four rectangular extensions are added to the top, bottom, left, and right of each square block, directly connected and aligned with the existing frame of the respective block."
            ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables: choose a train grid size and a different test grid size
        train_size = random.randint(12, 30)
        test_size = random.randint(12, 30)
        # ensure test grid size is different from train grid size
        while test_size == train_size:
            test_size = random.randint(12, 30)

        taskvars = {'grid_size': train_size}

        # Determine how many train examples to create (3-5)
        num_train = random.randint(3, 5)

        # Decide number of blocks for train examples (1..4), ensure test uses a different count
        train_block_count = random.randint(1, 4)
        # choose a test block count different from train_block_count
        possible_test_counts = [n for n in range(1, 5) if n != train_block_count]
        test_block_count = random.choice(possible_test_counts)
        
        train_pairs = []
        used_colors = set()
        
        for i in range(num_train):
            # All train examples should have the same number of blocks
            block_count = train_block_count
            
            # Generate unique colors for this example
            colors = self._get_unique_colors(used_colors)
            used_colors.update(colors)
            
            # Create grid variables for this specific example
            gridvars = {
                'block_count': block_count,
                'colors': colors
            }
            
            # Create input grid and its corresponding output
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create one test example: must have a different number of blocks than the train examples
        test_colors = self._get_unique_colors(used_colors)
        test_gridvars = {
            'block_count': test_block_count,
            'colors': test_colors
        }

        # For the test example use a taskvars copy where grid_size is the test size
        test_taskvars = taskvars.copy()
        test_taskvars['grid_size'] = test_size

        test_input = self.create_input(test_taskvars, test_gridvars)
        test_output = self.transform_input(test_input, test_taskvars)
        
        test_pairs = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_pairs,
            'test': test_pairs
        }
    
    def _get_unique_colors(self, used_colors):
        """Generate two unique colors not previously used"""
        available_colors = [c for c in range(1, 10) if c not in used_colors]
        if len(available_colors) < 2:
            # Reset if we've used all colors
            available_colors = list(range(1, 10))
        
        return random.sample(available_colors, 2)
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        block_count = gridvars.get('block_count', 1)
        interior_color, frame_color = gridvars['colors']

        # Decide on block sizes. Blocks are 3x3 or 4x4.
        if block_count == 1:
            block_sizes = [random.choice([3, 4])]
        elif block_count == 2:
            # When exactly two blocks, prefer different sizes (3 and 4)
            block_sizes = [3, 4]
            random.shuffle(block_sizes)
        else:
            # For 3 or 4 blocks, choose sizes randomly (allow repeats) from [3,4]
            block_sizes = [random.choice([3, 4]) for _ in range(block_count)]
        
        # Track placed blocks for separation validation
        placed_blocks = []  # List of (top_left_r, top_left_c, bottom_right_r, bottom_right_c)
        
        # Place each block
        for block_size in block_sizes:
            if block_size == 3:
                # For 3x3 block, interior is 1x1
                interior_size = 1
                min_margin = 1  # Minimum margin from edge
            else:  # block_size == 4
                # For 4x4 block, interior is 2x2
                interior_size = 2
                min_margin = 2  # Minimum margin from edge
            
            # Place the block with proper spacing
            placed = False
            max_attempts = 500  # Increase attempts for more complex placement
            attempts = 0
            
            while not placed and attempts < max_attempts:
                # Calculate valid positions
                min_row = min_margin
                max_row = grid_size - block_size - min_margin
                min_col = min_margin
                max_col = grid_size - block_size - min_col
                
                if max_row < min_row or max_col < min_col:
                    attempts += 1
                    continue  # Grid too small for this block
                
                row = random.randint(min_row, max_row)
                col = random.randint(min_col, max_col)
                
                # Define this block's boundaries
                block_top = row
                block_bottom = row + block_size - 1
                block_left = col
                block_right = col + block_size - 1
                
                # Check if this location works
                valid = True
                
                # Check for proper separation from other blocks (require at least 2 empty rows/cols)
                # Use a padded-box non-overlap test: pad previous block by `pad` cells
                # and ensure this block's bounding box does not intersect that padded box.
                pad = 2
                for prev_top, prev_left, prev_bottom, prev_right in placed_blocks:
                    padded_prev_top = prev_top - pad
                    padded_prev_left = prev_left - pad
                    padded_prev_bottom = prev_bottom + pad
                    padded_prev_right = prev_right + pad

                    # If this block's bounding box intersects the padded previous box -> invalid
                    if not (
                        block_top > padded_prev_bottom or
                        block_bottom < padded_prev_top or
                        block_left > padded_prev_right or
                        block_right < padded_prev_left
                    ):
                        valid = False
                        break
                
                if valid:
                    # Place the frame (outer rectangle)
                    for r in range(row, row + block_size):
                        for c in range(col, col + block_size):
                            if (r == row or r == row + block_size - 1 or 
                                c == col or c == col + block_size - 1):
                                grid[r, c] = frame_color
                    
                    # Place the interior (inner rectangle)
                    interior_start_r = row + (block_size - interior_size) // 2
                    interior_start_c = col + (block_size - interior_size) // 2
                    
                    for r in range(interior_start_r, interior_start_r + interior_size):
                        for c in range(interior_start_c, interior_start_c + interior_size):
                            grid[r, c] = interior_color
                    
                    # Record this block's boundaries
                    placed_blocks.append((block_top, block_left, block_bottom, block_right))
                    placed = True
                else:
                    attempts += 1
            
            if not placed:
                # If we couldn't place the block after max attempts, reset and try again
                grid = np.zeros((grid_size, grid_size), dtype=int)
                return self.create_input(taskvars, gridvars)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Find all distinct blocks in the grid
        blocks = self._find_blocks(grid)
        
        # Process each block
        for block in blocks:
            frame_coords = block['frame_coords']
            interior_coords = block['interior_coords']
            frame_color = block['frame_color']
            interior_color = block['interior_color']
            block_size = block['size']
            
            # 1. Swap colors
            for r, c in frame_coords:
                output_grid[r, c] = interior_color
                
            for r, c in interior_coords:
                output_grid[r, c] = frame_color
            
            # 2. Determine extension size based on interior size
            if block_size == 3:  # 3x3 block with 1x1 interior
                extension_width = 3
                extension_height = 1
            else:  # 4x4 block with 2x2 interior
                extension_width = 4
                extension_height = 2
            
            # 3. Get the frame's bounding box
            min_r = min(r for r, c in frame_coords)
            min_c = min(c for r, c in frame_coords)
            max_r = max(r for r, c in frame_coords)
            max_c = max(c for r, c in frame_coords)
            
            # 4. Add extensions with the color of the new interior (which was the frame color)
            extension_color = frame_color
            
            # Top extension
            for r in range(min_r - extension_height, min_r):
                for c in range(min_c, min_c + extension_width):
                    if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                        output_grid[r, c] = extension_color
            
            # Bottom extension
            for r in range(max_r + 1, max_r + 1 + extension_height):
                for c in range(min_c, min_c + extension_width):
                    if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                        output_grid[r, c] = extension_color
            
            # Left extension
            for r in range(min_r, min_r + extension_width):
                for c in range(min_c - extension_height, min_c):
                    if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                        output_grid[r, c] = extension_color
            
            # Right extension
            for r in range(min_r, min_r + extension_width):
                for c in range(max_c + 1, max_c + 1 + extension_height):
                    if 0 <= r < output_grid.shape[0] and 0 <= c < output_grid.shape[1]:
                        output_grid[r, c] = extension_color
        
        return output_grid
    
    def _find_blocks(self, grid):
        """
        Find all rectangular blocks in the grid, each with a frame and interior.
        Returns a list of dicts with information about each block.
        """
        blocks = []
        processed_cells = set()
        
        # Find all non-background cells that haven't been processed
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0 and (r, c) not in processed_cells:
                    # Check if this cell is part of a valid rectangular block
                    block_info = self._identify_block(grid, r, c)
                    
                    if block_info:
                        blocks.append(block_info)
                        # Mark all cells in this block as processed
                        for cell in block_info['frame_coords']:
                            processed_cells.add(cell)
                        for cell in block_info['interior_coords']:
                            processed_cells.add(cell)
        
        return blocks
    
    def _identify_block(self, grid, start_r, start_c):
        """
        Identify a rectangular block starting at the given position.
        Returns frame_coords, interior_coords, frame_color, interior_color,
        and size (3 or 4) if a valid block is found.
        """
        # Find boundaries of potential rectangular block
        color = grid[start_r, start_c]
        r, c = start_r, start_c
        
        # Find top-left corner of block
        while r > 0 and grid[r-1, c] == color:
            r -= 1
        while c > 0 and grid[r, c-1] == color:
            c -= 1
        
        top_left_r, top_left_c = r, c
        
        # Find bottom-right corner of block
        rows, cols = grid.shape
        while r+1 < rows and grid[r+1, c] == color:
            r += 1
        while c+1 < cols and grid[r, c+1] == color:
            c += 1
        
        bottom_right_r, bottom_right_c = r, c
        
        # Check if this is a potential rectangular block
        height = bottom_right_r - top_left_r + 1
        width = bottom_right_c - top_left_c + 1
        
        if height != width:
            return None  # Not a square block
        
        if height not in [3, 4]:
            return None  # Not a 3x3 or 4x4 block
        
        # Check if it has a frame structure
        frame_coords = set()
        interior_coords = set()
        frame_color = None
        interior_color = None
        
        # First, identify the frame
        for r in range(top_left_r, bottom_right_r + 1):
            for c in range(top_left_c, bottom_right_c + 1):
                if (r == top_left_r or r == bottom_right_r or 
                    c == top_left_c or c == bottom_right_c):
                    frame_coords.add((r, c))
                    if frame_color is None:
                        frame_color = grid[r, c]
                    elif grid[r, c] != frame_color:
                        return None  # Not a consistent frame color
        
        # Now, check if there's an interior with a different color
        interior_size = 1 if height == 3 else 2
        interior_start_r = top_left_r + (height - interior_size) // 2
        interior_start_c = top_left_c + (width - interior_size) // 2
        
        for r in range(interior_start_r, interior_start_r + interior_size):
            for c in range(interior_start_c, interior_start_c + interior_size):
                interior_coords.add((r, c))
                if interior_color is None:
                    interior_color = grid[r, c]
                elif grid[r, c] != interior_color:
                    return None  # Not a consistent interior color
        
        # Check that interior and frame have different colors
        if interior_color == frame_color:
            return None
        
        # Make sure all other cells in the block are empty
        for r in range(top_left_r, bottom_right_r + 1):
            for c in range(top_left_c, bottom_right_c + 1):
                if (r, c) not in frame_coords and (r, c) not in interior_coords:
                    if grid[r, c] != 0:
                        return None  # Unexpected value in block
        
        return {
            'frame_coords': frame_coords,
            'interior_coords': interior_coords,
            'frame_color': frame_color,
            'interior_color': interior_color,
            'size': height
        }


