from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task543a7ed5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "The background of the grid is completely filled with {color('background')} color.",
            "They contain several rectangular blocks of {color('block_color')} color.",
            "One of the blocks is a one-cell wide {color('block_color')} frame with all interior cells filled with {color('background')} color.",
            "The remaining blocks may be completely filled with {color('block_color')} color or may have a one-cell wide {color('background')} colored rectangle in the interior of the block.",
            "The {color('block_color')} blocks are always positioned within the interior of the grid and never overlap the grid border.",
            "All blocks should be completely separated from each other by at least two rows or columns."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are constructed by copying the input grids and identifying the rectangular blocks of {color('block_color')} color, which may or may not be completely filled with {color('block_color')} color but always have at least a {color('block_color')} frame around them.",
            "Once identified, add a one-cell wide {color('frame_color')} colored frame around each block.",
            "If any of the rectangular blocks consist of {color('background')} cells within their interior, change the color of those {color('background')} cells to {color('new_color')} within the framed area."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize variables with constraints
        grid_size = random.randint(14, 22)  # Reasonable size for 3 blocks with separation
        
        # Choose distinct colors for block, background, frame, and new interior
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        block_color = available_colors[0]
        background = available_colors[1]
        frame_color = available_colors[2]
        new_color = available_colors[3]
        
        taskvars = {
            'grid_size': grid_size,
            'block_color': block_color,
            'background': background,
            'frame_color': frame_color,
            'new_color': new_color
        }
        
        # Decide maximum number of blocks allowed based on grid size (between 2 and 5)
        if grid_size < 15:
            max_blocks = 2
        elif grid_size < 17:
            max_blocks = 3
        elif grid_size < 19:
            max_blocks = 4
        else:
            max_blocks = 5

        # Create 3 training grids and 1 test grid; num_blocks may vary per grid
        train_grids = []

        # First training grid: interior_blocks = 0, choose a random number of blocks
        num_blocks_1 = random.randint(2, max_blocks)
        grid_1 = self.create_input(taskvars, {'interior_blocks': 0, 'num_blocks': num_blocks_1})
        output_1 = self.transform_input(grid_1, taskvars)
        train_grids.append({'input': grid_1, 'output': output_1})

        # Second training grid: one interior block if possible
        num_blocks_2 = random.randint(2, max_blocks)
        interior_2 = 1 if num_blocks_2 > 1 else 0
        grid_2 = self.create_input(taskvars, {'interior_blocks': interior_2, 'num_blocks': num_blocks_2})
        output_2 = self.transform_input(grid_2, taskvars)
        train_grids.append({'input': grid_2, 'output': output_2})

        # Third training grid: two interior blocks (bounded by available blocks)
        num_blocks_3 = random.randint(2, max_blocks)
        interior_3 = min(2, max(0, num_blocks_3 - 1))
        grid_3 = self.create_input(taskvars, {'interior_blocks': interior_3, 'num_blocks': num_blocks_3})
        output_3 = self.transform_input(grid_3, taskvars)
        train_grids.append({'input': grid_3, 'output': output_3})

        # Create test grid: random configuration
        num_blocks_t = random.randint(2, max_blocks)
        interior_t = random.randint(0, max(0, num_blocks_t - 1))
        test_grid = self.create_input(taskvars, {'interior_blocks': interior_t, 'num_blocks': num_blocks_t})
        test_output = self.transform_input(test_grid, taskvars)

        return taskvars, {
            'train': train_grids,
            'test': [{'input': test_grid, 'output': test_output}]
        }
    
    def create_input(self, taskvars, gridvars):
        """
        Create an input grid with a variable number of rectangular blocks (2..5) according to the specifications
        """
        grid_size = taskvars['grid_size']
        block_color = taskvars['block_color']
        background = taskvars['background']
        
        # Number of blocks that have interior gaps
        interior_blocks = gridvars.get('interior_blocks', random.randint(0, 2))

        # Number of blocks to place (allow 2..5 depending on gridvars or grid_size)
        num_blocks = gridvars.get('num_blocks', None)
        if num_blocks is None:
            # Derive reasonable max based on grid size
            if grid_size < 15:
                max_blocks = 2
            elif grid_size < 17:
                max_blocks = 3
            elif grid_size < 19:
                max_blocks = 4
            else:
                max_blocks = 5
            num_blocks = random.randint(2, max_blocks)
        
        # Initialize grid with background color
        grid = np.full((grid_size, grid_size), background, dtype=int)
        
        def _generate_block_sizes():
            """Generate num_blocks different block sizes"""
            # Minimum size needed for a block (to potentially have interior)
            min_size = 5
            
            # Maximum size (to ensure blocks can fit with separation)
            effective_size = grid_size - 2  # Account for 1-cell border
            # Heuristic: allow more smaller blocks on larger grids
            max_size = max(min_size + 2, max( (effective_size - 4) // max(1, num_blocks - 1), min_size))
            
            if max_size <= min_size:
                max_size = min_size + 1  # Ensure a valid range
            
            # Generate num_blocks sets of block dimensions ensuring they're different
            sizes = []
            size_attempts = 0
            
            while len(sizes) < num_blocks and size_attempts < 200:
                size_attempts += 1
                
                # Range of dimensions for this block
                height = random.randint(min_size, max_size)
                width = random.randint(min_size, max_size)
                
                # Ensure blocks have different dimensions
                new_size = (height, width)
                if new_size not in sizes:
                    sizes.append(new_size)
            
            # If we couldn't get enough different sizes, make some adjustments
            if len(sizes) < num_blocks:
                # Add fixed sizes that are different
                base = (min_size, min_size)
                sizes = []
                for i in range(num_blocks):
                    sizes.append((min_size + (i % 3), min_size + ((i // 3) % 3)))
            
            return sizes
        
        def _try_place_blocks(sizes):
            """Try to place blocks with the given sizes"""
            # Start with a clean grid
            temp_grid = np.full((grid_size, grid_size), background, dtype=int)
            blocks = []
            block_positions = []
            
            random.shuffle(sizes)  # Randomize block order
            
            # Try to place each block
            for height, width in sizes:
                # Try multiple positions for this block
                block_placed = False
                for _ in range(30):  # Try 30 different positions
                    # Random position (leaving space for border)
                    max_row = max(1, grid_size - height - 1)
                    max_col = max(1, grid_size - width - 1)
                    
                    if max_row <= 1 or max_col <= 1:
                        # Block is too big for the grid
                        continue
                    
                    row = random.randint(1, max_row)
                    col = random.randint(1, max_col)
                    
                    # Check if this position is valid (no overlap with existing blocks)
                    valid_position = True
                    for existing_row, existing_col, existing_h, existing_w in block_positions:
                        # Check if blocks are at least 2 cells apart
                        if not (row + height + 2 <= existing_row or 
                                existing_row + existing_h + 2 <= row or
                                col + width + 2 <= existing_col or
                                existing_col + existing_w + 2 <= col):
                            valid_position = False
                            break
                    
                    if valid_position:
                        block_positions.append((row, col, height, width))
                        blocks.append((row, col, height, width))
                        block_placed = True
                        break
                
                if not block_placed:
                    return None, None
            
            return temp_grid, blocks
        
        def _create_blocks_on_grid(temp_grid, blocks, frame_block_idx, blocks_with_interior):
            """Create the blocks on the grid"""
            for i, (row, col, height, width) in enumerate(blocks):
                if i == frame_block_idx:
                    # Create a one-cell wide frame
                    # Top and bottom edges
                    temp_grid[row, col:col+width] = block_color
                    temp_grid[row+height-1, col:col+width] = block_color
                    # Left and right edges
                    temp_grid[row:row+height, col] = block_color
                    temp_grid[row:row+height, col+width-1] = block_color
                
                elif i in blocks_with_interior and height >= 5 and width >= 5:
                    # Fill the block
                    temp_grid[row:row+height, col:col+width] = block_color
                    
                    # Create an interior empty space
                    # The interior empty space dimensions (1 row high, 3 cols wide or vice versa)
                    if random.choice([True, False]) and height > 6:
                        interior_height = 3
                        interior_width = 1
                    else:
                        interior_height = 1
                        interior_width = min(3, width - 2)
                    
                    # Ensure we have valid ranges for interior placement
                    interior_row_max = height - interior_height - 2
                    interior_col_max = width - interior_width - 2
                    
                    if interior_row_max >= 2 and interior_col_max >= 2:
                        # Calculate position for the interior space (away from edges)
                        # Ensure valid range for randint
                        interior_row_start = 2
                        interior_row_end = max(interior_row_start, interior_row_max)
                        interior_col_start = 2
                        interior_col_end = max(interior_col_start, interior_col_max)
                        
                        # Use min when they're equal to avoid randint error
                        if interior_row_start == interior_row_end:
                            interior_row = interior_row_start
                        else:
                            interior_row = random.randint(interior_row_start, interior_row_end)
                            
                        if interior_col_start == interior_col_end:
                            interior_col = interior_col_start
                        else:
                            interior_col = random.randint(interior_col_start, interior_col_end)
                        
                        # Create the interior space
                        interior_row = row + interior_row
                        interior_col = col + interior_col
                        temp_grid[interior_row:interior_row+interior_height, 
                             interior_col:interior_col+interior_width] = background
                
                else:
                    # Fill the entire block with block_color
                    temp_grid[row:row+height, col:col+width] = block_color
            
            return temp_grid
        
        # Generate block sizes
        sizes = _generate_block_sizes()
        
        # Try to place blocks
        max_attempts = 200
        for attempt in range(max_attempts):
            temp_grid, blocks = _try_place_blocks(sizes)
            
            if blocks and len(blocks) == num_blocks:
                # Successfully placed all blocks
                # Decide which block is the frame and which have interiors
                frame_block_idx = random.randint(0, num_blocks - 1)
                
                # Decide which blocks have interior spaces
                blocks_with_interior = []
                remaining_blocks = [i for i in range(num_blocks) if i != frame_block_idx]
                
                # Ensure we don't try to add more interior blocks than available
                interior_count = min(interior_blocks, len(remaining_blocks))
                if interior_count > 0:
                    blocks_with_interior = random.sample(remaining_blocks, interior_count)
                
                # Create each block on the grid
                grid = _create_blocks_on_grid(temp_grid, blocks, frame_block_idx, blocks_with_interior)
                return grid
        
        # If we couldn't place blocks after max attempts, create a simple grid
        # Create a larger fixed-size grid to ensure blocks fit
        grid_size = max(grid_size, 20)
        grid = np.full((grid_size, grid_size), background, dtype=int)
        
        # Create a simple fallback arrangement with num_blocks placed in fixed-ish positions
        # Ensure these positions are far enough apart
        simple_blocks = []
        spacing = max(3, grid_size // (num_blocks + 1))
        for i in range(num_blocks):
            row = 2 + (i % 2) * (grid_size // 2 - 3)
            col = 2 + (i // 2) * (grid_size // max(2, num_blocks//2) - 3)
            h = 5
            w = 5 + (i % 2)
            simple_blocks.append((row, col, h, w))
        
        # Ensure blocks fit in the grid
        valid_blocks = []
        for row, col, height, width in simple_blocks:
            if row + height < grid_size and col + width < grid_size:
                valid_blocks.append((row, col, height, width))
        
        # If we don't have enough valid blocks, adjust sizes to create num_blocks small blocks
        if len(valid_blocks) < num_blocks:
            valid_blocks = []
            for i in range(num_blocks):
                row = 2 + (i // 2) * 5
                col = 2 + (i % 2) * (grid_size - 8)
                h = 4 + (i % 2)
                w = 4 + ((i+1) % 2)
                if row + h < grid_size and col + w < grid_size:
                    valid_blocks.append((row, col, h, w))

        # Choose a frame block index among valid blocks
        frame_block_idx = random.randint(0, len(valid_blocks) - 1) if valid_blocks else 0

        # Determine which blocks have interiors
        remaining_blocks = [i for i in range(len(valid_blocks)) if i != frame_block_idx]
        interior_count = min(interior_blocks, len(remaining_blocks))
        blocks_with_interior = random.sample(remaining_blocks, interior_count) if interior_count > 0 else []

        # Create the blocks on the grid
        grid = _create_blocks_on_grid(grid, valid_blocks, frame_block_idx, blocks_with_interior)

        return grid
    
    def transform_input(self, grid, taskvars):
        """
        Transform the input grid by adding frames around the blocks and 
        changing any background cells within those frames to new_color
        """
        # Create a copy of the input grid
        output = grid.copy()
        block_color = taskvars['block_color']
        background = taskvars['background']
        frame_color = taskvars['frame_color']
        new_color = taskvars['new_color']
        
        # Find all connected blocks of block_color
        blocks = find_connected_objects(grid, diagonal_connectivity=False, 
                                       background=background, monochromatic=True)
        
        # Filter for blocks of the specified color
        block_objects = blocks.with_color(block_color)
        
        # Process each block
        for block in block_objects:
            # Get the bounding box of the block
            box = block.bounding_box
            row_slice, col_slice = box
            
            # Calculate frame coordinates ensuring we stay within grid bounds
            top_row = max(0, row_slice.start - 1)
            bottom_row = min(grid.shape[0] - 1, row_slice.stop)
            left_col = max(0, col_slice.start - 1)
            right_col = min(grid.shape[1] - 1, col_slice.stop)
            
            # Add a frame around the block
            # Top frame
            if top_row < row_slice.start:
                output[top_row, left_col:right_col + 1] = frame_color
            # Bottom frame
            if bottom_row > row_slice.stop - 1:  
                output[bottom_row, left_col:right_col + 1] = frame_color
            # Left frame
            if left_col < col_slice.start:
                output[top_row:bottom_row + 1, left_col] = frame_color
            # Right frame
            if right_col > col_slice.stop - 1:
                output[top_row:bottom_row + 1, right_col] = frame_color
            
            # Create a mask for the expanded region (block + frame)
            expanded_region = np.zeros_like(grid, dtype=bool)
            expanded_region[top_row:bottom_row+1, left_col:right_col+1] = True
            
            # Mask for the original block
            block_mask = np.zeros_like(grid, dtype=bool)
            for r, c, _ in block.cells:
                block_mask[r, c] = True
            
            # Mask for the frame (cells we just colored)
            frame_mask = (output == frame_color)
            
            # Find interior cells (within expanded region, background color, not part of block or frame)
            interior_mask = (
                expanded_region &                 # Within expanded region
                (grid == background) &            # Was background color
                ~block_mask &                     # Not part of the original block
                ~frame_mask                       # Not part of the frame
            )
            
            # Change interior cells to new_color
            output[interior_mask] = new_color
        
        return output

