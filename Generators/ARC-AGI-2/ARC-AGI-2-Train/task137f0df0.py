from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import create_object, retry
import numpy as np
import random

class Task137f0df0Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each grid contains several {color('block_color')} 2×2 objects, where each object is made of 8-way connected cells, and the remaining cells are empty (0).",
            "The number of blocks should be a multiple of 3 and arranged in a regular grid layout — i.e., the same number of objects should be placed in each selected row and each selected column. For example, if one selected set of rows contains 2 blocks, then all other selected sets of rows must also contain 2 blocks. The same rule applies to the selected columns.",
            "First, decide which rows and columns will be used for placing the 2×2 blocks.",
            "Ensure that the selected rows and columns are evenly spaced — i.e., each is separated from the next by the same number of rows or columns.",
            "The number of blocks may vary across grids.",
            "There can be extra empty (0) rows or columns at the beginning or end of the grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying all rows and columns that lie between 2×2 {color('block_color')} blocks.",
            "Color all empty cells in these rows and columns with {color('fill_color')}, but only in the portions that lie between the blocks.",
            "This results in partially filled rows and columns with {color('fill_color')} cells. Once this is done, fill all remaining empty (0) cells in those same rows and columns with {color('extra_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Generate task variables
        grid_size = random.randint(12, 20)  # More conservative range to avoid constraint issues
        
        # Ensure colors are different
        colors = random.sample(range(1, 10), 3)
        block_color, fill_color, extra_color = colors
        
        taskvars = {
            'grid_size': grid_size,
            'block_color': block_color,
            'fill_color': fill_color,
            'extra_color': extra_color
        }
        
        # Generate train and test grids
        train_grids = []
        for i in range(3):
            # Use even spacing for training examples
            input_grid = self.create_input(taskvars, {'use_even_spacing': True})
            output_grid = self.transform_input(input_grid, taskvars)
            train_grids.append({'input': input_grid, 'output': output_grid})
        
        # Use uneven spacing for test case as requested
        test_input = self.create_input(taskvars, {'use_even_spacing': False})
        test_output = self.transform_input(test_input, taskvars)
        test_grids = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_grids, 'test': test_grids}

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        block_color = taskvars['block_color']
        use_even_spacing = gridvars.get('use_even_spacing', True)
        
        def generate_valid_grid():
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # More flexible approach to avoid constraint issues
            # Start with smaller numbers and adjust
            max_blocks_per_dim = min(3, (grid_size - 2) // 4)  # More conservative
            if max_blocks_per_dim < 2:
                max_blocks_per_dim = 2
            
            # Try different combinations until we find one that works
            valid_combinations = []
            for br in range(2, max_blocks_per_dim + 1):
                for bc in range(2, max_blocks_per_dim + 1):
                    if (br * bc) % 3 == 0:  # Multiple of 3
                        # Check if it fits with minimal spacing
                        min_rows_needed = br * 2 + (br - 1) * 1  # 1 space between blocks
                        min_cols_needed = bc * 2 + (bc - 1) * 1
                        if min_rows_needed <= grid_size and min_cols_needed <= grid_size:
                            valid_combinations.append((br, bc))
            
            if not valid_combinations:
                return None
            
            blocks_per_row, blocks_per_col = random.choice(valid_combinations)
            
            if use_even_spacing:
                # Even spacing
                max_row_spacing = (grid_size - blocks_per_row * 2) // max(1, blocks_per_row - 1)
                max_col_spacing = (grid_size - blocks_per_col * 2) // max(1, blocks_per_col - 1)
                
                row_spacing = random.randint(1, min(4, max_row_spacing))
                col_spacing = random.randint(1, min(4, max_col_spacing))
                
                # Calculate starting positions
                required_rows = blocks_per_row * 2 + (blocks_per_row - 1) * row_spacing
                required_cols = blocks_per_col * 2 + (blocks_per_col - 1) * col_spacing
                
                if required_rows > grid_size or required_cols > grid_size:
                    return None
                
                start_row = random.randint(0, grid_size - required_rows)
                start_col = random.randint(0, grid_size - required_cols)
                
                # Place blocks with even spacing
                for i in range(blocks_per_row):
                    for j in range(blocks_per_col):
                        row = start_row + i * (2 + row_spacing)
                        col = start_col + j * (2 + col_spacing)
                        
                        if row + 1 >= grid_size or col + 1 >= grid_size:
                            return None
                        
                        grid[row:row+2, col:col+2] = block_color
            
            else:
                # Uneven spacing - different gaps between blocks
                # Generate positions with varying spacing
                row_positions = []
                col_positions = []
                
                # For rows: start with first position, then add varying gaps
                current_row = random.randint(0, 2)
                row_positions.append(current_row)
                
                for i in range(1, blocks_per_row):
                    # Vary the gap (1 to 4 spaces, but different from previous)
                    possible_gaps = [1, 2, 3, 4]
                    if i > 1:  # Avoid same gap as previous
                        prev_gap = row_positions[i-1] - row_positions[i-2] - 2
                        if prev_gap in possible_gaps:
                            possible_gaps.remove(prev_gap)
                    
                    gap = random.choice(possible_gaps)
                    next_row = current_row + 2 + gap
                    
                    if next_row + 1 >= grid_size:
                        return None
                    
                    row_positions.append(next_row)
                    current_row = next_row
                
                # For columns: similar approach
                current_col = random.randint(0, 2)
                col_positions.append(current_col)
                
                for j in range(1, blocks_per_col):
                    possible_gaps = [1, 2, 3, 4]
                    if j > 1:  # Avoid same gap as previous
                        prev_gap = col_positions[j-1] - col_positions[j-2] - 2
                        if prev_gap in possible_gaps:
                            possible_gaps.remove(prev_gap)
                    
                    gap = random.choice(possible_gaps)
                    next_col = current_col + 2 + gap
                    
                    if next_col + 1 >= grid_size:
                        return None
                    
                    col_positions.append(next_col)
                    current_col = next_col
                
                # Place blocks at calculated positions
                for row_pos in row_positions:
                    for col_pos in col_positions:
                        if row_pos + 1 >= grid_size or col_pos + 1 >= grid_size:
                            return None
                        grid[row_pos:row_pos+2, col_pos:col_pos+2] = block_color
            
            return grid
        
        # Use retry with more attempts and better error handling
        try:
            return retry(generate_valid_grid, lambda x: x is not None, max_attempts=200)
        except ValueError:
            # If we still can't generate a valid grid, fall back to a simpler configuration
            grid = np.zeros((grid_size, grid_size), dtype=int)
            # Simple 2x3 arrangement (6 blocks = multiple of 3)
            positions = [(1, 1), (1, 5), (1, 9), (5, 1), (5, 5), (5, 9)]
            for i, (r, c) in enumerate(positions):
                if r + 1 < grid_size and c + 1 < grid_size:
                    grid[r:r+2, c:c+2] = block_color
                if i >= 5:  # Ensure we have at least 6 blocks
                    break
            return grid

    def transform_input(self, grid, taskvars):
        block_color = taskvars['block_color']
        fill_color = taskvars['fill_color']
        extra_color = taskvars['extra_color']
        
        output_grid = grid.copy()
        rows, cols = grid.shape
        
        # Find all 2x2 blocks by looking for their top-left corners
        block_positions = []
        for r in range(rows - 1):
            for c in range(cols - 1):
                if (grid[r, c] == block_color and 
                    grid[r, c+1] == block_color and
                    grid[r+1, c] == block_color and
                    grid[r+1, c+1] == block_color):
                    block_positions.append((r, c))
        
        if len(block_positions) < 2:
            return output_grid  # Need at least 2 blocks for "between" to make sense
        
        # Group blocks by rows and columns to find patterns
        block_top_rows = sorted(list(set(r for r, c in block_positions)))
        block_left_cols = sorted(list(set(c for r, c in block_positions)))
        
        # Step 1: Fill with fill_color only in the portions between blocks
        rows_to_fill_completely = set()
        cols_to_fill_completely = set()
        
        # Fill rows between blocks (only the portions between block columns)
        for i in range(len(block_top_rows) - 1):
            current_block_bottom = block_top_rows[i] + 1
            next_block_top = block_top_rows[i + 1]
            
            # Fill rows between these blocks
            for r in range(current_block_bottom + 1, next_block_top):
                # Only fill between the leftmost and rightmost block columns
                left_bound = min(block_left_cols)
                right_bound = max(block_left_cols) + 1  # +1 because block is 2 wide
                
                for c in range(left_bound, right_bound + 1):
                    if c < cols and output_grid[r, c] == 0:
                        output_grid[r, c] = fill_color
                
                rows_to_fill_completely.add(r)
        
        # Fill columns between blocks (only the portions between block rows)
        for i in range(len(block_left_cols) - 1):
            current_block_right = block_left_cols[i] + 1
            next_block_left = block_left_cols[i + 1]
            
            # Fill columns between these blocks
            for c in range(current_block_right + 1, next_block_left):
                # Only fill between the topmost and bottommost block rows
                top_bound = min(block_top_rows)
                bottom_bound = max(block_top_rows) + 1  # +1 because block is 2 tall
                
                for r in range(top_bound, bottom_bound + 1):
                    if r < rows and output_grid[r, c] == 0:
                        output_grid[r, c] = fill_color
                
                cols_to_fill_completely.add(c)
        
        # Step 2: Fill ALL remaining empty (0) cells in those same rows and columns with extra_color
        for r in rows_to_fill_completely:
            for c in range(cols):
                if output_grid[r, c] == 0:
                    output_grid[r, c] = extra_color
        
        for c in cols_to_fill_completely:
            for r in range(rows):
                if output_grid[r, c] == 0:
                    output_grid[r, c] = extra_color
        
        return output_grid

