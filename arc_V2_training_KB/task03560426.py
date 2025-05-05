from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import GridObject, find_connected_objects
from input_library import create_object, retry

class Tasktask03560426Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "They contain several single-colored rectangular objects, with all remaining cells being empty (0).",
            "All rectangular objects must be differently colored within a single grid, and the color set should vary across examples.",
            "Each rectangular object must be vertically connected to the bottom edge of the grid.",
            "The first rectangular object must always be connected to the bottom-left corner of the grid.",
            "Each block must be separated from the next by exactly one empty (0) column.",
            "The length and width of each block can vary across the grid.",
            "The vertical lengths of the blocks must be such that the total sum of their heights does not exceed {vars['grid_size'] + 3}."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all single-colored rectangular objects that are connected to the bottom edge of the grid.",
            "First, move the leftmost rectangular block (which starts at the bottom-left corner) so that it is now connected to the top-left corner.",
            "Next, take the second block (which appears to the right of the first in the input) and move it such that its top-left corner aligns exactly with the bottom-right corner of the previously placed block in the output.",
            "Repeat this process for all remaining blocks, placing each subsequent block so that its top-left corner aligns with the bottom-right corner of the previous block.",
            "In case of overlapping cells, the block that was originally to the right (i.e.,originally appeared farther to the right in input grid) has priority over the color during overlap."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Randomly select grid size within constraints
        # Ensure grid is big enough to fit at least 3 blocks
        grid_size = random.randint(10, 30)
        
        taskvars = {
            'grid_size': grid_size
        }
        
        # Generate 3-6 training examples
        num_train_examples = random.randint(3, 6)
        
        train_pairs = []
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
            
        # Generate 1 test example with special requirement (last column empty)
        test_input = self.create_input(taskvars, {'for_test': True})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        is_test_case = gridvars.get('for_test', False)
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Generate random block parameters within constraints
        available_colors = list(range(1, 10))  # Colors 1-9
        random.shuffle(available_colors)
        
        # Ensure we have at least 3 blocks and at most what can fit in the grid
        min_blocks = 3
        
        # Calculate maximum blocks that can fit with all constraints
        # For test case, we need the last column empty
        usable_width = grid_size - 1 if is_test_case else grid_size
        
        # Each block + gap takes at least 3 columns (except last block which needs only 2)
        # Formula: (usable_width - 2) / 3 + 1 for the last block that doesn't need a gap
        max_possible_blocks = (usable_width - 2) // 3 + 1
        
        # Safety check: if max_possible_blocks < min_blocks, use a larger grid
        if max_possible_blocks < min_blocks:
            # Recursively try again with a larger grid
            new_gridvars = gridvars.copy()
            new_gridvars['retry_count'] = gridvars.get('retry_count', 0) + 1
            
            # Prevent infinite recursion
            if new_gridvars['retry_count'] > 5:
                # Force fixed values as a fallback
                grid_size = 15
                max_possible_blocks = 5 
                usable_width = grid_size - 1 if is_test_case else grid_size
            else:
                return self.create_input(
                    {'grid_size': grid_size + 2}, 
                    new_gridvars
                )
        
        # Decide how many blocks to create - handle case when min==max
        if min_blocks >= max_possible_blocks:
            num_blocks = min_blocks
        else:
            num_blocks = random.randint(min_blocks, min(max_possible_blocks, len(available_colors)))
        
        # Calculate how much space we need to fill exactly
        # We need (num_blocks - 1) gaps of 1 column each
        total_block_width = usable_width - (num_blocks - 1)
        
        # Distribute width across blocks, ensuring each is at least 2 wide
        min_width = 2
        widths = []
        
        # Check if enough space for minimum width blocks
        if total_block_width < min_width * num_blocks:
            # Not enough space, retry with different parameters
            new_gridvars = gridvars.copy()
            new_gridvars['retry_count'] = gridvars.get('retry_count', 0) + 1
            
            if new_gridvars['retry_count'] > 5:
                # Force some reasonable values as fallback
                min_width = 1
                num_blocks = (total_block_width // min_width)
            else:
                return self.create_input(
                    {'grid_size': grid_size + 2}, 
                    new_gridvars
                )
        
        # First assign minimum width to all blocks
        remaining_width = total_block_width - (min_width * num_blocks)
        
        # Distribute extra width, ensuring last block gets priority to fill to the edge
        for i in range(num_blocks):
            if i == num_blocks - 1:
                # Last block gets all remaining width
                widths.append(min_width + remaining_width)
                remaining_width = 0
            else:
                # Other blocks get random extra width
                blocks_left = num_blocks - i - 1  # Exclude last block
                max_extra = remaining_width // max(1, blocks_left) if blocks_left > 0 else remaining_width
                extra = random.randint(0, max_extra) if max_extra > 0 else 0
                widths.append(min_width + extra)
                remaining_width -= extra
        
        # Now plan the heights, ensuring total doesn't exceed grid_size + 3
        min_height = 2
        max_total_height = grid_size + 3
        
        heights = []
        remaining_height = max_total_height - (min_height * num_blocks)
        
        if remaining_height < 0:
            # Not enough height capacity, retry with different parameters
            new_gridvars = gridvars.copy()
            new_gridvars['retry_count'] = gridvars.get('retry_count', 0) + 1
            
            if new_gridvars['retry_count'] > 5:
                # Use smaller heights as fallback
                min_height = 1
                remaining_height = 0
            else:
                return self.create_input(
                    {'grid_size': grid_size + 2}, 
                    new_gridvars
                )
            
        for i in range(num_blocks):
            blocks_left = num_blocks - i
            max_extra = remaining_height // max(1, blocks_left) if blocks_left > 0 else 0
            
            # Limit height to grid_size
            max_allowed = min(grid_size - min_height, max_extra)
            extra = random.randint(0, max_allowed) if max_allowed > 0 else 0
            
            heights.append(min_height + extra)
            remaining_height -= extra
            
        # Create the blocks
        current_col = 0
        for i in range(num_blocks):
            width = widths[i]
            height = heights[i]
            color = available_colors[i]
            
            # Calculate top-left corner position
            row_start = grid_size - height
            col_start = current_col
            
            # Add the block to the grid
            for r in range(row_start, grid_size):
                for c in range(col_start, col_start + width):
                    if 0 <= r < grid_size and 0 <= c < grid_size:
                        grid[r, c] = color
            
            # Move to next position (add block width + 1 for gap)
            current_col += width + 1
            
        # Verify we have the correct number of blocks
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        if len(objects.objects) < min_blocks:
            # If we've retried too many times, return what we have
            if gridvars.get('retry_count', 0) > 5:
                return grid
                
            # Otherwise try again
            new_gridvars = gridvars.copy()
            new_gridvars['retry_count'] = gridvars.get('retry_count', 0) + 1
            return self.create_input(taskvars, new_gridvars)
            
        # Final check for test case: ensure the last column is empty
        if is_test_case and not np.all(grid[:, -1] == 0):
            # If we've retried too many times, force last column to be empty
            if gridvars.get('retry_count', 0) > 5:
                grid[:, -1] = 0
                return grid
                
            # Otherwise try again
            new_gridvars = gridvars.copy()
            new_gridvars['retry_count'] = gridvars.get('retry_count', 0) + 1
            return self.create_input(taskvars, new_gridvars)
            
        # Final check for training: ensure no extra empty columns at the end
        if not is_test_case and np.all(grid[:, -1] == 0) and np.all(grid[:, -2] == 0):
            # If we've retried too many times, return what we have
            if gridvars.get('retry_count', 0) > 5:
                return grid
                
            # Otherwise try again
            new_gridvars = gridvars.copy()
            new_gridvars['retry_count'] = gridvars.get('retry_count', 0) + 1
            return self.create_input(taskvars, new_gridvars)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        output_grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Sort objects by their leftmost column position
        sorted_objects = sorted(objects.objects, key=lambda obj: min(c for _, c, _ in obj.cells))
        
        if not sorted_objects or len(sorted_objects) < 3:
            # If we don't have at least 3 objects, return empty grid
            # This shouldn't happen with our improved input generation
            return output_grid
        
        # Extract block information - original position and dimensions
        blocks = []
        for obj in sorted_objects:
            color = next(iter(obj.colors))
            rows = [r for r, _, _ in obj.cells]
            cols = [c for _, c, _ in obj.cells]
            min_row = min(rows)
            min_col = min(cols)
            height = max(rows) - min_row + 1
            width = max(cols) - min_col + 1
            blocks.append({
                'color': color,
                'height': height,
                'width': width,
                'original_pos': (min_row, min_col)
            })
        
        # 1. Place the first (leftmost) block at the top-left corner (0,0)
        first_block = blocks[0]
        for r in range(first_block['height']):
            for c in range(first_block['width']):
                if r < grid_size and c < grid_size:
                    output_grid[r, c] = first_block['color']
        
        # Current position of the bottom-right corner of the last placed block
        current_bottom_right = (first_block['height'] - 1, first_block['width'] - 1)
        
        # 2. For each subsequent block, place its top-left at the bottom-right of the previous
        for i in range(1, len(blocks)):
            block = blocks[i]
            
            # Calculate new position
            new_top_left = current_bottom_right
            
            # Place this block
            for r in range(block['height']):
                for c in range(block['width']):
                    new_r = new_top_left[0] + r
                    new_c = new_top_left[1] + c
                    
                    # Check bounds and place (with later blocks having priority)
                    if 0 <= new_r < grid_size and 0 <= new_c < grid_size:
                        output_grid[new_r, new_c] = block['color']
            
            # Update bottom-right for next block
            current_bottom_right = (
                new_top_left[0] + block['height'] - 1,
                new_top_left[1] + block['width'] - 1
            )
        
        return output_grid

