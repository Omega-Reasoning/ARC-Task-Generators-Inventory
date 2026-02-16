from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, Any, List, Tuple, Set

class Task09c534e7Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains {color('block_color')} rectangular objects, each at least 3 cells wide and 3 cells long, with some {color('block_color')} vertical and horizontal lines connecting the {color('block_color')} rectangular objects. All remaining cells are empty (0).",
            "The {color('block_color')} rectangular objects are arranged in several groups, with all objects in each group connected by some vertical and horizontal {color('block_color')} lines.",
            "Each group should have at least 2 blocks, where the size of each block may vary.",
            "In each group of blocks, choose one block and place a single colored cell in its interior, leaving the {color('block_color')} edges unchanged.",
            "The single colored cells are different for each group of blocks.",
            "The group of blocks appears as a chain, where consecutive blocks are connected only to their immediate neighbors, and not all blocks are directly connected to every other block in the group."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying different groups of blocks, where a group is defined as blocks connected by vertical and horizontal {color('block_color')} lines.",
            "Each group contains one block that has a single colored cell different from {color('block_color')}.",
            "The output grid is constructed by changing the color of all {color('block_color')} cells that lie in the interior of {color('block_color')} blocks to match the single-colored cell of each group, only leaving a one-cell-wide frame around each block in {color('block_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Create task variables
        taskvars = {
            'block_color': random.randint(1, 9)
        }
        
        # Create training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        block_color = taskvars['block_color']
        
        # Create grid with random size
        height = random.randint(15, 30)
        width = random.randint(15, 30)
        
        def _generate_valid_grid():
            grid = np.zeros((height, width), dtype=int)
            
            # Create 2-4 groups of blocks
            num_groups = random.randint(2, 4)
            group_colors = []
            
            # Select different colors for each group (different from block_color)
            available_colors = [c for c in range(1, 10) if c != block_color]
            group_colors = random.sample(available_colors, num_groups)
            
            placed_blocks = []  # List of (top, left, bottom, right) for each block
            
            for group_idx in range(num_groups):
                # Each group has 2-3 blocks
                blocks_in_group = random.randint(2, 3)
                group_blocks = []
                
                for block_idx in range(blocks_in_group):
                    # Try to place a block
                    for attempt in range(50):
                        # Block dimensions (at least 3x3)
                        block_height = random.randint(3, 6)
                        block_width = random.randint(3, 6)
                        
                        if block_idx == 0:
                            # First block in group - place anywhere
                            top = random.randint(1, height - block_height - 1)
                            left = random.randint(1, width - block_width - 1)
                        else:
                            # Subsequent blocks - place near previous block
                            prev_block = group_blocks[-1]
                            prev_top, prev_left, prev_bottom, prev_right = prev_block
                            
                            # Try to place adjacent to previous block
                            positions = []
                            
                            # Above
                            if prev_top - block_height - 2 >= 1:
                                positions.append((prev_top - block_height - 2, prev_left))
                            # Below  
                            if prev_bottom + 2 + block_height < height - 1:
                                positions.append((prev_bottom + 2, prev_left))
                            # Left
                            if prev_left - block_width - 2 >= 1:
                                positions.append((prev_top, prev_left - block_width - 2))
                            # Right
                            if prev_right + 2 + block_width < width - 1:
                                positions.append((prev_top, prev_right + 2))
                            
                            if not positions:
                                continue
                                
                            top, left = random.choice(positions)
                        
                        bottom = top + block_height - 1
                        right = left + block_width - 1
                        
                        # Additional bounds check
                        if bottom >= height or right >= width:
                            continue
                        
                        # Check if this block overlaps with any existing block
                        overlap = False
                        for existing in placed_blocks:
                            if not (bottom + 1 < existing[0] or top - 1 > existing[2] or 
                                   right + 1 < existing[1] or left - 1 > existing[3]):
                                overlap = True
                                break
                        
                        if not overlap:
                            group_blocks.append((top, left, bottom, right))
                            placed_blocks.append((top, left, bottom, right))
                            break
                    else:
                        # Failed to place block
                        return None
                
                if len(group_blocks) < 2:
                    return None
                
                # Draw the blocks
                for top, left, bottom, right in group_blocks:
                    # Additional bounds check before drawing
                    if bottom < height and right < width:
                        grid[top:bottom+1, left:right+1] = block_color
                
                # Connect blocks in the group with lines
                for i in range(len(group_blocks) - 1):
                    block1 = group_blocks[i]
                    block2 = group_blocks[i + 1]
                    
                    # Connect middle of blocks
                    b1_center_r = (block1[0] + block1[2]) // 2
                    b1_center_c = (block1[1] + block1[3]) // 2
                    b2_center_r = (block2[0] + block2[2]) // 2  
                    b2_center_c = (block2[1] + block2[3]) // 2
                    
                    # Draw connecting line with bounds checking
                    if abs(b1_center_r - b2_center_r) > abs(b1_center_c - b2_center_c):
                        # Vertical connection
                        col = b1_center_c
                        start_row = min(b1_center_r, b2_center_r)
                        end_row = max(b1_center_r, b2_center_r)
                        for r in range(start_row, end_row + 1):
                            if 0 <= r < height and 0 <= col < width:
                                grid[r, col] = block_color
                    else:
                        # Horizontal connection
                        row = b1_center_r
                        start_col = min(b1_center_c, b2_center_c)
                        end_col = max(b1_center_c, b2_center_c)
                        for c in range(start_col, end_col + 1):
                            if 0 <= row < height and 0 <= c < width:
                                grid[row, c] = block_color
                
                # Place single colored cell in one block of the group
                chosen_block = random.choice(group_blocks)
                top, left, bottom, right = chosen_block
                
                # Ensure the chosen block is within bounds
                if bottom < height and right < width:
                    # Place in interior (not on edges)
                    interior_rows = list(range(top + 1, min(bottom, height - 1)))
                    interior_cols = list(range(left + 1, min(right, width - 1)))
                    
                    if interior_rows and interior_cols:
                        cell_r = random.choice(interior_rows)
                        cell_c = random.choice(interior_cols)
                        
                        # Final bounds check before placing the colored cell
                        if 0 <= cell_r < height and 0 <= cell_c < width:
                            grid[cell_r, cell_c] = group_colors[group_idx]
            
            return grid
        
        # Generate a valid grid
        result = retry(_generate_valid_grid, lambda x: x is not None, max_attempts=100)
        return result
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        block_color = taskvars['block_color']
        output_grid = grid.copy()
        height, width = grid.shape
        
        # Find all connected components of block_color using flood fill
        visited = np.zeros((height, width), dtype=bool)
        components = []
        
        for start_r in range(height):
            for start_c in range(width):
                if not visited[start_r, start_c] and grid[start_r, start_c] == block_color:
                    # Flood fill to find connected component
                    component = set()
                    stack = [(start_r, start_c)]
                    
                    while stack:
                        r, c = stack.pop()
                        if (r < 0 or r >= height or c < 0 or c >= width or 
                            visited[r, c] or grid[r, c] != block_color):
                            continue
                        
                        visited[r, c] = True
                        component.add((r, c))
                        
                        # Add 4-connected neighbors
                        stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
                    
                    if component:
                        components.append(component)
        
        # Process each connected component
        for component in components:
            # Find the special color in this component
            special_color = None
            
            # First check if any cell in the component has a special color
            for r, c in component:
                if grid[r, c] != block_color and grid[r, c] != 0:
                    special_color = grid[r, c]
                    break
            
            # If not found, check cells that might be inside blocks within this component
            if special_color is None:
                for r, c in component:
                    # Check adjacent cells for special colors
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < height and 0 <= nc < width and
                                grid[nr, nc] != 0 and grid[nr, nc] != block_color):
                                special_color = grid[nr, nc]
                                break
                        if special_color is not None:
                            break
                    if special_color is not None:
                        break
            
            if special_color is not None:
                # Find rectangular blocks within this component and fill their interiors
                component_mask = np.zeros((height, width), dtype=bool)
                for r, c in component:
                    component_mask[r, c] = True
                
                processed = set()
                
                # Look for rectangular blocks
                for r in range(height):
                    for c in range(width):
                        if (r, c) in component and (r, c) not in processed:
                            # Try to find a rectangular block starting from this point
                            best_block = None
                            best_area = 0
                            
                            # Try different rectangle sizes (at least 3x3)
                            for end_r in range(r + 2, min(height, r + 10)):
                                for end_c in range(c + 2, min(width, c + 10)):
                                    # Check if this forms a valid rectangular block
                                    if end_r >= height or end_c >= width:
                                        continue
                                    
                                    # Check if all border cells are in the component
                                    is_valid_rect = True
                                    
                                    # Check top and bottom borders
                                    for check_c in range(c, end_c + 1):
                                        if not component_mask[r, check_c] or not component_mask[end_r, check_c]:
                                            is_valid_rect = False
                                            break
                                    
                                    if is_valid_rect:
                                        # Check left and right borders
                                        for check_r in range(r, end_r + 1):
                                            if not component_mask[check_r, c] or not component_mask[check_r, end_c]:
                                                is_valid_rect = False
                                                break
                                    
                                    if is_valid_rect:
                                        # Check that most of the rectangle is filled
                                        filled_cells = 0
                                        total_cells = (end_r - r + 1) * (end_c - c + 1)
                                        
                                        for check_r in range(r, end_r + 1):
                                            for check_c in range(c, end_c + 1):
                                                if component_mask[check_r, check_c]:
                                                    filled_cells += 1
                                        
                                        fill_ratio = filled_cells / total_cells
                                        if fill_ratio >= 0.7:  # At least 70% filled
                                            area = total_cells
                                            if area > best_area:
                                                best_area = area
                                                best_block = (r, c, end_r, end_c)
                            
                            if best_block is not None:
                                top, left, bottom, right = best_block
                                
                                # Fill the interior of this block with the special color
                                for fill_r in range(top + 1, bottom):
                                    for fill_c in range(left + 1, right):
                                        if 0 <= fill_r < height and 0 <= fill_c < width:
                                            output_grid[fill_r, fill_c] = special_color
                                
                                # Mark all cells in this block as processed
                                for proc_r in range(top, bottom + 1):
                                    for proc_c in range(left, right + 1):
                                        processed.add((proc_r, proc_c))
        
        return output_grid

