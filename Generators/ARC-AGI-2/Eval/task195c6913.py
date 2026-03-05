from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import create_object, random_cell_coloring, retry, Contiguity
import numpy as np
import random

class Task195c6913Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}×{vars['grid_size']}.",
            "Each input grid has a predominantly colored background, on top of which lies an irregular blob shape along with several 2x2 blocks and single cells.",
            "The 2x2 colored blocks have fixed positions. There are exactly 3 blocks in the top-left area starting at position (1, 1), with remaining blocks horizontally aligned, each placed with exactly two vertical cells between them. Additionally, there is always one more 2x2 block that starts at position ({vars['grid_size'] - 4}, {vars['grid_size'] - 4}).",
            "The colors of the 3 top-left 2x2 blocks follow this rule: the first two blocks have the same color, and the third block has a different color",
            "These 2x2 blocks must always appear on the predominant background of the grid.",
            "The blob extends from left to right and bottom to top. However, as it approaches the opposite borders of the grid, it gradually shrinks. For example, if the blob starts from the bottom, it should be relatively broad there and become narrower toward the top.",
            "There can be one blob or multiple blobs in the grid.",
            "There is exactly one single cell in the first column within the irregularly shaped area (blob) of the grid. Its color must match that of the first 2x2 block located at position (1, 1). This cell must be properly embedded within the blob, with blob-colored cells both above and below it."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying two key colors - the background color and the blob color.",
            "The 2x2 blocks always lie on the background region, while the single colored cell in the first column lies inside the blob.",
            "The transformation begins from this single colored cell and builds a path that always tries to move right and upward as long as there is blob color available.",
            "The path must strictly stay within the blob region.",
            "The path follows a three-color repeating pattern based on the 2x2 blocks.",
            "The path first exhausts rightward movement until it hits the blob boundary, then exhausts upward movement until it hits the blob boundary, then continues rightward again, and so on.",
            "At every turning point in the path, a colored cell is added using the bottom-right 2x2 block color.",
            "This turn indicator cell is placed adjacent to the turning point in the background region.",
            "Once the path is completely finished and cannot extend any longer, a final indicator cell is added at the end of the path using the bottom-right block color.",
            "If the last segment of the path was vertical (upward movement), the final indicator is placed above the last cell. If the last segment was horizontal (rightward movement), the final indicator is placed to the right of the last cell.",
            "However, if the path stopped because it hit the grid border, no final indicator is added since it cannot be placed outside the grid boundaries.",
            "Finally, after all transformations are complete, all 2x2 blocks are removed from the output and replaced with the background color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Generate task variables
        taskvars = {
            'grid_size': random.randint(19, 30)
        }
        
        # Create train examples (3) - all with exactly 3 top-left blocks
        train_examples = []
        for i in range(3):
            gridvars = {'num_top_blocks': 3}  # Always 3 blocks
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Test example - also with exactly 3 top-left blocks
        test_gridvars = {'num_top_blocks': 3}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        num_top_blocks = gridvars['num_top_blocks']
        
        # Choose colors ensuring they're all different
        available_colors = list(range(1, 10))
        
        # Choose background color
        background_color = random.choice(available_colors)
        available_colors.remove(background_color)
        
        # Choose blob color (different from background)
        blob_color = random.choice(available_colors)
        available_colors.remove(blob_color)
        
        # Choose colors for 2x2 blocks (different from background and blob)
        block_colors = self._choose_block_colors(num_top_blocks, available_colors)
        
        # Remove block colors from available colors
        for color in block_colors:
            if color in available_colors:
                available_colors.remove(color)
        
        # Choose bottom-right block color (different from blob and other blocks)
        bottom_right_color = random.choice(available_colors)
        
        # Create grid with background
        grid = np.full((grid_size, grid_size), background_color, dtype=int)
        
        # Create irregular blob regions avoiding block areas
        blob_mask = self._create_blob_regions_avoiding_blocks(grid_size, num_top_blocks)
        
        # Apply blob to grid
        grid[blob_mask] = blob_color
        
        # Place 2x2 blocks at fixed positions (must be on background)
        self._place_2x2_blocks(grid, grid_size, num_top_blocks, block_colors, bottom_right_color, background_color)
        
        # Add exactly one single cell in first column within blob areas (ensuring it's embedded)
        first_block_color = block_colors[0]
        self._add_single_cell_in_blob(grid, blob_mask, first_block_color, blob_color)
        
        return grid
    
    def _get_block_forbidden_areas(self, grid_size, num_top_blocks):
        """Get areas where blobs should not grow to avoid interfering with 2x2 blocks"""
        forbidden = set()
        
        # Top-left block areas (with some padding) - always 3 blocks
        for i in range(3):
            start_row = 1
            start_col = 1 + i * 3
            # Add padding around blocks
            for r in range(max(0, start_row - 1), min(grid_size, start_row + 3)):
                for c in range(max(0, start_col - 1), min(grid_size, start_col + 3)):
                    forbidden.add((r, c))
        
        # Bottom-right block area (with padding)
        bottom_right_row = grid_size - 4
        bottom_right_col = grid_size - 4
        if bottom_right_row >= 0 and bottom_right_col >= 0:
            for r in range(max(0, bottom_right_row - 1), min(grid_size, bottom_right_row + 3)):
                for c in range(max(0, bottom_right_col - 1), min(grid_size, bottom_right_col + 3)):
                    forbidden.add((r, c))
        
        return forbidden
    
    def _create_blob_regions_avoiding_blocks(self, grid_size, num_top_blocks):
        """Create irregular blob regions that avoid block areas"""
        def blob_generator():
            blob_mask = np.zeros((grid_size, grid_size), dtype=bool)
            forbidden_areas = self._get_block_forbidden_areas(grid_size, num_top_blocks)
            
            # Decide blob extension behavior - higher chance for top-right extension
            extend_to_top_right = random.choice([True, True, False])
            
            # Always use single region for better top-right extension
            blob_mask = self._create_single_blob_region_safe(grid_size, blob_mask, forbidden_areas, extend_to_top_right)
            
            # Fill holes in blob to ensure contiguity
            blob_mask = self._fill_blob_holes(blob_mask, forbidden_areas)
            
            return blob_mask
        
        # Use retry to ensure we get a reasonable blob
        def is_valid_blob(blob_mask):
            blob_cells = np.sum(blob_mask)
            # Ensure blob starts from first column and has embedded cells
            has_embedded_cells = self._has_embedded_cells_in_first_column(blob_mask)
            return blob_cells >= grid_size * 3 and blob_cells <= grid_size * grid_size * 0.7 and has_embedded_cells
        
        return retry(blob_generator, is_valid_blob)
    
    def _has_embedded_cells_in_first_column(self, blob_mask):
        """Check if first column has cells with blob cells above and below them"""
        grid_size = blob_mask.shape[0]
        
        # Find cells in first column that have blob cells above and below
        for r in range(2, grid_size - 2):  # Leave room for cells above and below
            if blob_mask[r, 0]:  # Current cell is blob
                # Check if there are blob cells above and below
                has_above = any(blob_mask[r - i, 0] for i in range(1, 3))  # Check 2 cells above
                has_below = any(blob_mask[r + i, 0] for i in range(1, 3))  # Check 2 cells below
                
                if has_above and has_below:
                    return True
        
        return False
    
    def _fill_blob_holes(self, blob_mask, forbidden_areas):
        """Fill small holes within blob regions to make them more contiguous"""
        grid_size = blob_mask.shape[0]
        filled_mask = blob_mask.copy()
        
        # Find potential holes (background cells surrounded by blob cells)
        for r in range(1, grid_size - 1):
            for c in range(1, grid_size - 1):
                if not blob_mask[r, c] and (r, c) not in forbidden_areas:
                    # Count blob neighbors
                    blob_neighbors = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                                if blob_mask[nr, nc]:
                                    blob_neighbors += 1
                    
                    # If surrounded by mostly blob cells, fill it
                    if blob_neighbors >= 6:  # At least 6 out of 8 neighbors are blob
                        filled_mask[r, c] = True
        
        return filled_mask
    
    def _create_single_blob_region_safe(self, grid_size, blob_mask, forbidden_areas, extend_to_top_right):
        """Create a single large blob region avoiding forbidden areas"""
        # Start from middle to lower-middle of first column
        start_r = random.randint(grid_size//2, 3*grid_size//4)
        start_c = 0
        
        # Ensure starting position is not forbidden
        if (start_r, start_c) in forbidden_areas:
            # Find a safe starting position
            safe_positions = [(r, 0) for r in range(grid_size//2, 3*grid_size//4) 
                            if (r, 0) not in forbidden_areas]
            if safe_positions:
                start_r, start_c = random.choice(safe_positions)
            else:
                start_r = 3*grid_size//4  # Fallback
        
        # Use flood fill approach for more controlled growth
        to_fill = [(start_r, start_c)]
        blob_mask[start_r, start_c] = True
        current_size = 1
        target_size = random.randint(grid_size*4, grid_size*6)
        
        while to_fill and current_size < target_size:
            # Process all current cells
            next_to_fill = []
            
            for current_r, current_c in to_fill:
                # Try to grow from this cell
                directions = [(-1,0), (1,0), (0,1), (-1,1), (1,1)]
                if current_c > 0:
                    directions.extend([(0,-1), (-1,-1), (1,-1)])
                
                random.shuffle(directions)
                
                for dr, dc in directions:
                    new_r, new_c = current_r + dr, current_c + dc
                    if (0 <= new_r < grid_size and 0 <= new_c < grid_size and 
                        not blob_mask[new_r, new_c] and (new_r, new_c) not in forbidden_areas):
                        
                        growth_prob = 0.4  # Lower base probability for more controlled growth
                        
                        # Favor rightward expansion
                        if dc > 0:
                            growth_prob *= 1.5
                        
                        # Gradual shrinking as we move away from origin
                        if new_r < grid_size//3 or new_r > 3*grid_size//4:
                            growth_prob *= 0.6
                        if new_c > 2*grid_size//3:
                            growth_prob *= 0.5
                        
                        # Encourage more contiguous growth
                        blob_neighbors = self._count_blob_neighbors(blob_mask, new_r, new_c)
                        if blob_neighbors >= 2:
                            growth_prob *= 1.3
                        
                        if random.random() < growth_prob:
                            blob_mask[new_r, new_c] = True
                            next_to_fill.append((new_r, new_c))
                            current_size += 1
                            
                            if current_size >= target_size:
                                break
                
                if current_size >= target_size:
                    break
            
            to_fill = next_to_fill
        
        # Phase 2: If extending to top-right, create extension
        if extend_to_top_right:
            self._create_top_right_extension(grid_size, blob_mask, forbidden_areas)
        
        return blob_mask
    
    def _count_blob_neighbors(self, blob_mask, r, c):
        """Count how many blob neighbors a cell has"""
        grid_size = blob_mask.shape[0]
        count = 0
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if blob_mask[nr, nc]:
                        count += 1
        return count
    
    def _create_top_right_extension(self, grid_size, blob_mask, forbidden_areas):
        """Create a dramatic extension to the top-right corner"""
        # Find rightmost blob cells as starting points for extension
        rightmost_cells = []
        for r in range(grid_size):
            for c in range(grid_size-1, -1, -1):
                if blob_mask[r, c]:
                    rightmost_cells.append((r, c))
                    break
        
        if not rightmost_cells:
            return
        
        # Choose extension starting points from upper portion of rightmost cells
        upper_rightmost = [cell for cell in rightmost_cells if cell[0] < 2*grid_size//3]
        if not upper_rightmost:
            upper_rightmost = rightmost_cells[:len(rightmost_cells)//2]
        
        # Create multiple extension paths to top-right
        for start_r, start_c in upper_rightmost[:3]:  # Use up to 3 starting points
            self._grow_extension_path(grid_size, blob_mask, forbidden_areas, start_r, start_c)
    
    def _grow_extension_path(self, grid_size, blob_mask, forbidden_areas, start_r, start_c):
        """Grow an extension path toward top-right corner"""
        active_cells = [(start_r, start_c)]
        extension_size = 0
        max_extension_size = grid_size * 2
        
        while active_cells and extension_size < max_extension_size:
            if not active_cells:
                break
            
            current_r, current_c = random.choice(active_cells)
            
            # Strong bias toward top-right
            directions = [(0,1), (-1,1), (-1,0)]  # right, up-right, up
            # Add some other directions with lower probability
            if random.random() < 0.3:
                directions.extend([(1,1), (1,0)])  # down-right, down
            
            random.shuffle(directions)
            
            grew = False
            for dr, dc in directions:
                new_r, new_c = current_r + dr, current_c + dc
                if (0 <= new_r < grid_size and 0 <= new_c < grid_size and 
                    not blob_mask[new_r, new_c] and (new_r, new_c) not in forbidden_areas):
                    
                    growth_prob = 0.6  # More controlled growth
                    
                    # Strong bias toward top-right corner
                    if dc > 0:  # Moving right
                        growth_prob *= 1.5
                    if dr < 0:  # Moving up
                        growth_prob *= 1.4
                    if dc > 0 and dr < 0:  # Moving up-right
                        growth_prob *= 1.8
                    
                    # Encourage contiguous growth
                    blob_neighbors = self._count_blob_neighbors(blob_mask, new_r, new_c)
                    if blob_neighbors >= 2:
                        growth_prob *= 1.2
                    
                    # Extra bonus for reaching top-right area
                    if new_r < grid_size//3 and new_c > 2*grid_size//3:
                        growth_prob *= 2.0
                    
                    # Reduce probability as we get very close to edges
                    if new_r < 2 or new_c > grid_size - 3:
                        growth_prob *= 0.7
                    
                    if random.random() < growth_prob:
                        blob_mask[new_r, new_c] = True
                        active_cells.append((new_r, new_c))
                        extension_size += 1
                        grew = True
                        break
            
            if not grew:
                active_cells.remove((current_r, current_c))
    
    def _choose_block_colors(self, num_blocks, available_colors):
        """Choose colors for 2x2 blocks according to the rules - always 3 blocks with A, A, B pattern"""
        # Always 3 blocks: first two same color, third different
        first_color = random.choice(available_colors)
        remaining_colors = [c for c in available_colors if c != first_color]
        third_color = random.choice(remaining_colors)
        return [first_color, first_color, third_color]
    
    def _place_2x2_blocks(self, grid, grid_size, num_blocks, block_colors, bottom_right_color, background_color):
        """Place 2x2 blocks at fixed positions with single line separation - always 3 blocks"""
        
        # Place top-left area blocks with single line separation - always 3 blocks
        for i in range(3):
            start_row = 1
            start_col = 1 + i * 3  # 2 cells for block + 1 cell gap
            color = block_colors[i]
            
            # Ensure block fits in grid
            if start_row + 1 < grid_size and start_col + 1 < grid_size:
                # Place the block (area should already be clear due to forbidden areas)
                grid[start_row:start_row+2, start_col:start_col+2] = color
        
        # Place bottom-right corner block
        bottom_right_row = grid_size - 4
        bottom_right_col = grid_size - 4
        if bottom_right_row >= 0 and bottom_right_col >= 0:
            # Place the block (area should already be clear due to forbidden areas)
            grid[bottom_right_row:bottom_right_row+2, bottom_right_col:bottom_right_col+2] = bottom_right_color
    
    def _add_single_cell_in_blob(self, grid, blob_mask, cell_color, blob_color):
        """Add exactly one single cell in first column within blob area, ensuring it's embedded"""
        # Find all positions in first column that are within blob and have blob cells above and below
        embedded_positions = []
        
        for r in range(2, grid.shape[0] - 2):  # Leave room for cells above and below
            if blob_mask[r, 0] and grid[r, 0] == blob_color:  # Must be blob cell
                # Check if there are blob cells above and below
                has_above = any(blob_mask[r - i, 0] for i in range(1, 3))  # Check 2 cells above
                has_below = any(blob_mask[r + i, 0] for i in range(1, 3))  # Check 2 cells below
                
                if has_above and has_below:
                    embedded_positions.append(r)
        
        if not embedded_positions:
            # Fallback: just find any blob cell in first column
            for r in range(grid.shape[0]):
                if blob_mask[r, 0] and grid[r, 0] == blob_color:
                    embedded_positions.append(r)
        
        if embedded_positions:
            # Select exactly one position to place the single cell
            selected_row = random.choice(embedded_positions)
            grid[selected_row, 0] = cell_color
    
    def transform_input(self, grid, taskvars):

        import numpy as np

        output = grid.copy()
        n = grid.shape[0]

        # -------------------------
        # Identify background color
        # -------------------------
        background = grid[0,0]

        # -------------------------
        # Identify blob color
        # -------------------------
        values, counts = np.unique(grid, return_counts=True)
        order = np.argsort(counts)[::-1]

        blob_color = background
        for idx in order:
            if values[idx] != background:
                blob_color = values[idx]
                break

        # -------------------------
        # Extract block colors
        # -------------------------
        block_colors = []

        for i in range(3):

            r = 1
            c = 1 + i*3

            if r+1 < n and c+1 < n:

                if (grid[r,c] == grid[r+1,c] ==
                    grid[r,c+1] == grid[r+1,c+1]):

                    block_colors.append(grid[r,c])

        # bottom-right block color
        br_color = None
        br_r = n-4
        br_c = n-4

        if br_r >= 0 and br_c >= 0:
            if (grid[br_r,br_c] == grid[br_r+1,br_c] ==
                grid[br_r,br_c+1] == grid[br_r+1,br_c+1]):

                br_color = grid[br_r,br_c]

        # -------------------------
        # Find starting position
        # -------------------------
        start = None
        first_block_color = block_colors[0] if block_colors else None

        if first_block_color is not None:

            for r in range(n):
                if grid[r,0] == first_block_color:
                    start = (r,0)
                    break

        if start is None:
            return output

        r,c = start

        # -------------------------
        # Path color pattern
        # -------------------------
        if len(block_colors) >= 3:
            pattern = [block_colors[0], block_colors[0], block_colors[2]]
        else:
            pattern = [first_block_color]

        pattern_index = 0
        for i,v in enumerate(pattern):
            if v == grid[r,c]:
                pattern_index = i
                break

        last_direction = None

        # -------------------------
        # Build path
        # -------------------------
        while True:

            moved_right = False

            while (c+1 < n and grid[r,c+1] == blob_color):

                c += 1
                pattern_index = (pattern_index+1) % len(pattern)
                output[r,c] = pattern[pattern_index]

                moved_right = True
                last_direction = "horizontal"

            if moved_right and r>0 and grid[r-1,c] == blob_color and br_color is not None:

                for nr,nc in [(r,c+1),(r+1,c),(r-1,c),(r,c-1)]:

                    if 0<=nr<n and 0<=nc<n and output[nr,nc]==background:
                        output[nr,nc] = br_color
                        break

            moved_up = False

            while (r>0 and grid[r-1,c] == blob_color):

                r -= 1
                pattern_index = (pattern_index+1) % len(pattern)
                output[r,c] = pattern[pattern_index]

                moved_up = True
                last_direction = "vertical"

            if moved_up and c+1<n and grid[r,c+1] == blob_color and br_color is not None:

                for nr,nc in [(r,c+1),(r+1,c),(r-1,c),(r,c-1)]:

                    if 0<=nr<n and 0<=nc<n and output[nr,nc]==background:
                        output[nr,nc] = br_color
                        break

            if not moved_right and not moved_up:
                break

        # -------------------------
        # Final indicator
        # -------------------------
        if br_color is not None:

            if last_direction == "vertical" and r>0:

                if output[r-1,c] == background:
                    output[r-1,c] = br_color

            elif last_direction == "horizontal" and c<n-1:

                if output[r,c+1] == background:
                    output[r,c+1] = br_color

        # -------------------------
        # Remove all blocks
        # -------------------------
        for i in range(3):

            rr = 1
            cc = 1 + i*3

            if rr+1 < n and cc+1 < n:
                output[rr:rr+2,cc:cc+2] = background

        if br_r >= 0 and br_c >= 0:
            output[br_r:br_r+2,br_c:br_c+2] = background

        return output
    
