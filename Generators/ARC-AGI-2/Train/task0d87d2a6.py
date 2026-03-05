from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject
from Framework.input_library import retry, create_object, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task0d87d2a6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains one or two pairs of {color('line_color')} cells and several {color('block')} rectangular objects, with all remaining cells being empty (0).",
            "Each pair of {color('line_color')} cells must be placed such that the two cells lie at the start and end of a row or a column, excluding the four grid corners — no {color('line_color')} cells may be placed in the corners.",
            "If there are two pairs, ensure that one pair is placed in a row and the other in a column.",
            "All {color('line_color')} cells and {color('block')} rectangular objects must be completely separated from each other by at least one layer of empty (0) cells on all sides.",
            "Each {color('block')} rectangular object must be at least 2 cells wide and 2 cells long.",
            "The positions of both the {color('block')} rectangular objects and the {color('line_color')} cells must vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all pairs of {color('line_color')} cells.",
            "For each identified pair of {color('line_color')} cells, fill the entire row with {color('line_color')} if the pair is placed at the start and end of a row, or fill the entire column if the pair is placed at the start and end of a column.",
            "Apply this rule to all valid pairs of {color('line_color')} cells in the grid.",
            "After drawing all such lines, change the color of any {color('block')} rectangular object to {color('line_color')} if it is touched or intersected by any of the drawn lines."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid with line pairs and block objects."""
        height = gridvars['height']
        width = gridvars['width']
        line_color = taskvars['line_color']
        block_color = taskvars['block']
        num_pairs = gridvars['num_pairs']
        num_blocks = gridvars['num_blocks']
        
        # We'll retry the entire generation process if we can't place all pairs
        for generation_attempt in range(20):
            # Initialize grid
            grid = np.zeros((height, width), dtype=int)
            
            # Track occupied cells (including buffer zones)
            occupied = set()
            
            def mark_occupied_with_buffer(cells):
                """Mark cells and their 1-cell buffer as occupied."""
                for r, c in cells:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                occupied.add((nr, nc))
            
            # Determine pair types
            pair_types = []
            if num_pairs == 1:
                pair_types = [random.choice(['row', 'col'])]
            else:
                pair_types = ['row', 'col']
                for _ in range(num_pairs - 2):
                    pair_types.append(random.choice(['row', 'col']))
                random.shuffle(pair_types)
            
            # Count how many of each type we need
            num_row_pairs = sum(1 for pt in pair_types if pt == 'row')
            num_col_pairs = sum(1 for pt in pair_types if pt == 'col')
            
            # First, place rectangular blocks strategically
            # Ensure blocks are distributed to allow for the needed line pairs
            blocks_info = []  # Store (row_range, col_range) for each block
            
            # Available rows/cols for placement (excluding corner rows/cols)
            available_rows = list(range(1, height - 1))
            available_cols = list(range(1, width - 1))
            random.shuffle(available_rows)
            random.shuffle(available_cols)
            
            # Try to place blocks that will allow for our needed pairs
            blocks_placed = 0
            for block_idx in range(num_blocks):
                max_attempts = 100
                placed = False
                
                for _ in range(max_attempts):
                    # Random block size (at least 2x2)
                    block_h = random.randint(2, min(5, height // 3))
                    block_w = random.randint(2, min(5, width // 3))
                    
                    # Random position
                    r = random.randint(0, height - block_h)
                    c = random.randint(0, width - block_w)
                    
                    # Check if block and buffer are free
                    block_cells = [(r + dr, c + dc) for dr in range(block_h) for dc in range(block_w)]
                    
                    # Check buffer zone
                    buffer_cells = set()
                    for br in range(r - 1, r + block_h + 1):
                        for bc in range(c - 1, c + block_w + 1):
                            if 0 <= br < height and 0 <= bc < width:
                                buffer_cells.add((br, bc))
                    
                    if any(cell in occupied for cell in buffer_cells):
                        continue
                    
                    # Place block
                    for cell in block_cells:
                        grid[cell] = block_color
                    
                    blocks_info.append((range(r, r + block_h), range(c, c + block_w)))
                    mark_occupied_with_buffer(block_cells)
                    placed = True
                    blocks_placed += 1
                    break
                
                if not placed and block_idx < 2:
                    # Must place at least 2 blocks
                    break
            
            if blocks_placed < 2:
                continue  # Retry entire generation
            
            # Now place line pairs such that they intersect at least one block
            line_pairs_placed = 0
            
            for pair_type in pair_types:
                max_attempts = 100
                placed = False
                
                for _ in range(max_attempts):
                    if pair_type == 'row':
                        # Pick a row, avoid corners
                        row = random.randint(1, height - 2)
                        cells = [(row, 0), (row, width - 1)]
                        
                        # Check if occupied
                        if any(cell in occupied for cell in cells):
                            continue
                        
                        # Check if this row intersects at least one block
                        intersects_block = any(row in row_range for row_range, _ in blocks_info)
                        
                        if not intersects_block:
                            continue
                        
                        line_pairs_placed += 1
                        for cell in cells:
                            grid[cell] = line_color
                            mark_occupied_with_buffer([cell])
                        placed = True
                        break
                    else:  # column
                        col = random.randint(1, width - 2)
                        cells = [(0, col), (height - 1, col)]
                        
                        # Check if occupied
                        if any(cell in occupied for cell in cells):
                            continue
                        
                        # Check if this column intersects at least one block
                        intersects_block = any(col in col_range for _, col_range in blocks_info)
                        
                        if not intersects_block:
                            continue
                        
                        line_pairs_placed += 1
                        for cell in cells:
                            grid[cell] = line_color
                            mark_occupied_with_buffer([cell])
                        placed = True
                        break
                
                if not placed:
                    # Couldn't place this pair, try entire generation again
                    break
            
            # Check if we successfully placed all pairs
            if line_pairs_placed == num_pairs:
                return grid
        
        # If we get here, we failed after all attempts
        raise ValueError(f"Could not generate valid grid after 20 attempts (need {num_pairs} pairs, {num_row_pairs} rows, {num_col_pairs} cols)")
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input grid according to the transformation reasoning chain."""
        output = grid.copy()
        line_color = taskvars['line_color']
        block_color = taskvars['block']
        height, width = grid.shape
        
        # Find all line_color cells
        line_cells = list(zip(*np.where(grid == line_color)))
        
        # Identify pairs and draw lines
        lines_drawn = []
        
        # Check for row pairs
        for r in range(height):
            row_cells = [(row, col) for row, col in line_cells if row == r]
            if len(row_cells) == 2:
                # Check if they're at start and end
                cols = sorted([col for _, col in row_cells])
                if cols == [0, width - 1]:
                    # Fill entire row
                    output[r, :] = line_color
                    lines_drawn.append(('row', r))
        
        # Check for column pairs
        for c in range(width):
            col_cells = [(row, col) for row, col in line_cells if col == c]
            if len(col_cells) == 2:
                # Check if they're at start and end
                rows = sorted([row for row, _ in col_cells])
                if rows == [0, height - 1]:
                    # Fill entire column
                    output[:, c] = line_color
                    lines_drawn.append(('col', c))
        
        # Find all block objects and check if they're touched by lines
        blocks = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        blocks = blocks.with_color(block_color)
        
        for block in blocks:
            touched = False
            for r, c, _ in block.cells:
                # Check if this cell is now line_color (was touched by a line)
                if output[r, c] == line_color:
                    touched = True
                    break
            
            if touched:
                # Change entire block to line_color
                for r, c, _ in block.cells:
                    output[r, c] = line_color
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and generate train/test grids."""
        # Choose colors
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        line_color = all_colors[0]
        block_color = all_colors[1]
        
        taskvars = {
            'line_color': line_color,
            'block': block_color
        }
        
        # Create training examples (3-6)
        num_train = random.randint(3, 6)
        train_pairs = []
        
        for i in range(num_train):
            # Random grid size
            height = random.randint(12, 30)
            width = random.randint(12, 30)
            
            # 1 or 2 pairs for training
            num_pairs = random.randint(1, 2)
            num_blocks = random.randint(2, 5)
            
            gridvars = {
                'height': height,
                'width': width,
                'num_pairs': num_pairs,
                'num_blocks': num_blocks
            }
            
            # Generate valid input (create_input now handles retries internally)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with 3 pairs (2 rows, 1 column)
        test_height = random.randint(15, 30)
        test_width = random.randint(15, 30)
        test_num_blocks = random.randint(3, 6)
        
        # For test, we need specific pair configuration
        # We'll create a custom gridvars that forces 2 row + 1 col arrangement
        max_outer_attempts = 10
        test_input = None
        
        for outer_attempt in range(max_outer_attempts):
            try:
                # Try multiple internal attempts with retry logic
                for inner_attempt in range(20):
                    grid = np.zeros((test_height, test_width), dtype=int)
                    occupied = set()
                    blocks_info = []
                    
                    def mark_occupied_with_buffer(cells):
                        for r, c in cells:
                            for dr in [-1, 0, 1]:
                                for dc in [-1, 0, 1]:
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < test_height and 0 <= nc < test_width:
                                        occupied.add((nr, nc))
                    
                    # First, place blocks
                    blocks_placed = 0
                    for _ in range(test_num_blocks * 10):
                        if blocks_placed >= test_num_blocks:
                            break
                        
                        block_h = random.randint(2, min(5, test_height // 3))
                        block_w = random.randint(2, min(5, test_width // 3))
                        r = random.randint(0, test_height - block_h)
                        c = random.randint(0, test_width - block_w)
                        
                        block_cells = [(r + dr, c + dc) for dr in range(block_h) for dc in range(block_w)]
                        buffer_cells = set()
                        for br in range(r - 1, r + block_h + 1):
                            for bc in range(c - 1, c + block_w + 1):
                                if 0 <= br < test_height and 0 <= bc < test_width:
                                    buffer_cells.add((br, bc))
                        
                        if any(cell in occupied for cell in buffer_cells):
                            continue
                        
                        for cell in block_cells:
                            grid[cell] = block_color
                        blocks_info.append((range(r, r + block_h), range(c, c + block_w)))
                        mark_occupied_with_buffer(block_cells)
                        blocks_placed += 1
                    
                    if blocks_placed < 2:
                        continue
                    
                    # Place 2 row pairs that intersect blocks
                    row_pairs_placed = 0
                    for _ in range(200):
                        if row_pairs_placed >= 2:
                            break
                        row = random.randint(1, test_height - 2)
                        cells = [(row, 0), (row, test_width - 1)]
                        
                        if any(cell in occupied for cell in cells):
                            continue
                        
                        intersects_block = any(row in row_range for row_range, _ in blocks_info)
                        if not intersects_block:
                            continue
                        
                        for cell in cells:
                            grid[cell] = line_color
                            mark_occupied_with_buffer([cell])
                        row_pairs_placed += 1
                    
                    if row_pairs_placed < 2:
                        continue
                    
                    # Place 1 column pair that intersects a block
                    col_placed = False
                    for _ in range(200):
                        col = random.randint(1, test_width - 2)
                        cells = [(0, col), (test_height - 1, col)]
                        
                        if any(cell in occupied for cell in cells):
                            continue
                        
                        intersects_block = any(col in col_range for _, col_range in blocks_info)
                        if not intersects_block:
                            continue
                        
                        for cell in cells:
                            grid[cell] = line_color
                            mark_occupied_with_buffer([cell])
                        col_placed = True
                        break
                    
                    if col_placed:
                        test_input = grid
                        break
                
                if test_input is not None:
                    break
                    
            except Exception as e:
                # Try again with different dimensions
                test_height = random.randint(15, 30)
                test_width = random.randint(15, 30)
                continue
        
        if test_input is None:
            raise ValueError("Could not create valid test grid")
        
        test_output = self.transform_input(test_input, taskvars)
        
        test_pairs = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_pairs,
            'test': test_pairs
        }

