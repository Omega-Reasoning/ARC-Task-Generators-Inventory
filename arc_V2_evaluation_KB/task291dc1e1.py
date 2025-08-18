from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task291dc1e1Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "The first row and either the first or last column are filled with {color('color1')} and {color('color2')}, with one color assigned to the row and the other to the column.",
            "The cell at the intersection of the filled row and column is empty: it is the first cell of the first row if the first column is filled, otherwise the last cell of the first row if the last column is filled.",
            "If the first row is filled with {color('color1')}, the grid has more rows than columns.",
            "If {color('color1')} is used in the first or last column, the grid has 8 rows; otherwise, it has 8 columns (equivalently, the {color('color1')} line always has length 8).",
            "All remaining cells are filled with {color('background_color')}.",
            "The grid contains several rectangular blocks; one dimension is exactly 2 (e.g., 2×2, 2×3, 2×6, 6×2, …). The first block starts at (2,2). All blocks are placed on {color('background_color')}.",
            "The side of length 2 is aligned with the line filled by {color('color1')}: If {color('color1')} is in the first row (→ 8 columns), blocks are (2×k) and arranged as vertical stacks with exactly one empty column between stacks; specifically at column pairs 2–3 and 5–6. If {color('color1')} is in the first or last column (→ 8 rows), blocks are (k×2) and arranged as horizontal layers with exactly one empty row between layers; for example at row pairs 1–2 and 4–5."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by first identifying the empty (0) cell in the first row, which can be either the first cell of the last cell of the first rows.",
            "Next, collect all smaller rectangular blocks which are located in the grid and arranged in rows or columns  and restack them in the output grid, keeping them in a sequence: start with the first block (the one diagonally positioned relative to the empty cell in the first row), then continue in the same set of columns or rows, depending on whether color1 appears in a row or a column, respectively. Once all blocks in the set are added, move to the next set of columns or rows and stack the remaining blocks.",
            "Blocks are stacked together without gaps.",
            "The result is a smaller, tightly packed grid containing all blocks in order.",
            "The output grid is filled with the same background color as the input grid, except for the blocks.",
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        color1 = taskvars['color1']
        color2 = taskvars['color2']
        background_color = taskvars['background_color']
        
        # Randomly decide if color1 is in row (True) or column (False)
        color1_in_row = random.choice([True, False])
        
        if color1_in_row:
            # color1 in first row, more rows than columns, 8 columns total
            cols = 8
            rows = random.randint(cols + 1, min(30, cols + 10))  # Ensure more rows than columns
            grid = np.full((rows, cols), background_color, dtype=int)
            
            # Fill first row with color1
            grid[0, :] = color1
            
            # Randomly choose first or last column for color2
            col_choice = random.choice([0, cols-1])
            grid[:, col_choice] = color2
            
            # Empty intersection cell (always 0, not background_color)
            grid[0, col_choice] = 0
            
            # Add 2×k blocks in vertical stacks at columns 2-3 and 5-6
            block_cols = [(2, 3), (5, 6)]
            block_colors = [c for c in range(1, 10) if c not in [color1, color2, background_color]]
            
            for col_pair in block_cols:
                if len(block_colors) == 0:
                    break
                    
                # Create blocks with random heights starting from row 2
                current_row = 2
                while current_row < rows - 1:  # Leave space for at least one block
                    if len(block_colors) == 0:
                        break
                        
                    max_height = min(6, rows - current_row)
                    if max_height < 2:
                        break
                    block_height = random.randint(1, max_height // 2) * 2  # Ensure even height
                    block_color = random.choice(block_colors)
                    block_colors.remove(block_color)
                    
                    # Place 2×k block (width=2, height=block_height)
                    grid[current_row:current_row+block_height, col_pair[0]:col_pair[1]+1] = block_color
                    
                    current_row += block_height + 1  # +1 for gap
                    
        else:
            # color1 in first or last column, 8 rows total
            rows = 8
            cols = random.randint(10, 30)
            grid = np.full((rows, cols), background_color, dtype=int)
            
            # Randomly choose first or last column for color1
            col_choice = random.choice([0, cols-1])
            grid[:, col_choice] = color1
            
            # Fill first row with color2
            grid[0, :] = color2
            
            # Empty intersection cell (always 0, not background_color)
            grid[0, col_choice] = 0
            
            # Add k×2 blocks in horizontal layers at rows 2-3 and 5-6
            block_rows = [(2, 3), (5, 6)]
            block_colors = [c for c in range(1, 10) if c not in [color1, color2, background_color]]
            
            for row_pair in block_rows:
                if len(block_colors) == 0:
                    break
                    
                # Create blocks with random widths starting from column 2
                current_col = 2
                while current_col < cols - 2:  # Leave space to avoid overlapping with last column
                    if len(block_colors) == 0:
                        break
                        
                    max_width = min(6, cols - current_col - 1)  # -1 to avoid last column
                    if max_width < 2:
                        break
                    block_width = random.randint(1, max_width // 2) * 2  # Ensure even width
                    block_color = random.choice(block_colors)
                    block_colors.remove(block_color)
                    
                    # Place k×2 block (height=2, width=block_width)
                    grid[row_pair[0]:row_pair[1]+1, current_col:current_col+block_width] = block_color
                    
                    current_col += block_width + 1  # +1 for gap
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        color1 = taskvars['color1']
        color2 = taskvars['color2']
        background_color = taskvars['background_color']
        
        rows, cols = grid.shape
        
        # Find if color1 is in row or column
        color1_in_row = np.any(grid[0, :] == color1)
        
        # Find empty intersection cell (should be 0)
        if color1_in_row:
            # color1 in first row, find which column has the empty cell (0)
            empty_col = 0 if grid[0, 0] == 0 else cols - 1
        else:
            # color1 in column, intersection is at first row
            empty_col = 0 if grid[0, 0] == 0 else cols - 1
        
        # Find all blocks (non-background, non-color1, non-color2, non-zero connected components)
        mask = (grid != background_color) & (grid != color1) & (grid != color2) & (grid != 0)
        blocks = find_connected_objects(np.where(mask, grid, 0), diagonal_connectivity=False, background=0)
        
        if len(blocks) == 0:
            return np.array([[background_color]])
        
        # Sort blocks by position based on empty cell position
        if color1_in_row:

            
            sorted_blocks = []
            for obj in blocks.objects:
                min_col = min(c for r, c, _ in obj.cells)
                min_row = min(r for r, c, _ in obj.cells)
                
                if empty_col == cols - 1:
                    # Empty cell at top-right, start with rightmost column (5-6)
                    col_priority = 0 if min_col == 5 else 1
                else:
                    # Empty cell at top-left, start with leftmost column (2-3)  
                    col_priority = 0 if min_col == 2 else 1
                sort_key = (col_priority, min_col, min_row)
                sorted_blocks.append((sort_key, obj))
                
            sorted_blocks.sort(key=lambda x: x[0])
            blocks.objects = [obj for _, obj in sorted_blocks]
        else:

            
            sorted_blocks = []
            for obj in blocks.objects:
                min_row = min(r for r, c, _ in obj.cells)
                min_col = min(c for r, c, _ in obj.cells)
                
                if empty_col == cols - 1:
                    # Empty cell at top-right (color1 in last column), start with top row, rightmost first
                    row_priority = 0 if min_row == 2 else 1
                    sort_key = (row_priority, min_row, -min_col)  # Negative to sort rightmost first
                else:
                    # Empty cell at top-left (color1 in first column), start with top row, leftmost first
                    row_priority = 0 if min_row == 2 else 1
                    sort_key = (row_priority, min_row, min_col)
                sorted_blocks.append((sort_key, obj))
                
            sorted_blocks.sort(key=lambda x: x[0])
            blocks.objects = [obj for _, obj in sorted_blocks]
        
        # Calculate output grid size based on block arrangement
        num_blocks = len(blocks)
        total_height = num_blocks * 2
        max_width = 0
        
        for block in blocks:
            block_array = block.to_array()
            h, w = block_array.shape
            
            if color1_in_row:
                # When color1 is in first row, blocks get rotated to horizontal
                if h > 2:
                    # Calculate new width after rotation: h×w becomes 2×(h*w/2)
                    total_cells = h * w
                    new_width = total_cells // 2
                    max_width = max(max_width, new_width)
                else:
                    max_width = max(max_width, w)
            else:
                # When color1 is in column, blocks maintain original width
                max_width = max(max_width, w)
        
        if total_height == 0 or max_width == 0:
            return np.array([[background_color]])
        
        # Create output grid and stack blocks
        output_grid = np.full((total_height, max_width), background_color, dtype=int)
        current_row = 0
        
        for block in blocks:
            block_array = block.to_array()
            h, w = block_array.shape
            
            if color1_in_row:
                # When color1 is in first row, rotate blocks from vertical to horizontal
                # The block should be exactly 2 rows high but maintain total area
                if h > 2:
                    # Reshape vertical block to horizontal: h×w becomes 2×(h*w/2)
                    total_cells = h * w
                    new_width = total_cells // 2
                    
                    # Create horizontal version with same color
                    block_color = block_array[block_array != 0][0] if np.any(block_array != 0) else background_color
                    block_2rows = np.full((2, new_width), block_color, dtype=int)
                elif h == 2:
                    # Already 2 rows
                    block_2rows = block_array
                else:  # h == 1
                    # If block is 1 row, duplicate it to make 2 rows
                    block_2rows = np.vstack([block_array, block_array])
                    
                # Update width for centering
                w = block_2rows.shape[1]
            else:
                # When color1 is in column, blocks stay as 2 rows high
                if h == 1:
                    # If block is 1 row, duplicate it to make 2 rows
                    block_2rows = np.vstack([block_array, block_array])
                elif h == 2:
                    # Already 2 rows
                    block_2rows = block_array
                else:
                    # If more than 2 rows, take first 2 rows
                    block_2rows = block_array[:2, :]
            
            # Center horizontally in the output grid
            start_col = (max_width - w) // 2
            output_grid[current_row:current_row+2, start_col:start_col+w] = block_2rows
            current_row += 2
            
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random task variables
        colors = list(range(1, 10))
        random.shuffle(colors)
        
        taskvars = {
            'color1': colors[0],
            'color2': colors[1], 
            'background_color': colors[2]
        }
        
        # Generate 3-5 train examples and 1 test example
        num_train = random.randint(3, 5)
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

