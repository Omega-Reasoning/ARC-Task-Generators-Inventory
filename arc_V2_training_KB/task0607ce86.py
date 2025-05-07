from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task0607ce86Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have {vars['col']} columns and a varying number of rows.",
            "They contain rectangular objects, multiple single-colored cells, with the remaining cells being empty (0).",
            "One of the colors, in each rectangular object must be {color('object')}.",
            "The first rectangular object begins either from position (1,1) or (1,2), and there are either 6 or 9 rectangular objects in total, all of the same size.",
            "If there are 6 objects, they are arranged into two groups of adjacent columns, with 3 objects in each group. If there are 9 objects, they are arranged into three such groups.",
            "Along with these column groups, there must also be groups of adjacent rows, so that the rectangular objects are placed in fixed row and column groups, maintaining both vertical and horizontal alignment.",
            "Each group of adjacent rows and columns must be separated from the next by exactly one row or column, respectively.",
            "To construct each rectangular object, begin with a horizontal colored strip placed in either the top or bottom row of the rectangle.",
            "This horizontal strip must span the full width of the rectangle.",
            "The remaining area of the rectangle is filled with additional strips—either horizontal or vertical.",
            "Each rectangular object must be horizontally symmetrical, meaning the left half should be a reflection of the right half.",
            "After creating one such object, replicate it to get a total of 6 or 9 objects, placing each in its respective group and aligned position as described.",
            "Once all rectangular objects are placed (each being identical), randomly scatter multiple single-colored cells across the grid. These may overlap empty cells or already-colored cells.",
            "The single-colored cells should mostly be {color('object')}, and occasionally use one of the colors used in the rectangular object.",
            "Ensure that at least one rectangular object remains completely intact, with no single-colored cell overlapping any of its cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all rectangular objects and single-colored cells. The first rectangular object starts from either (1,1) or (1,2).",
            "First, remove all extra single-colored cells from the grid—these may or may not be connected to each other or to the rectangular objects, and may distort the rectangular shapes.",
            "Then, identify the rectangular object that is horizontally symmetrical and has either the first row or the last row completely filled with the same colored cells.",
            "Replace all remaining rectangular objects in the grid with exact copies of this identified symmetrical rectangular object.",
            "The final grid should contain only the 6 or 9 (depending on the number of rectangular objects in input) identical rectangular objects, with no extra single-colored cells."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'col': random.randint(18, 30),
            'object': random.randint(1, 9)
        }
        
        # Create varied train and test examples
        train_data = []
        
        # Create 2 input grids with 6 objects and top row filled
        for _ in range(2):
            gridvars = {
                'num_objects': 6,
                'strip_position': 'top',
                'strip_arrangement': 'vertical'
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Create 1 input grid with 9 objects and bottom row filled
        gridvars = {
            'num_objects': 9,
            'strip_position': 'bottom',
            'strip_arrangement': 'horizontal'
        }
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_data.append({'input': input_grid, 'output': output_grid})
        
        # Create 1 test grid, randomly choosing parameters
        test_gridvars = {
            'num_objects': random.choice([6, 9]),
            'strip_position': random.choice(['top', 'bottom']),
            'strip_arrangement': random.choice(['horizontal', 'vertical'])
        }
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_data,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars, gridvars):
        """Create an input grid with rectangular objects and scattered cells."""
        # Extract grid variables
        num_objects = gridvars['num_objects']
        strip_position = gridvars['strip_position']
        strip_arrangement = gridvars['strip_arrangement']
        
        # Determine the number of column groups and rows based on num_objects
        if num_objects == 6:
            column_groups = 2
            objects_per_group = 3
        else:  # 9 objects
            column_groups = 3
            objects_per_group = 3
        
        # Choose colors for the blocks
        object_color = taskvars['object']
        # Choose two other colors different from object_color
        available_colors = [c for c in range(1, 10) if c != object_color]
        color_a = random.choice(available_colors)
        available_colors.remove(color_a)
        color_b = random.choice(available_colors)
        
        # Keep track of colors used in the rectangular objects
        colors_used = [object_color, color_a, color_b]
        
        # Define block dimensions - ensure blocks are not too large
        block_width = random.choice([3, 4])
        block_height = random.choice([4, 5])
        
        # Create a template for a single block
        block = np.zeros((block_height, block_width), dtype=int)
        
        # Fill the block according to specifications
        if strip_position == 'top':
            # Fill top row with object_color
            block[0, :] = object_color
            
            if strip_arrangement == 'vertical':
                # Add four vertical stripes beneath the top row
                for i in range(1, block_height):
                    block[i, 0] = color_a
                    block[i, block_width-1] = color_a
                    block[i, 1:block_width-1] = color_b
            else:  # horizontal
                # Add horizontal stripes
                stripe_height = (block_height - 1) // 4
                remainder = (block_height - 1) % 4
                
                current_row = 1
                # First stripe with color_a
                block[current_row:current_row+stripe_height, :] = color_a
                current_row += stripe_height
                
                # Two middle stripes with color_b
                middle_height = 2 * stripe_height + remainder
                block[current_row:current_row+middle_height, :] = color_b
                current_row += middle_height
                
                # Last stripe with color_a
                block[current_row:, :] = color_a
                
        else:  # bottom
            # Fill bottom row with object_color
            block[block_height-1, :] = object_color
            
            if strip_arrangement == 'vertical':
                # Add four vertical stripes
                for i in range(block_height-1):
                    block[i, 0] = color_a
                    block[i, block_width-1] = color_a
                    block[i, 1:block_width-1] = color_b
            else:  # horizontal
                # Add horizontal stripes
                stripe_height = (block_height - 1) // 4
                remainder = (block_height - 1) % 4
                
                # First stripe with color_a
                block[0:stripe_height, :] = color_a
                
                # Two middle stripes with color_b
                middle_height = 2 * stripe_height + remainder
                block[stripe_height:stripe_height+middle_height, :] = color_b
                
                # Last stripe with color_a
                block[stripe_height+middle_height:block_height-1, :] = color_a
        
        # Ensure horizontal symmetry - create the perfectly symmetrical template
        for r in range(block_height):
            for c in range(block_width // 2):
                block[r, block_width-1-c] = block[r, c]
        
        # Calculate grid dimensions
        total_cols = column_groups * block_width + (column_groups - 1)
        total_rows = objects_per_group * block_height + (objects_per_group - 1)
        
        # Add padding to grid for the starting position options
        start_row = 1
        start_col = random.choice([1, 2])
        
        grid_width = max(total_cols + start_col + 1, taskvars['col'])  # Ensure minimum width
        grid_height = total_rows + start_row + 1  # +1 for bottom padding
        
        # Initialize the grid with zeros
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Randomly select one block to keep symmetrical and intact
        intact_row_group = random.randint(0, objects_per_group - 1)
        intact_col_group = random.randint(0, column_groups - 1)
        intact_row_start = start_row + intact_row_group * (block_height + 1)
        intact_col_start = start_col + intact_col_group * (block_width + 1)
        
        # Store the position of the intact block for the transform function
        self._intact_block_position = (intact_row_start, intact_col_start, block_height, block_width)
        
        # Track all block positions
        block_positions = []
        
        # Place blocks in the grid
        for group_idx in range(column_groups):
            col_start = start_col + group_idx * (block_width + 1)
            
            for obj_idx in range(objects_per_group):
                row_start = start_row + obj_idx * (block_height + 1)
                
                # Check if this block would fit in the grid
                if (row_start + block_height <= grid_height and 
                    col_start + block_width <= grid_width):
                    
                    # Place the block
                    block_copy = block.copy()
                    
                    # If this is not the intact block, deliberately make it asymmetrical
                    if row_start != intact_row_start or col_start != intact_col_start:
                        # Make 1-2 strategic modifications to break symmetry
                        # Choose 1 or 2 positions in the block to modify
                        num_modifications = random.randint(1, 2)
                        
                        for _ in range(num_modifications):
                            # Choose a position on the left side of the block
                            mod_row = random.randint(1, block_height - 2)  # Avoid top/bottom rows
                            mod_col = random.randint(0, (block_width // 2) - 1)  # Left side only
                            
                            # Change the color to another color used in the block
                            current_color = block_copy[mod_row, mod_col]
                            new_color = random.choice([c for c in colors_used if c != current_color])
                            
                            # Apply the change only to one side to break symmetry
                            block_copy[mod_row, mod_col] = new_color
                    
                    grid[row_start:row_start+block_height, col_start:col_start+block_width] = block_copy
                    
                    # Record the position of this block
                    block_positions.append((row_start, col_start, block_height, block_width))
        
        # Remember the intact block's coordinates
        intact_block_coords = (intact_row_start, intact_col_start, block_height, block_width)
        
        # Add scattered single-colored cells
        num_scattered = random.randint(grid_width * grid_height // 15, grid_width * grid_height // 8)
        
        for _ in range(num_scattered):
            r = random.randint(0, grid_height-1)
            c = random.randint(0, grid_width-1)
            
            # Skip if this is part of the intact block or its surrounding cells
            row_start, col_start, block_h, block_w = intact_block_coords
            if (row_start-1 <= r <= row_start + block_h and 
                col_start-1 <= c <= col_start + block_w):
                continue
            
            # Choose color from the colors used in the rectangular block
            # with higher probability for object_color
            if random.random() < 0.7:
                color = object_color
            else:
                color = random.choice([color_a, color_b])
                
            grid[r, c] = color
        
        # Ensure the grid has exactly col columns (as per taskvars)
        if grid.shape[1] > taskvars['col']:
            grid = grid[:, :taskvars['col']]
        elif grid.shape[1] < taskvars['col']:
            padding = np.zeros((grid.shape[0], taskvars['col'] - grid.shape[1]), dtype=int)
            grid = np.hstack((grid, padding))
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """Transform the input grid by finding the intact symmetrical block and replicating it."""
        # Create a copy of the grid
        result = np.zeros_like(grid)
        
        # Extract object color
        object_color = taskvars['object']
        
        # Step 1: Look for the symmetrical block
        symmetrical_block = None
        symmetrical_block_position = None
        
        # Scan through the grid looking for rectangular objects
        for r in range(1, grid.shape[0]-2):  # Need at least 3 rows
            for c in range(1, grid.shape[1]-2):  # Need at least 3 columns
                if grid[r, c] != 0:
                    # Try to find dimensions of this potential block
                    width = 0
                    for col in range(c, min(c+6, grid.shape[1])):
                        if col < grid.shape[1] and grid[r, col] != 0:
                            width += 1
                        else:
                            break
                    
                    height = 0
                    for row in range(r, min(r+6, grid.shape[0])):
                        if row < grid.shape[0] and grid[row, c] != 0:
                            height += 1
                        else:
                            break
                    
                    # Check if dimensions make sense for a block
                    if 3 <= width <= 5 and 3 <= height <= 6:
                        # Extract the potential block
                        if r + height <= grid.shape[0] and c + width <= grid.shape[1]:
                            block_region = grid[r:r+height, c:c+width].copy()
                            
                            # Check for horizontal symmetry
                            is_symmetric = True
                            for br in range(height):
                                for bc in range(width // 2):
                                    if block_region[br, bc] != block_region[br, width-1-bc]:
                                        is_symmetric = False
                                        break
                                if not is_symmetric:
                                    break
                            
                            # Check for filled top or bottom row
                            top_filled = all(block_region[0, bc] == block_region[0, 0] and block_region[0, 0] != 0 for bc in range(width))
                            bottom_filled = all(block_region[height-1, bc] == block_region[height-1, 0] and block_region[height-1, 0] != 0 for bc in range(width))
                            
                            # Check if this block contains the object color
                            has_object_color = object_color in block_region
                            
                            # Check if this block is isolated (surrounded by zeros)
                            is_isolated = True
                            for check_r in range(max(0, r-1), min(grid.shape[0], r+height+1)):
                                for check_c in range(max(0, c-1), min(grid.shape[1], c+width+1)):
                                    # Skip the block itself
                                    if r <= check_r < r+height and c <= check_c < c+width:
                                        continue
                                    
                                    # If surrounding cell is not empty
                                    if grid[check_r, check_c] != 0:
                                        is_isolated = False
                                        break
                                if not is_isolated:
                                    break
                            
                            # If this is a symmetrical, isolated block with filled top/bottom row
                            if is_symmetric and (top_filled or bottom_filled) and has_object_color and is_isolated:
                                symmetrical_block = block_region
                                symmetrical_block_position = (r, c, height, width)
                                break
                    
                if symmetrical_block is not None:
                    break
            if symmetrical_block is not None:
                break
        
        # If we didn't find a symmetrical block, look more broadly
        if symmetrical_block is None:
            # Search for any block that is symmetrical even if not isolated
            for r in range(1, grid.shape[0]-2):
                for c in range(1, grid.shape[1]-2):
                    if grid[r, c] != 0:
                        for height in [4, 5]:
                            for width in [3, 4]:
                                if (r + height <= grid.shape[0] and 
                                    c + width <= grid.shape[1] and
                                    np.any(grid[r:r+height, c:c+width] != 0)):
                                    
                                    block_region = grid[r:r+height, c:c+width].copy()
                                    
                                    # Check for horizontal symmetry
                                    is_symmetric = True
                                    for br in range(height):
                                        for bc in range(width // 2):
                                            if block_region[br, bc] != block_region[br, width-1-bc]:
                                                is_symmetric = False
                                                break
                                        if not is_symmetric:
                                            break
                                    
                                    # Check for filled top or bottom row
                                    top_filled = all(block_region[0, bc] == block_region[0, 0] and block_region[0, 0] != 0 for bc in range(width))
                                    bottom_filled = all(block_region[height-1, bc] == block_region[height-1, 0] and block_region[height-1, 0] != 0 for bc in range(width))
                                    
                                    # Check if it contains the object color
                                    has_object_color = object_color in block_region
                                    
                                    if is_symmetric and (top_filled or bottom_filled) and has_object_color:
                                        symmetrical_block = block_region
                                        symmetrical_block_position = (r, c, height, width)
                                        break
                                
                            if symmetrical_block is not None:
                                break
                    if symmetrical_block is not None:
                        break
                if symmetrical_block is not None:
                    break
        
        # If we found a symmetrical block, place it in the correct pattern
        if symmetrical_block is not None:
            r, c, height, width = symmetrical_block_position
            
            # Determine the pattern based on the block position
            # First figure out the starting coordinates (1,1) or (1,2)
            # and how many column groups (2 or 3)
            
            # Calculate potential pattern spacings
            for start_col in [1, 2]:
                # Try to find a matching pattern
                col_groups = 3 if grid.shape[1] >= 25 else 2
                objects_per_group = 3
                
                # Find all expected block positions for this pattern
                block_positions = []
                for group_idx in range(col_groups):
                    col_pos = start_col + group_idx * (width + 1)
                    
                    for obj_idx in range(objects_per_group):
                        row_pos = 1 + obj_idx * (height + 1)
                        
                        # Check bounds
                        if (row_pos + height <= grid.shape[0] and 
                            col_pos + width <= grid.shape[1]):
                            block_positions.append((row_pos, col_pos))
                
                # Check if our found symmetrical block position is close to this pattern
                for row_pos, col_pos in block_positions:
                    if abs(row_pos - r) <= 1 and abs(col_pos - c) <= 1:
                        # Found a matching pattern, place blocks at all positions
                        for pos_r, pos_c in block_positions:
                            if (pos_r + height <= grid.shape[0] and 
                                pos_c + width <= grid.shape[1]):
                                result[pos_r:pos_r+height, pos_c:pos_c+width] = symmetrical_block
                        
                        return result
            
            # If we couldn't match to a standard pattern, try to infer it from grid spacing
            # Look for regular spacing in the grid
            row_densities = [np.count_nonzero(grid[r, :]) for r in range(grid.shape[0])]
            col_densities = [np.count_nonzero(grid[:, c]) for c in range(grid.shape[1])]
            
            # Find rows and columns with blocks
            block_rows = [i for i, d in enumerate(row_densities) if d > np.mean(row_densities)]
            block_cols = [i for i, d in enumerate(col_densities) if d > np.mean(col_densities)]
            
            # Group consecutive rows and columns
            row_groups = []
            current_group = []
            for i in sorted(block_rows):
                if not current_group or i == current_group[-1] + 1:
                    current_group.append(i)
                else:
                    if len(current_group) >= height - 1:
                        row_groups.append(current_group)
                    current_group = [i]
            if len(current_group) >= height - 1:
                row_groups.append(current_group)
            
            col_groups = []
            current_group = []
            for i in sorted(block_cols):
                if not current_group or i == current_group[-1] + 1:
                    current_group.append(i)
                else:
                    if len(current_group) >= width - 1:
                        col_groups.append(current_group)
                    current_group = [i]
            if len(current_group) >= width - 1:
                col_groups.append(current_group)
            
            # For each row and column group, check if our symmetrical block is there
            # and use the pattern from the grid
            for row_group in row_groups:
                r_start = min(row_group)
                r_end = max(row_group) + 1
                
                for col_group in col_groups:
                    c_start = min(col_group)
                    c_end = max(col_group) + 1
                    
                    # Check if this matches our symmetrical block position
                    if (r_start <= r < r_end and c_start <= c < c_end):
                        # Place the symmetrical block in all regions
                        for rg in row_groups:
                            rs = min(rg)
                            re = max(rg) + 1
                            
                            for cg in col_groups:
                                cs = min(cg)
                                ce = max(cg) + 1
                                
                                # If this region is roughly the same size as our block
                                if abs(re - rs - height) <= 1 and abs(ce - cs - width) <= 1:
                                    # Place the block
                                    if (rs + height <= grid.shape[0] and cs + width <= grid.shape[1]):
                                        result[rs:rs+height, cs:cs+width] = symmetrical_block
                        
                        return result
        
        # If we still haven't found a valid symmetrical block, create a default pattern
        # First try standard positions
        for start_col in [1, 2]:
            if grid[1, start_col] != 0:
                # Try to find a block here
                width = 0
                for c in range(start_col, min(start_col + 6, grid.shape[1])):
                    if grid[1, c] != 0:
                        width += 1
                    else:
                        break
                
                height = 0
                for r in range(1, min(1 + 6, grid.shape[0])):
                    if grid[r, start_col] != 0:
                        height += 1
                    else:
                        break
                
                if 3 <= width <= 5 and 3 <= height <= 6:
                    # Extract the block
                    block = grid[1:1+height, start_col:start_col+width].copy()
                    
                    # Create a symmetrical version
                    for r in range(height):
                        for c in range(width // 2):
                            block[r, width-1-c] = block[r, c]
                    
                    # Ensure top or bottom row is filled
                    if np.any(block[0, :] != 0):
                        color = max(set(block[0, :]) - {0}, key=lambda x: np.count_nonzero(block[0, :] == x))
                        block[0, :] = color
                    elif np.any(block[height-1, :] != 0):
                        color = max(set(block[height-1, :]) - {0}, key=lambda x: np.count_nonzero(block[height-1, :] == x))
                        block[height-1, :] = color
                    
                    # Place in standard pattern
                    col_groups = 3 if grid.shape[1] >= 25 else 2
                    objects_per_group = 3
                    
                    for group_idx in range(col_groups):
                        col_pos = start_col + group_idx * (width + 1)
                        
                        for obj_idx in range(objects_per_group):
                            row_pos = 1 + obj_idx * (height + 1)
                            
                            # Check bounds
                            if (row_pos + height <= grid.shape[0] and 
                                col_pos + width <= grid.shape[1]):
                                result[row_pos:row_pos+height, col_pos:col_pos+width] = block
                    
                    return result
        
        # Last resort - create a basic symmetrical block pattern
        width = 3
        height = 4
        
        basic_block = np.zeros((height, width), dtype=int)
        basic_block[0, :] = object_color  # Top row filled with object color
        basic_block[1:, 0] = object_color  # Left column
        basic_block[1:, width-1] = object_color  # Right column
        basic_block[1:, 1:width-1] = (object_color % 8) + 1  # Middle
        
        # Place in standard pattern
        col_groups = 3 if grid.shape[1] >= 25 else 2
        objects_per_group = 3
        
        for group_idx in range(col_groups):
            col_pos = 1 + group_idx * (width + 1)
            
            for obj_idx in range(objects_per_group):
                row_pos = 1 + obj_idx * (height + 1)
                
                # Check bounds
                if (row_pos + height <= grid.shape[0] and 
                    col_pos + width <= grid.shape[1]):
                    result[row_pos:row_pos+height, col_pos:col_pos+width] = basic_block
        
        return result

