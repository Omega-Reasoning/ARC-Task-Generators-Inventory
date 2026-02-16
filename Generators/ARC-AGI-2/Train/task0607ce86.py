from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, retry, Contiguity
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects

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
        
        train_data = []
        
        # 2 train examples: 6 objects, top strip, vertical interior
        for _ in range(2):
            gridvars = {
                'num_objects': 6,
                'strip_position': 'top',
                'strip_arrangement': 'vertical'
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # 1 train example: 9 objects, bottom strip, horizontal interior
        gridvars = {
            'num_objects': 9,
            'strip_position': 'bottom',
            'strip_arrangement': 'horizontal'
        }
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_data.append({'input': input_grid, 'output': output_grid})
        
        # 1 test example with random parameters
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

        # Store for transform, so it knows how many blocks to reproduce
        self._num_objects = num_objects
        self._column_groups = column_groups
        self._objects_per_group = objects_per_group
        
        # Choose colors for the blocks
        object_color = taskvars['object']
        available_colors = [c for c in range(1, 10) if c != object_color]
        color_a = random.choice(available_colors)
        available_colors.remove(color_a)
        color_b = random.choice(available_colors)
        
        colors_used = [object_color, color_a, color_b]
        
        # Define block dimensions
        block_width = random.choice([3, 4])
        block_height = random.choice([4, 5])
        
        # Create a template for a single block
        block = np.zeros((block_height, block_width), dtype=int)
        
        # Fill the block according to specifications
        if strip_position == 'top':
            # Top row with object_color
            block[0, :] = object_color
            
            if strip_arrangement == 'vertical':
                # Vertical stripes beneath the top row
                for i in range(1, block_height):
                    block[i, 0] = color_a
                    block[i, block_width-1] = color_a
                    block[i, 1:block_width-1] = color_b
            else:  # horizontal
                stripe_height = (block_height - 1) // 4
                remainder = (block_height - 1) % 4
                
                current_row = 1
                block[current_row:current_row+stripe_height, :] = color_a
                current_row += stripe_height
                
                middle_height = 2 * stripe_height + remainder
                block[current_row:current_row+middle_height, :] = color_b
                current_row += middle_height
                
                block[current_row:, :] = color_a
                
        else:  # bottom
            # Bottom row with object_color
            block[block_height-1, :] = object_color
            
            if strip_arrangement == 'vertical':
                for i in range(block_height-1):
                    block[i, 0] = color_a
                    block[i, block_width-1] = color_a
                    block[i, 1:block_width-1] = color_b
            else:  # horizontal
                stripe_height = (block_height - 1) // 4
                remainder = (block_height - 1) % 4
                
                block[0:stripe_height, :] = color_a
                middle_height = 2 * stripe_height + remainder
                block[stripe_height:stripe_height+middle_height, :] = color_b
                block[stripe_height+middle_height:block_height-1, :] = color_a
        
        # Ensure horizontal symmetry
        for r in range(block_height):
            for c in range(block_width // 2):
                block[r, block_width-1-c] = block[r, c]
        
        # Calculate grid dimensions
        total_cols = column_groups * block_width + (column_groups - 1)
        total_rows = objects_per_group * block_height + (objects_per_group - 1)
        
        # Starting position
        start_row = 1
        start_col = random.choice([1, 2])
        
        grid_width = max(total_cols + start_col + 1, taskvars['col'])
        grid_height = total_rows + start_row + 1
        
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Randomly select one block to keep symmetrical and intact
        intact_row_group = random.randint(0, objects_per_group - 1)
        intact_col_group = random.randint(0, column_groups - 1)
        intact_row_start = start_row + intact_row_group * (block_height + 1)
        intact_col_start = start_col + intact_col_group * (block_width + 1)
        
        self._intact_block_position = (intact_row_start, intact_col_start, block_height, block_width)
        
        block_positions = []
        
        # Place blocks
        for group_idx in range(column_groups):
            col_start = start_col + group_idx * (block_width + 1)
            
            for obj_idx in range(objects_per_group):
                row_start = start_row + obj_idx * (block_height + 1)
                
                if (row_start + block_height <= grid_height and 
                    col_start + block_width <= grid_width):
                    
                    block_copy = block.copy()
                    
                    # Break symmetry for all but the intact block
                    if row_start != intact_row_start or col_start != intact_col_start:
                        num_modifications = random.randint(1, 2)
                        
                        for _ in range(num_modifications):
                            mod_row = random.randint(1, block_height - 2)  # avoid top/bottom row
                            mod_col = random.randint(0, (block_width // 2) - 1)
                            
                            current_color = block_copy[mod_row, mod_col]
                            new_color = random.choice([c for c in colors_used if c != current_color])
                            block_copy[mod_row, mod_col] = new_color
                    
                    grid[row_start:row_start+block_height, col_start:col_start+block_width] = block_copy
                    block_positions.append((row_start, col_start, block_height, block_width))
        
        intact_block_coords = (intact_row_start, intact_col_start, block_height, block_width)
        
        # Add scattered single-colored cells
        num_scattered = random.randint(grid_width * grid_height // 15, grid_width * grid_height // 8)
        
        for _ in range(num_scattered):
            r = random.randint(0, grid_height-1)
            c = random.randint(0, grid_width-1)
            
            row_start, col_start, block_h, block_w = intact_block_coords
            if (row_start-1 <= r <= row_start + block_h and 
                col_start-1 <= c <= col_start + block_w):
                continue
            
            if random.random() < 0.7:
                color = object_color
            else:
                color = random.choice([color_a, color_b])
                
            grid[r, c] = color
        
        # Ensure the grid has exactly taskvars['col'] columns
        if grid.shape[1] > taskvars['col']:
            grid = grid[:, :taskvars['col']]
        elif grid.shape[1] < taskvars['col']:
            padding = np.zeros((grid.shape[0], taskvars['col'] - grid.shape[1]), dtype=int)
            grid = np.hstack((grid, padding))
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """Transform the input grid by finding the intact symmetrical block and replicating it."""
        result = np.zeros_like(grid)
        
        object_color = taskvars['object']

        # Use stored num_objects to control how many blocks to output
        num_objects = getattr(self, "_num_objects", None)
        if num_objects == 6:
            col_groups_from_num = 2
            objects_per_group = 3
        elif num_objects == 9:
            col_groups_from_num = 3
            objects_per_group = 3
        else:
            # Fallback (should rarely happen)
            col_groups_from_num = 3 if grid.shape[1] >= 25 else 2
            objects_per_group = 3
        
        # -----------------------------
        # Step 1: find a symmetrical block
        # -----------------------------
        symmetrical_block = None
        symmetrical_block_position = None
        
        for r in range(1, grid.shape[0]-2):
            for c in range(1, grid.shape[1]-2):
                if grid[r, c] != 0:
                    width = 0
                    for col in range(c, min(c+6, grid.shape[1])):
                        if grid[r, col] != 0:
                            width += 1
                        else:
                            break
                    
                    height = 0
                    for row in range(r, min(r+6, grid.shape[0])):
                        if grid[row, c] != 0:
                            height += 1
                        else:
                            break
                    
                    if 3 <= width <= 5 and 3 <= height <= 6:
                        if r + height <= grid.shape[0] and c + width <= grid.shape[1]:
                            block_region = grid[r:r+height, c:c+width].copy()
                            
                            # Check symmetry
                            is_symmetric = True
                            for br in range(height):
                                for bc in range(width // 2):
                                    if block_region[br, bc] != block_region[br, width-1-bc]:
                                        is_symmetric = False
                                        break
                                if not is_symmetric:
                                    break
                            
                            top_filled = all(block_region[0, bc] == block_region[0, 0] and block_region[0, 0] != 0 for bc in range(width))
                            bottom_filled = all(block_region[height-1, bc] == block_region[height-1, 0] and block_region[height-1, 0] != 0 for bc in range(width))
                            has_object_color = object_color in block_region
                            
                            # Check isolation
                            is_isolated = True
                            for check_r in range(max(0, r-1), min(grid.shape[0], r+height+1)):
                                for check_c in range(max(0, c-1), min(grid.shape[1], c+width+1)):
                                    if r <= check_r < r+height and c <= check_c < c+width:
                                        continue
                                    if grid[check_r, check_c] != 0:
                                        is_isolated = False
                                        break
                                if not is_isolated:
                                    break
                            
                            if is_symmetric and (top_filled or bottom_filled) and has_object_color and is_isolated:
                                symmetrical_block = block_region
                                symmetrical_block_position = (r, c, height, width)
                                break
                if symmetrical_block is not None:
                    break
            if symmetrical_block is not None:
                break
        
        # Broader search if not found
        if symmetrical_block is None:
            for r in range(1, grid.shape[0]-2):
                for c in range(1, grid.shape[1]-2):
                    if grid[r, c] != 0:
                        for height in [4, 5]:
                            for width in [3, 4]:
                                if (r + height <= grid.shape[0] and 
                                    c + width <= grid.shape[1] and
                                    np.any(grid[r:r+height, c:c+width] != 0)):
                                    
                                    block_region = grid[r:r+height, c:c+width].copy()
                                    
                                    is_symmetric = True
                                    for br in range(height):
                                        for bc in range(width // 2):
                                            if block_region[br, bc] != block_region[br, width-1-bc]:
                                                is_symmetric = False
                                                break
                                        if not is_symmetric:
                                            break
                                    
                                    top_filled = all(block_region[0, bc] == block_region[0, 0] and block_region[0, 0] != 0 for bc in range(width))
                                    bottom_filled = all(block_region[height-1, bc] == block_region[height-1, 0] and block_region[height-1, 0] != 0 for bc in range(width))
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
        
        # -----------------------------
        # Step 2: place symmetrical block according to pattern
        # -----------------------------
        if symmetrical_block is not None:
            r, c, height, width = symmetrical_block_position
            
            # Try patterns with start_col = 1 or 2 using known col_groups and objects_per_group
            for start_col in [1, 2]:
                col_groups = col_groups_from_num
                # objects_per_group already defined from num_objects
                
                block_positions = []
                for group_idx in range(col_groups):
                    col_pos = start_col + group_idx * (width + 1)
                    for obj_idx in range(objects_per_group):
                        row_pos = 1 + obj_idx * (height + 1)
                        if (row_pos + height <= grid.shape[0] and 
                            col_pos + width <= grid.shape[1]):
                            block_positions.append((row_pos, col_pos))
                
                # Check if the found symmetrical block aligns with one of these positions
                for row_pos, col_pos in block_positions:
                    if abs(row_pos - r) <= 1 and abs(col_pos - c) <= 1:
                        # Place block at all positions in this pattern
                        for pos_r, pos_c in block_positions:
                            if (pos_r + height <= grid.shape[0] and 
                                pos_c + width <= grid.shape[1]):
                                result[pos_r:pos_r+height, pos_c:pos_c+width] = symmetrical_block
                        return result
            
            # If we couldn't match to a standard pattern, infer spacing from density
            row_densities = [np.count_nonzero(grid[r_i, :]) for r_i in range(grid.shape[0])]
            col_densities = [np.count_nonzero(grid[:, c_i]) for c_i in range(grid.shape[1])]
            
            block_rows = [i for i, d in enumerate(row_densities) if d > np.mean(row_densities)]
            block_cols = [i for i, d in enumerate(col_densities) if d > np.mean(col_densities)]
            
            row_groups_list = []
            current_group = []
            for i in sorted(block_rows):
                if not current_group or i == current_group[-1] + 1:
                    current_group.append(i)
                else:
                    if len(current_group) >= height - 1:
                        row_groups_list.append(current_group)
                    current_group = [i]
            if len(current_group) >= height - 1:
                row_groups_list.append(current_group)
            
            col_groups_list = []
            current_group = []
            for i in sorted(block_cols):
                if not current_group or i == current_group[-1] + 1:
                    current_group.append(i)
                else:
                    if len(current_group) >= width - 1:
                        col_groups_list.append(current_group)
                    current_group = [i]
            if len(current_group) >= width - 1:
                col_groups_list.append(current_group)
            
            for row_group in row_groups_list:
                r_start = min(row_group)
                r_end = max(row_group) + 1
                
                for col_group in col_groups_list:
                    c_start = min(col_group)
                    c_end = max(col_group) + 1
                    
                    if (r_start <= r < r_end and c_start <= c < c_end):
                        # Place symmetrical block in all similar-sized regions
                        for rg in row_groups_list:
                            rs = min(rg)
                            re = max(rg) + 1
                            
                            for cg in col_groups_list:
                                cs = min(cg)
                                ce = max(cg) + 1
                                
                                if abs(re - rs - height) <= 1 and abs(ce - cs - width) <= 1:
                                    if (rs + height <= grid.shape[0] and cs + width <= grid.shape[1]):
                                        result[rs:rs+height, cs:cs+width] = symmetrical_block
                        return result
        
        # -----------------------------
        # Step 3: fallback – reconstruct a symmetrical block and place it
        # -----------------------------
        for start_col in [1, 2]:
            if start_col < grid.shape[1] and grid[1, start_col] != 0:
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
                    block = grid[1:1+height, start_col:start_col+width].copy()
                    
                    # Make block symmetric
                    for r in range(height):
                        for c in range(width // 2):
                            block[r, width-1-c] = block[r, c]
                    
                    # Ensure top or bottom row is filled
                    if np.any(block[0, :] != 0):
                        colors = [val for val in block[0, :] if val != 0]
                        color = max(set(colors), key=lambda x: colors.count(x)) if colors else object_color
                        block[0, :] = color
                    elif np.any(block[height-1, :] != 0):
                        colors = [val for val in block[height-1, :] if val != 0]
                        color = max(set(colors), key=lambda x: colors.count(x)) if colors else object_color
                        block[height-1, :] = color
                    
                    col_groups = col_groups_from_num
                    # objects_per_group already set from num_objects
                    
                    for group_idx in range(col_groups):
                        col_pos = start_col + group_idx * (width + 1)
                        for obj_idx in range(objects_per_group):
                            row_pos = 1 + obj_idx * (height + 1)
                            if (row_pos + height <= grid.shape[0] and 
                                col_pos + width <= grid.shape[1]):
                                result[row_pos:row_pos+height, col_pos:col_pos+width] = block
                    return result
        
        # -----------------------------
        # Step 4: last resort – basic block pattern with correct count
        # -----------------------------
        width = 3
        height = 4
        
        basic_block = np.zeros((height, width), dtype=int)
        basic_block[0, :] = object_color
        basic_block[1:, 0] = object_color
        basic_block[1:, width-1] = object_color
        basic_block[1:, 1:width-1] = (object_color % 8) + 1
        
        col_groups = col_groups_from_num
        # objects_per_group already set
        
        for group_idx in range(col_groups):
            col_pos = 1 + group_idx * (width + 1)
            for obj_idx in range(objects_per_group):
                row_pos = 1 + obj_idx * (height + 1)
                if (row_pos + height <= grid.shape[0] and 
                    col_pos + width <= grid.shape[1]):
                    result[row_pos:row_pos+height, col_pos:col_pos+width] = basic_block
        
        return result
