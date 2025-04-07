from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects

class Task673ef223Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain two {color('object_color')} vertical lines and several {color('cell_color')} cells, with the remaining cells being empty (0).",
            "The two {color('object_color')} vertical lines have the exact same length and are always placed in the first and last columns, with one line appearing in the top half and the other in the bottom half of the grid.",
            "The number of {color('cell_color')} cells is less than the length of the {color('object_color')} line, with exactly one {color('cell_color')} cell placed in a single row, and the row must be one of those occupied by the top {color('object_color')} line.",
            "Each {color('cell_color')} cell can be positioned anywhere within the row but must never touch the {color('object_color')} line."
        ]
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and changing each {color('cell_color')} cell to {color('fill_color')}.",
            "Once the color has been changed, fill specific empty (0) cells in the rows containing a {color('fill_color')} cell with {color('cell_color')} color, starting immediately after each {color('fill_color')} cell and continuing until reaching the {color('object_color')} cell in the same row.",
            "The {color('object_color')} cells remain unchanged, and only the empty (0) cells between the {color('fill_color')} cell and the {color('object_color')} cell in the same row are filled with {color('cell_color')} color.",
            "Then, all empty (0) cells in the rows of the lower {color('object_color')} line that correspond to the rows containing the {color('cell_color')} cells in the upper part of the grid are filled with {color('cell_color')}."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # 1. Initialize task variables
        taskvars = {}
        
        # Choose distinct colors
        colors = random.sample(range(1, 10), 3)
        taskvars['object_color'] = colors[0]
        taskvars['cell_color'] = colors[1]
        taskvars['fill_color'] = colors[2]
        
        # 2. Create train and test grids
        num_train_examples = random.randint(3, 6)
        train_test_data = {
            'train': [],
            'test': []
        }
        
        # Ensure we have diversity in line placement
        first_in_top = {'first_line_top': True}
        first_in_bottom = {'first_line_top': False}
        
        # Create at least one example with first line in top half and one with first line in bottom half
        input_grid = self.create_input(taskvars, first_in_top)
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['train'].append({
            'input': input_grid,
            'output': output_grid
        })
        
        input_grid = self.create_input(taskvars, first_in_bottom)
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['train'].append({
            'input': input_grid,
            'output': output_grid
        })
        
        # Create remaining train examples with random line placement
        for _ in range(num_train_examples - 2):
            gridvars = {'first_line_top': random.choice([True, False])}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_test_data['train'].append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with random line placement
        gridvars = {'first_line_top': random.choice([True, False])}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_test_data['test'].append({
            'input': input_grid,
            'output': output_grid
        })
        
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        # 1. Determine grid size
        num_rows = random.randint(10, 24)  # Ensuring enough rows for two lines with half separation
        num_cols = random.randint(5, 15)
        
        # Ensure we have at least 3 columns (needed for cell placement)
        num_cols = max(3, num_cols)
        
        # 2. Initialize empty grid
        grid = np.zeros((num_rows, num_cols), dtype=int)
        
        # 3. Determine which line goes where based on gridvars
        first_line_top = gridvars.get('first_line_top', random.choice([True, False]))
        
        if first_line_top:
            # First column: top half, Last column: bottom half
            top_line_col = 0
            bottom_line_col = num_cols - 1
        else:
            # First column: bottom half, Last column: top half
            top_line_col = num_cols - 1
            bottom_line_col = 0
        
        # 4. Calculate line placement to ensure separation
        # Calculate initial line length (will adjust if needed)
        max_line_length = num_rows // 3  # At most 1/3 of rows to ensure space
        line_length = random.randint(3, max_line_length)
        
        # Divide the grid into quarters for placement
        quarter = num_rows // 4
        
        # Top line must be in top half
        top_start = random.randint(0, quarter * 2 - line_length)
        top_end = top_start + line_length
        
        # Bottom line must be in bottom half with at least 2 rows separation
        # Ensure at least 2 empty rows between top_end and bottom_start
        min_bottom_start = top_end + 2
        
        # Check if we need to adjust line length to ensure separation
        available_space = num_rows - min_bottom_start
        
        if available_space < line_length:
            # Reduce line length to fit
            line_length = max(2, available_space)
            
            # Recalculate top line to maintain balance
            max_top_end = num_rows - line_length - 2
            top_start = random.randint(0, max_top_end - line_length)
            top_end = top_start + line_length
            
            min_bottom_start = top_end + 2
        
        # Place bottom line
        bottom_start = random.randint(min_bottom_start, num_rows - line_length)
        bottom_end = bottom_start + line_length
        
        # Verify we have at least 2 empty rows
        assert bottom_start - top_end >= 2, "Insufficient separation between lines"
        
        # 5. Place top vertical line
        for i in range(top_start, top_end):
            grid[i, top_line_col] = taskvars['object_color']
        
        # 6. Place bottom vertical line
        for i in range(bottom_start, bottom_end):
            grid[i, bottom_line_col] = taskvars['object_color']
        
        # 7. Place cell_color cells in rows occupied by top vertical line
        top_rows = list(range(top_start, top_end))
        
        # Decide how many cells to place (less than line_length)
        num_cells = random.randint(1, min(3, line_length - 1))
        cell_rows = random.sample(top_rows, num_cells)
        
        for row in cell_rows:
            # Define valid columns where cells can be placed (not touching object_color lines)
            valid_cols = []
            for col in range(num_cols):
                # Check if this column is not part of any vertical line
                if col != top_line_col and col != bottom_line_col:
                    # Also ensure it's not adjacent to any vertical line
                    if (col > 0 and grid[row, col-1] == taskvars['object_color']) or \
                       (col < num_cols-1 and grid[row, col+1] == taskvars['object_color']):
                        continue
                    valid_cols.append(col)
            
            if valid_cols:  # Only place a cell if there are valid positions
                col = random.choice(valid_cols)
                grid[row, col] = taskvars['cell_color']
        
        # Verify that we have cells placed
        has_cells = False
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == taskvars['cell_color']:
                    has_cells = True
                    break
            if has_cells:
                break
        
        # If no cells were placed, place at least one
        if not has_cells and top_rows:
            row = random.choice(top_rows)
            valid_cols = [c for c in range(1, num_cols-1) if c != top_line_col and c != bottom_line_col]
            if valid_cols:
                col = random.choice(valid_cols)
                grid[row, col] = taskvars['cell_color']
        
        return grid

    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid
        output = np.copy(grid)
        
        # Find all the cell_color cells and convert them to fill_color
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == taskvars['cell_color']:
                    # Step 1: Change cell_color to fill_color
                    output[r, c] = taskvars['fill_color']
                    
                    # Step 2: Fill empty cells from the fill_color cell until the object_color cell
                    # Find the object_color cell in the same row
                    found_object = False
                    
                    # Check if there's an object_color to the right
                    for col in range(c + 1, grid.shape[1]):
                        if grid[r, col] == taskvars['object_color']:
                            # Fill empty cells between fill_color and object_color
                            for fill_col in range(c + 1, col):
                                if grid[r, fill_col] == 0:  # Only fill empty cells
                                    output[r, fill_col] = taskvars['cell_color']
                            found_object = True
                            break
                    
                    # If no object_color found to the right, check to the left
                    if not found_object:
                        for col in range(c - 1, -1, -1):
                            if grid[r, col] == taskvars['object_color']:
                                # Fill empty cells between object_color and fill_color
                                for fill_col in range(col + 1, c):
                                    if grid[r, fill_col] == 0:  # Only fill empty cells
                                        output[r, fill_col] = taskvars['cell_color']
                                break
        
        # Track rows that had a cell_color (now fill_color) cell
        cell_rows = set()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == taskvars['cell_color']:
                    cell_rows.add(r)
        
        # Find the top and bottom vertical lines
        first_col_object = False
        last_col_object = False
        
        # Check if first column has object_color
        for r in range(grid.shape[0]):
            if grid[r, 0] == taskvars['object_color']:
                first_col_object = True
                break
        
        # Check if last column has object_color
        for r in range(grid.shape[0]):
            if grid[r, grid.shape[1]-1] == taskvars['object_color']:
                last_col_object = True
                break
        
        # Determine which column has the top line and which has the bottom line
        top_line_rows = set()
        bottom_line_rows = set()
        top_line_col = -1
        bottom_line_col = -1
        
        if first_col_object and last_col_object:
            # Find all object_color rows in first column
            first_col_rows = [r for r in range(grid.shape[0]) if grid[r, 0] == taskvars['object_color']]
            # Find all object_color rows in last column
            last_col_rows = [r for r in range(grid.shape[0]) if grid[r, grid.shape[1]-1] == taskvars['object_color']]
            
            # Calculate average row positions to determine which is top/bottom
            first_col_avg = sum(first_col_rows) / len(first_col_rows) if first_col_rows else 0
            last_col_avg = sum(last_col_rows) / len(last_col_rows) if last_col_rows else 0
            
            if first_col_avg < last_col_avg:
                # First column is top, last column is bottom
                top_line_rows = set(first_col_rows)
                bottom_line_rows = set(last_col_rows)
                top_line_col = 0
                bottom_line_col = grid.shape[1] - 1
            else:
                # Last column is top, first column is bottom
                top_line_rows = set(last_col_rows)
                bottom_line_rows = set(first_col_rows)
                top_line_col = grid.shape[1] - 1
                bottom_line_col = 0
        
        # Find the mapping between top line rows and bottom line rows
        if top_line_rows and bottom_line_rows:
            top_min = min(top_line_rows)
            top_max = max(top_line_rows)
            bottom_min = min(bottom_line_rows)
            bottom_max = max(bottom_line_rows)
            
            # For each cell row in the top part
            for cell_row in cell_rows:
                if cell_row in top_line_rows:
                    # Calculate the corresponding row in the bottom part
                    if top_max > top_min:  # Prevent division by zero
                        relative_position = (cell_row - top_min) / (top_max - top_min)
                        corresponding_row = int(bottom_min + relative_position * (bottom_max - bottom_min))
                        
                        # Fill all empty cells in the corresponding row
                        for col in range(grid.shape[1]):
                            if grid[corresponding_row, col] == 0:
                                output[corresponding_row, col] = taskvars['cell_color']
                                
        return output
