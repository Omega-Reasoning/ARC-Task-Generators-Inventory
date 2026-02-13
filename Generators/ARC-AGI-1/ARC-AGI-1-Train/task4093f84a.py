from arc_task_generator import ARCTaskGenerator
import numpy as np
import random
from transformation_library import find_connected_objects

class Task4093f84aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain one {color('object_color')} rectangular block and multiple same-colored cells.",
            "The {color('object_color')} rectangular block is positioned around the center of the grid and sized such that it spans either several rows or columns completely.",
            "The same-colored cells are placed throughout the grid, with some on both sides of the block."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and determining whether the {color('object_color')} rectangular block spans several complete rows or columns.",
            "If the block spans rows, move all colored cells vertically until they touch the block.",
            "If the block spans columns, move all colored cells horizontally until they touch the block.",
            "If multiple colored cells are in the same row or column on the same side of the block, move them all to be adjacent to the block.",
            "Finally, change the color of all same-colored cells not part of the rectangular block to {color('object_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        grid_size = random.randint(8, 15)
        object_color = random.randint(1, 9)
        
        taskvars = {
            'grid_size': grid_size,
            'object_color': object_color
        }
        
        num_train_examples = random.randint(3, 4)
        train_data = []
        
        # Ensure we have both row-spanning and column-spanning examples
        orientations = ['row'] * (num_train_examples // 2) + ['column'] * ((num_train_examples + 1) // 2)
        random.shuffle(orientations)
        
        for i in range(num_train_examples):
            # Generate a different color for colored cells
            same_color = random.randint(1, 9)
            while same_color == object_color:
                same_color = random.randint(1, 9)
            
            gridvars = {
                'same_color': same_color,
                'orientation': orientations[i]
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_same_color = random.randint(1, 9)
        while test_same_color == object_color:
            test_same_color = random.randint(1, 9)
        
        test_orientation = random.choice(['row', 'column'])
        
        test_gridvars = {
            'same_color': test_same_color,
            'orientation': test_orientation
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_data = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_data,
            'test': test_data
        }
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        same_color = gridvars['same_color']
        orientation = gridvars['orientation']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create a rectangular block in the middle
        center = grid_size // 2
        span_size = random.randint(2, min(4, grid_size - 6))
        
        if orientation == 'row':
            # Block spans rows
            row_start = max(1, center - span_size // 2)
            row_end = min(grid_size - 1, row_start + span_size)
            
            # Make the block span the entire width
            for row in range(row_start, row_end):
                grid[row, :] = object_color
                
            # Add cells, ensuring some are on each side of the block
            num_above = random.randint(2, 4)
            num_below = random.randint(2, 4)
            
            # Map to track cells per column on each side
            above_cols = {}
            below_cols = {}
            
            # Place cells above the block
            for _ in range(num_above):
                attempts = 0
                while attempts < 50:
                    row = random.randint(0, row_start - 1)
                    col = random.randint(0, grid_size - 1)
                    
                    if grid[row, col] == 0:
                        if col not in above_cols:
                            above_cols[col] = []
                        
                        # Ensure no more than 2 cells in the same column on this side
                        if len(above_cols[col]) < 2:
                            grid[row, col] = same_color
                            above_cols[col].append(row)
                            break
                    
                    attempts += 1
            
            # Place cells below the block
            for _ in range(num_below):
                attempts = 0
                while attempts < 50:
                    row = random.randint(row_end, grid_size - 1)
                    col = random.randint(0, grid_size - 1)
                    
                    if grid[row, col] == 0:
                        if col not in below_cols:
                            below_cols[col] = []
                        
                        # Ensure no more than 2 cells in the same column on this side
                        if len(below_cols[col]) < 2:
                            grid[row, col] = same_color
                            below_cols[col].append(row)
                            break
                    
                    attempts += 1
                    
        else:  # Block spans columns
            # Block spans columns
            col_start = max(1, center - span_size // 2)
            col_end = min(grid_size - 1, col_start + span_size)
            
            # Make the block span the entire height
            for col in range(col_start, col_end):
                grid[:, col] = object_color
                
            # Add cells, ensuring some are on each side of the block
            num_left = random.randint(2, 4)
            num_right = random.randint(2, 4)
            
            # Maps to track cells per row on each side
            left_rows = {}
            right_rows = {}
            
            # Place cells to the left of the block
            for _ in range(num_left):
                attempts = 0
                while attempts < 50:
                    row = random.randint(0, grid_size - 1)
                    col = random.randint(0, col_start - 1)
                    
                    if grid[row, col] == 0:
                        if row not in left_rows:
                            left_rows[row] = []
                        
                        # Ensure no more than 2 cells in the same row on this side
                        if len(left_rows[row]) < 2:
                            grid[row, col] = same_color
                            left_rows[row].append(col)
                            break
                    
                    attempts += 1
            
            # Place cells to the right of the block
            for _ in range(num_right):
                attempts = 0
                while attempts < 50:
                    row = random.randint(0, grid_size - 1)
                    col = random.randint(col_end, grid_size - 1)
                    
                    if grid[row, col] == 0:
                        if row not in right_rows:
                            right_rows[row] = []
                        
                        # Ensure no more than 2 cells in the same row on this side
                        if len(right_rows[row]) < 2:
                            grid[row, col] = same_color
                            right_rows[row].append(col)
                            break
                    
                    attempts += 1
        
        return grid
    
    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        output_grid = grid.copy()
        
        # Find the rectangular block and all cells
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        main_block = None
        
        for obj in objects:
            if object_color in obj.colors:
                main_block = obj
                break
        
        if main_block is None:
            return output_grid
        
        # Determine block orientation and boundaries
        box = main_block.bounding_box
        block_height = box[0].stop - box[0].start
        block_width = box[1].stop - box[1].start
        
        # Clear all non-block cells first
        for obj in objects:
            if object_color not in obj.colors:
                for r, c, color in obj.cells:
                    output_grid[r, c] = 0
        
        # Get all cells not of the main block color
        other_cells = []
        for obj in objects:
            if object_color not in obj.colors:
                for r, c, color in obj.cells:
                    other_cells.append((r, c, color))
        
        # Group cells by row/column for handling multiple cells
        if block_width > block_height:  # Block spans rows
            # Group cells by column
            above_cells = {}  # column -> list of rows
            below_cells = {}  # column -> list of rows
            
            for r, c, color in other_cells:
                if r < box[0].start:  # Above the block
                    if c not in above_cells:
                        above_cells[c] = []
                    above_cells[c].append((r, color))
                elif r >= box[0].stop:  # Below the block
                    if c not in below_cells:
                        below_cells[c] = []
                    below_cells[c].append((r, color))
            
            # Move cells and update their colors
            for col, rows in above_cells.items():
                # Place cell just above the block
                output_grid[box[0].start - 1, col] = object_color
                
                # If there are multiple cells in this column above the block,
                # also place a second cell above the first one
                if len(rows) > 1:
                    output_grid[box[0].start - 2, col] = object_color
            
            for col, rows in below_cells.items():
                # Place cell just below the block
                output_grid[box[0].stop, col] = object_color
                
                # If there are multiple cells in this column below the block,
                # also place a second cell below the first one
                if len(rows) > 1:
                    output_grid[box[0].stop + 1, col] = object_color
                    
        else:  # Block spans columns
            # Group cells by row
            left_cells = {}   # row -> list of columns
            right_cells = {}  # row -> list of columns
            
            for r, c, color in other_cells:
                if c < box[1].start:  # Left of the block
                    if r not in left_cells:
                        left_cells[r] = []
                    left_cells[r].append((c, color))
                elif c >= box[1].stop:  # Right of the block
                    if r not in right_cells:
                        right_cells[r] = []
                    right_cells[r].append((c, color))
            
            # Move cells and update their colors
            for row, cols in left_cells.items():
                # Place a cell just to the left of the block
                output_grid[row, box[1].start - 1] = object_color
                
                # If there are multiple cells in this row to the left of the block,
                # also place a second cell to the left of the first one
                if len(cols) > 1:
                    output_grid[row, box[1].start - 2] = object_color
            
            for row, cols in right_cells.items():
                # Place a cell just to the right of the block
                output_grid[row, box[1].stop] = object_color
                
                # If there are multiple cells in this row to the right of the block,
                # also place a second cell to the right of the first one
                if len(cols) > 1:
                    output_grid[row, box[1].stop + 1] = object_color
        
        return output_grid