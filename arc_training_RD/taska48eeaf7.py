from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject
from input_library import retry, create_object

class Taska48eeafGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "The grid contains one main block which is a 2x2 square in {color('object_color')} color (between 1-9) and this block can be placed anywhere in the grid.",
            "The main block 2x2 square is covered with two types of cells namely empty (0) cells and scattered cells (<5), where the scattered cells are of {color('cell_color')} color.",
            "These scattered cells are placed such that they are either horizontal or vertical or diagonal to the 2x2 main block square cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid main 2x2 square block.",
            "And the scattered cells around the main block, are attached like tails to the main block with respect to their position. Suppose if one of the scattered cells is horizontally placed across the main block, then it gets attached besides the main block horizontally. The same rule is applied for cells being vertical and diagonally placed.",
            "The attachment should not disturb the main block. The scattered cells must occur just as an extensions of the main block in the output grid",
            "Also, the scattered cell alone must be attached to the main block, it must not form like a series of attachments."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def color_name(self, color: int) -> str:
        color_map = {
            0: "black",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "cyan",
            9: "brown"
        }
        return color_map.get(color, f"color_{color}")
    
    def create_input(self, taskvars):
        # Get colors from taskvars
        main_block_color = taskvars['object_color']
        scattered_color = taskvars['cell_color']
        
        # Determine grid size
        rows = random.randint(10, 20)  # Larger minimum to allow for scattered cells
        cols = random.randint(10, 20)
        grid = np.zeros((rows, cols), dtype=int)
        
        # Place the 2x2 main block
        max_row = rows - 2  # Ensure space for the 2x2 block
        max_col = cols - 2
        
        if max_row < 0 or max_col < 0:
            # Grid too small, create a larger one
            return self.create_input(taskvars)
            
        top_row = random.randint(0, max_row)
        left_col = random.randint(0, max_col)
        
        # Create the 2x2 block
        grid[top_row:top_row+2, left_col:left_col+2] = main_block_color
        
        # Define corners of the 2x2 block
        corners = [
            (top_row, left_col),        # Top-left
            (top_row, left_col+1),      # Top-right
            (top_row+1, left_col),      # Bottom-left
            (top_row+1, left_col+1)     # Bottom-right
        ]
        
        # Randomly choose 1-4 scattered cells to place
        num_scattered = random.randint(1, 4)
        
        # Track successful placements
        successful_placements = 0
        max_attempts = 20
        
        # Try to place cells with different alignments
        alignment_types = ['horizontal', 'vertical', 'diagonal']
        random.shuffle(alignment_types)
        
        for alignment in alignment_types:
            if successful_placements >= num_scattered:
                break
                
            for attempt in range(max_attempts):
                if alignment == 'horizontal':
                    # Horizontal alignment (same row as one of the main block cells)
                    main_cell = random.choice(corners)
                    r = main_cell[0]
                    
                    # Choose a column that's not adjacent to the main block
                    if random.choice([True, False]) and left_col >= 3:  # Left side
                        c = random.randint(0, left_col-2)
                    elif left_col + 2 < cols - 2:  # Right side
                        c = random.randint(left_col+3, cols-1)
                    else:
                        continue  # No space, try another attempt
                
                elif alignment == 'vertical':
                    # Vertical alignment (same column as one of the main block cells)
                    main_cell = random.choice(corners)
                    c = main_cell[1]
                    
                    # Choose a row that's not adjacent to the main block
                    if random.choice([True, False]) and top_row >= 3:  # Above
                        r = random.randint(0, top_row-2)
                    elif top_row + 2 < rows - 2:  # Below
                        r = random.randint(top_row+3, rows-1)
                    else:
                        continue  # No space, try another attempt
                
                elif alignment == 'diagonal':
                    # For diagonal alignment, we need to draw a diagonal line from one of the corners
                    main_cell = random.choice(corners)
                    main_r, main_c = main_cell
                    
                    # Choose a direction: up-left, up-right, down-left, down-right
                    direction = random.choice(['up-left', 'up-right', 'down-left', 'down-right'])
                    
                    # Calculate offsets to maintain true diagonal alignment
                    offset = random.randint(2, 5)  # At least 2 cells away, but not too far
                    
                    if direction == 'up-left':
                        r = main_r - offset
                        c = main_c - offset
                    elif direction == 'up-right':
                        r = main_r - offset
                        c = main_c + offset
                    elif direction == 'down-left':
                        r = main_r + offset
                        c = main_c - offset
                    else:  # down-right
                        r = main_r + offset
                        c = main_c + offset
                    
                    # Check bounds
                    if not (0 <= r < rows and 0 <= c < cols):
                        continue  # Out of bounds, try another attempt
                
                # Check if the position is empty and not too close to existing scattered cells
                if (0 <= r < rows and 0 <= c < cols and 
                    grid[r, c] == 0 and 
                    all(abs(r-existing_r) > 1 or abs(c-existing_c) > 1 
                        for existing_r, existing_c in zip(*np.where(grid == scattered_color)))):
                    grid[r, c] = scattered_color
                    successful_placements += 1
                    break  # Successfully placed this cell
        
        # Make sure we actually placed at least one scattered cell
        if successful_placements == 0:
            # If we couldn't place any scattered cells, try again with a larger grid
            return self.create_input(taskvars)
        
        return grid
    
    def transform_input(self, input_grid):
        # Copy the input grid to start with
        output_grid = np.zeros_like(input_grid)
        
        # Find the main 2x2 block
        objects = find_connected_objects(input_grid, diagonal_connectivity=False)
        
        # Get the largest object (which should be the 2x2 block)
        main_block = max(objects.objects, key=lambda obj: len(obj))
        main_block_color = list(main_block.colors)[0]  # Get the color of the main block
        
        # Get scattered cells (all objects that aren't the main block)
        scattered_cells = [obj for obj in objects.objects if obj != main_block]
        
        # Create the output by placing main block first
        main_block.paste(output_grid)
        
        # Get the boundaries and cells of the main block
        main_block_cells = {(r, c) for r, c, _ in main_block.cells}
        
        # Find corners of the main block
        corner_positions = []
        for r, c, _ in main_block.cells:
            is_corner = sum(1 for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)] 
                           if (r+dr, c+dc) in main_block_cells) < 4
            if is_corner:
                corner_positions.append((r, c))
        
        # For each scattered cell, find a position adjacent to the main block
        for cell in scattered_cells:
            # Get the cell's position - take first position if multiple exist
            cell_positions = list(cell.cells)
            if not cell_positions:
                continue
            
            r, c, color = cell_positions[0]  # Take the first position
            
            # Find alignment and closest main block cell
            closest_main_cell = None
            min_distance = float('inf')
            alignment_type = None
            
            for main_r, main_c in corner_positions:
                # Check horizontal alignment
                if r == main_r:
                    dist = abs(c - main_c)
                    if dist < min_distance:
                        min_distance = dist
                        closest_main_cell = (main_r, main_c)
                        alignment_type = 'horizontal'
                
                # Check vertical alignment
                elif c == main_c:
                    dist = abs(r - main_r)
                    if dist < min_distance:
                        min_distance = dist
                        closest_main_cell = (main_r, main_c)
                        alignment_type = 'vertical'
                
                # Check diagonal alignment
                elif abs(r - main_r) == abs(c - main_c):
                    dist = abs(r - main_r)  # Diagonal distance
                    if dist < min_distance:
                        min_distance = dist
                        closest_main_cell = (main_r, main_c)
                        alignment_type = 'diagonal'
            
            # If no alignment found, skip this cell
            if not closest_main_cell:
                continue
            
            main_r, main_c = closest_main_cell
            
            # Determine attachment position based on alignment
            attachment_position = None
            
            if alignment_type == 'horizontal':
                if c < main_c:  # Cell is to the left
                    # Attach to the left of the main block
                    left_edge = min(c for r, c in main_block_cells if r == main_r)
                    attachment_position = (main_r, left_edge - 1, color)
                else:  # Cell is to the right
                    # Attach to the right of the main block
                    right_edge = max(c for r, c in main_block_cells if r == main_r)
                    attachment_position = (main_r, right_edge + 1, color)
            
            elif alignment_type == 'vertical':
                if r < main_r:  # Cell is above
                    # Attach above the main block
                    top_edge = min(r for r, c in main_block_cells if c == main_c)
                    attachment_position = (top_edge - 1, main_c, color)
                else:  # Cell is below
                    # Attach below the main block
                    bottom_edge = max(r for r, c in main_block_cells if c == main_c)
                    attachment_position = (bottom_edge + 1, main_c, color)
            
            elif alignment_type == 'diagonal':
                # Determine which diagonal direction
                row_diff = r - main_r
                col_diff = c - main_c
                
                # Determine the corner of the main block to attach to
                corner_r, corner_c = None, None
                
                if row_diff < 0 and col_diff < 0:  # Top-left
                    corner_r = min(r for r, c in main_block_cells)
                    corner_c = min(c for r, c in main_block_cells)
                    attachment_position = (corner_r - 1, corner_c - 1, color)
                elif row_diff < 0 and col_diff > 0:  # Top-right
                    corner_r = min(r for r, c in main_block_cells)
                    corner_c = max(c for r, c in main_block_cells)
                    attachment_position = (corner_r - 1, corner_c + 1, color)
                elif row_diff > 0 and col_diff < 0:  # Bottom-left
                    corner_r = max(r for r, c in main_block_cells)
                    corner_c = min(c for r, c in main_block_cells)
                    attachment_position = (corner_r + 1, corner_c - 1, color)
                elif row_diff > 0 and col_diff > 0:  # Bottom-right
                    corner_r = max(r for r, c in main_block_cells)
                    corner_c = max(c for r, c in main_block_cells)
                    attachment_position = (corner_r + 1, corner_c + 1, color)
            
            # Add the attachment to the output grid if valid
            if attachment_position:
                attach_r, attach_c, attach_color = attachment_position
                if (0 <= attach_r < output_grid.shape[0] and 
                    0 <= attach_c < output_grid.shape[1] and 
                    (attach_r, attach_c) not in main_block_cells):
                    output_grid[attach_r, attach_c] = attach_color
        
        return output_grid

    def create_grids(self):
        num_train_pairs = random.randint(3, 5)
        train_pairs = []

        # Select colors for the task (between 1-9)
        object_color = random.randint(1, 9)
        cell_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        taskvars = {
            'object_color': object_color,
            'cell_color': cell_color
        }
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Make taskvars available to transform_input
        self.taskvars = taskvars

        # Replace {color('object_color')} etc. in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('object_color')}", color_fmt('object_color'))
                 .replace("{color('cell_color')}", color_fmt('cell_color'))
            for chain in self.input_reasoning_chain
        ]
        
        for _ in range(num_train_pairs):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)] 
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)