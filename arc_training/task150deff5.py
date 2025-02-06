import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import find_connected_objects
import itertools

class ARCTask150deff5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['columns']}",
            "There is only one 8-way connected object in the input grid.",
            "The object is made up of the same building block.",
            "The building block consists of a 2x2 square and either a 1x3 or 3x1 rectangle, the square and the rectangle share a common edge or cell between them.",
            "The 8-way connected object is formed by placing 2 or 3 of the building blocks so that they are connected.",
            "The entire 8-way connected object has a color {color('input_color')}.",
            "The remaining cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "First identify the 8-way connected object.",
            "Then identify the building block of the 8-way connected object which is a 2x2 square and either a 1x3 or 3x1 rectangle, the square and the rectangle share a common edge or cell between them.",
            "For all the building blocks, color the 2x2 square with {color('color_1')} and the rectangle with color {color('color_2')}.",
            "The remaining cells remain empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['columns']
        input_color = taskvars['input_color']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        def create_building_block():
            # Create base 2x2 square
            square_pos = (0, 0)  # Fixed position for square
            
            # Randomly choose orientation of the rectangle (horizontal or vertical)
            is_horizontal = random.choice([True, False])
            
            if is_horizontal:
                # For a horizontal 1x3 rectangle, allow attachment along any of the four edges of the 2x2 square.
                # Each candidate is defined as: (square_offset, rectangle_offset, final_shape)
                #
                # Right attachments (rectangle to the right of the square)
                right_configs = [
                    ((0, 0), (0, 2), (2, 5)),  # Top aligned: square at (0,0), rectangle at (0,2)
                    ((0, 0), (1, 2), (2, 5))   # Bottom aligned: square at (0,0), rectangle at (1,2)
                ]
                # Left attachments (rectangle to the left)
                left_configs = [
                    ((0, 3), (0, 0), (2, 5)),  # Top aligned: square at (0,3), rectangle at (0,0)
                    ((0, 3), (1, 0), (2, 5))   # Bottom aligned: square at (0,3), rectangle at (1,0)
                ]
                # Top attachments (rectangle above the square)
                top_configs = [
                    ((1, 0), (0, 0), (3, 3)),  # Left aligned: square at (1,0), rectangle at (0,0)
                    ((1, 1), (0, 0), (3, 3))   # Right aligned: square at (1,1), rectangle at (0,0)
                ]
                # Bottom attachments (rectangle below the square)
                bottom_configs = [
                    ((0, 0), (2, 0), (3, 3)),  # Left aligned: square at (0,0), rectangle at (2,0)
                    ((0, 1), (2, 0), (3, 3))   # Right aligned: square at (0,1), rectangle at (2,0)
                ]
                positions = right_configs + left_configs + top_configs + bottom_configs
            else:
                # For a vertical 3x1 rectangle, allow attachment along any of the four edges of the 2x2 square.
                #
                # Top attachments (rectangle above the square)
                top_configs = [
                    ((3, 0), (0, 0), (5, 2)),  # Left aligned: square at (3,0), rectangle at (0,0)
                    ((3, 0), (0, 1), (5, 2))   # Right aligned: square at (3,0), rectangle at (0,1)
                ]
                # Bottom attachments (rectangle below the square)
                bottom_configs = [
                    ((0, 0), (2, 0), (5, 2)),  # Left aligned: square at (0,0), rectangle at (2,0)
                    ((0, 0), (2, 1), (5, 2))   # Right aligned: square at (0,0), rectangle at (2,1)
                ]
                # Left attachments (rectangle to the left of the square)
                left_configs = [
                    ((0, 1), (0, 0), (3, 3)),  # Top aligned: square at (0,1), rectangle at (0,0)
                    ((1, 1), (0, 0), (3, 3))   # Bottom aligned: square at (1,1), rectangle at (0,0)
                ]
                # Right attachments (rectangle to the right)
                right_configs = [
                    ((0, 0), (0, 2), (3, 3)),  # Top aligned: square at (0,0), rectangle at (0,2)
                    ((1, 0), (0, 2), (3, 3))   # Bottom aligned: square at (1,0), rectangle at (0,2)
                ]
                positions = top_configs + bottom_configs + left_configs + right_configs
            
            # Randomly choose one candidate configuration:
            square_offset, rect_offset, final_shape = random.choice(positions)
            
            # Create the block using the chosen final shape:
            block = np.zeros(final_shape, dtype=int)
            
            # Place the 2x2 square at the designated square_offset:
            block[square_offset[0]:square_offset[0]+2, square_offset[1]:square_offset[1]+2] = input_color
            
            # Place the rectangle in the chosen orientation:
            if is_horizontal:
                # Horizontal rectangle: 1x3
                r, c = rect_offset
                block[r, c:c+3] = input_color
            else:
                # Vertical rectangle: 3x1
                r, c = rect_offset
                block[r:r+3, c] = input_color
            
            # Verify that both square and rectangle are present
            square_count = np.sum(block[square_pos[0]:square_pos[0]+2, square_pos[1]:square_pos[1]+2] == input_color)
            rect_count = np.sum(block == input_color) - square_count
            
            # If either part is missing, create the block again
            if square_count != 4 or rect_count != 3:
                return create_building_block()
        
            return block
        
        def rotate_block(block):
            """Helper function to rotate a block by 90 degrees"""
            return np.rot90(block)

        def place_blocks():
            max_attempts = 10  # Add maximum attempts for the overall function
            for attempt in range(max_attempts):
                num_blocks = random.randint(2, 3)
                blocks = [create_building_block() for _ in range(num_blocks)]
                
                # Create a working grid large enough for the blocks
                max_size = 15
                working_grid = np.zeros((max_size, max_size), dtype=int)
                
                # Place first block in center
                first_block = blocks[0]
                start_r = (max_size - first_block.shape[0]) // 2
                start_c = (max_size - first_block.shape[1]) // 2
                working_grid[start_r:start_r + first_block.shape[0], 
                            start_c:start_c + first_block.shape[1]] = first_block
                
                # For each remaining block
                all_blocks_placed = True
                for block in blocks[1:]:
                    placed = False
                    attempts = 0
                    max_placement_attempts = 50  # Reduced from 100 to fail faster
                    
                    while not placed and attempts < max_placement_attempts:
                        attempts += 1
                        
                        # Find all colored cells in the current structure
                        colored_cells = list(zip(*np.where(working_grid == input_color)))
                        if not colored_cells:
                            continue
                        
                        # Choose a random colored cell as connection point
                        anchor_r, anchor_c = random.choice(colored_cells)
                        
                        # Try placing with larger offsets to prevent overlap
                        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
                        random.shuffle(directions)
                        
                        # Try all possible rotations of the block
                        current_block = block
                        for _ in range(4):  # Try all 4 rotations (0, 90, 180, 270 degrees)
                            for dr, dc in directions:
                                try:
                                    new_r = anchor_r + dr
                                    new_c = anchor_c + dc
                                    
                                    # Check bounds
                                    if (new_r >= 0 and new_r + current_block.shape[0] <= max_size and 
                                        new_c >= 0 and new_c + current_block.shape[1] <= max_size):
                                        
                                        # Check if space is empty
                                        target_region = working_grid[new_r:new_r + current_block.shape[0],
                                                                   new_c:new_c + current_block.shape[1]]
                                        
                                        if np.all(target_region == 0):
                                            # Create a temporary grid to check connectivity
                                            temp_grid = working_grid.copy()
                                            temp_grid[new_r:new_r + current_block.shape[0],
                                                    new_c:new_c + current_block.shape[1]] = current_block
                                            
                                            # Verify 8-way connectivity
                                            objects = find_connected_objects(temp_grid, diagonal_connectivity=True)
                                            if len(objects) == 1:
                                                working_grid = temp_grid
                                                placed = True
                                                break
                                
                                except IndexError:
                                    continue
                            
                            if placed:
                                break
                        
                        if not placed:
                            all_blocks_placed = False
                            break
                    
                    if not placed:
                        continue
                
                if all_blocks_placed:
                    # Verify final configuration
                    objects = find_connected_objects(working_grid, diagonal_connectivity=True)
                    if len(objects) == 1:
                        # Trim and return the working grid
                        rows_with_content = np.any(working_grid != 0, axis=1)
                        cols_with_content = np.any(working_grid != 0, axis=0)
                        return working_grid[rows_with_content][:, cols_with_content]
            
            return None  # Return None if all attempts fail
        
        # Generate the object and place it on the grid
        object_matrix = retry(place_blocks, 
                             lambda x: x is not None and np.count_nonzero(x) > 0,
                             max_attempts=10)
        
        if object_matrix is None or object_matrix.shape[0] >= rows - 2 or object_matrix.shape[1] >= cols - 2:
            # Fallback if object is too large for the grid
            return self.create_fallback_input(taskvars)
        
        # Ensure there's enough space to place the object
        max_row = rows - object_matrix.shape[0] - 1
        max_col = cols - object_matrix.shape[1] - 1
        
        if max_row < 1 or max_col < 1:
            return self.create_fallback_input(taskvars)
        
        # Place the object randomly on the grid
        start_row = random.randint(1, max_row)
        start_col = random.randint(1, max_col)
        grid[start_row:start_row + object_matrix.shape[0], 
             start_col:start_col + object_matrix.shape[1]] = object_matrix
        
        return grid

    def create_fallback_input(self, taskvars):
        """Create a simple valid configuration when the main generation fails"""
        rows = taskvars['rows']
        cols = taskvars['columns']
        input_color = taskvars['input_color']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Create a simple L-shaped structure (2x2 square with 1x3 rectangle)
        simple_shape = np.zeros((4, 4), dtype=int)
        simple_shape[1:3, 1:3] = input_color  # 2x2 square
        simple_shape[3, 1:4] = input_color    # 1x3 rectangle
        
        # Calculate maximum valid starting positions
        max_row = rows - simple_shape.shape[0]
        max_col = cols - simple_shape.shape[1]
        
        # Ensure we have space to place the shape
        if max_row < 1 or max_col < 1:
            start_row = (rows - simple_shape.shape[0]) // 2
            start_col = (cols - simple_shape.shape[1]) // 2
        else:
            # Random placement within valid bounds
            start_row = random.randint(1, max_row)
            start_col = random.randint(1, max_col)
        
        grid[start_row:start_row + simple_shape.shape[0],
             start_col:start_col + simple_shape.shape[1]] = simple_shape
        
        return grid

    def transform_input(self, grid, taskvars):
        color_1 = taskvars['color_1']  # for squares
        color_2 = taskvars['color_2']  # for rectangles

        # Start with an output that is entirely background.
        output_grid = np.zeros_like(grid, dtype=int)

        # Identify the single 8-way connected object.
        objects = find_connected_objects(grid, diagonal_connectivity=True)
        if len(objects) != 1:
            return output_grid

        # Build a working grid marking object cells with 1.
        working_grid = np.zeros_like(grid, dtype=int)
        for cell in objects[0].cells:
            working_grid[cell[0], cell[1]] = 1

        grid_rows, grid_cols = working_grid.shape

        # S: the set of (r,c) that belongs to the object.
        S = set(zip(*np.where(working_grid == 1)))

        # ---------------------------------------------------------------------
        # Define candidate configurations for a building block.
        # Each candidate configuration is meant to mimic one way
        # that the 2×2 square and its attached 1×3 (or 3×1) rectangle can be formed.
        #
        # For the generation, these came from:
        #   • Horizontal-right attachments:
        #       - ((0,0), (0,2), (2,5))  and ((0,0), (1,2), (2,5))
        #   • Horizontal-left attachments:
        #       - ((0,3), (0,0), (2,5))  and ((0,3), (1,0), (2,5))
        #   • Horizontal top attachments:
        #       - ((1,0), (0,0), (3,3))  and ((1,1), (0,0), (3,3))
        #   • Horizontal bottom attachments:
        #       - ((0,0), (2,0), (3,3))  and ((0,1), (2,0), (3,3))
        #   • And similarly for vertical placements.
        #
        # In the candidate config we store:
        #   'final_shape'  : the window size (either (2,5), (3,3), or (5,2)).
        #   'square_offset': the top-left of the 2×2 square inside that window.
        #   'rect_offset'  : the top-left of the rectangle region within that window.
        #
        # (Later we compute the full list of relative offsets.)
        # ---------------------------------------------------------------------
        candidate_configs = [
            # Horizontal-right attachments (window is 2×5, rectangle is 1×3)
            {'final_shape': (2, 5), 'square_offset': (0, 0), 'rect_offset': (0, 2)},
            {'final_shape': (2, 5), 'square_offset': (0, 0), 'rect_offset': (1, 2)},
            # Horizontal-left attachments
            {'final_shape': (2, 5), 'square_offset': (0, 3), 'rect_offset': (0, 0)},
            {'final_shape': (2, 5), 'square_offset': (0, 3), 'rect_offset': (1, 0)},
            # Horizontal top attachments (window 3×3; rectangle takes the top row)
            {'final_shape': (3, 3), 'square_offset': (1, 0), 'rect_offset': (0, 0)},
            {'final_shape': (3, 3), 'square_offset': (1, 1), 'rect_offset': (0, 0)},
            # Horizontal bottom attachments (window 3×3; rectangle takes the bottom row)
            {'final_shape': (3, 3), 'square_offset': (0, 0), 'rect_offset': (2, 0)},
            {'final_shape': (3, 3), 'square_offset': (0, 1), 'rect_offset': (2, 0)},
            # Vertical top attachments (window 5×2; rectangle is 3×1 on the top)
            {'final_shape': (5, 2), 'square_offset': (3, 0), 'rect_offset': (0, 0)},
            {'final_shape': (5, 2), 'square_offset': (3, 0), 'rect_offset': (0, 1)},
            # Vertical bottom attachments (window 5×2; rectangle is 3×1 on the bottom)
            {'final_shape': (5, 2), 'square_offset': (0, 0), 'rect_offset': (2, 0)},
            {'final_shape': (5, 2), 'square_offset': (0, 0), 'rect_offset': (2, 1)},
            # Vertical left attachments (window 3×3; rectangle is the left column)
            {'final_shape': (3, 3), 'square_offset': (0, 1), 'rect_offset': (0, 0)},
            {'final_shape': (3, 3), 'square_offset': (1, 1), 'rect_offset': (0, 0)},
            # Vertical right attachments (window 3×3; rectangle is the right column)
            {'final_shape': (3, 3), 'square_offset': (0, 0), 'rect_offset': (0, 2)},
            {'final_shape': (3, 3), 'square_offset': (1, 0), 'rect_offset': (0, 2)},
        ]

        # ---------------------------------------------------------------------
        # For each candidate configuration, compute the full list of relative
        # offsets that define the 2×2 square and 1×3 rectangle.
        # (Also, skip a candidate if the square and rectangle would overlap—
        # we require 4 + 3 = 7 distinct cells.)
        # ---------------------------------------------------------------------
        def compute_candidate_offsets(config):
            fr, fc = config['final_shape']
            so_r, so_c = config['square_offset']
            # The square covers a 2×2 region:
            sq_offsets = [(so_r, so_c), (so_r, so_c + 1),
                          (so_r + 1, so_c), (so_r + 1, so_c + 1)]
            # Compute rectangle offsets according to window shape.
            ro, rc = config['rect_offset']
            rect_offsets = []
            if config['final_shape'] == (2, 5):  # horizontal candidate (1×3 rectangle)
                # Rectangle occupies one row (either row 0 or row 1)
                for delta in range(3):
                    rect_offsets.append((ro, rc + delta))
            elif config['final_shape'] == (5, 2):  # vertical candidate (3×1 rectangle)
                for delta in range(3):
                    rect_offsets.append((ro + delta, rc))
            elif config['final_shape'] == (3, 3):
                # In a 3×3 window the candidate can be interpreted as either horizontal or vertical.
                # Decide based on the relation between rect_offset and square_offset.
                if ro < so_r:
                    # Top attachment (rectangle is top row)
                    for delta in range(3):
                        rect_offsets.append((0, delta))
                elif ro > so_r:
                    # Bottom attachment
                    for delta in range(3):
                        rect_offsets.append((2, delta))
                elif rc < so_c:
                    # Left attachment
                    for delta in range(3):
                        rect_offsets.append((delta, 0))
                elif rc > so_c:
                    # Right attachment
                    for delta in range(3):
                        rect_offsets.append((delta, 2))
                else:
                    # Ill‐defined configuration.
                    return None, None
            else:
                return None, None
            # Ensure the two sets of offsets are disjoint (i.e. total 7 cells).
            if len(set(sq_offsets + rect_offsets)) != 7:
                return None, None
            return sq_offsets, rect_offsets

        valid_candidates = []
        for config in candidate_configs:
            sq_rel, rect_rel = compute_candidate_offsets(config)
            if sq_rel is None:
                continue
            # The candidate pattern (relative positions) is the union of square and rectangle offsets.
            cand_pattern = set(sq_rel + rect_rel)
            window_rows, window_cols = config['final_shape']
            for i in range(grid_rows - window_rows + 1):
                for j in range(grid_cols - window_cols + 1):
                    # Compute the candidate's absolute cell positions.
                    cand_cells = {(i + dr, j + dc) for (dr, dc) in cand_pattern}
                    if cand_cells.issubset(S):
                        abs_sq = [(i + dr, j + dc) for (dr, dc) in sq_rel]
                        abs_rect = [(i + dr, j + dc) for (dr, dc) in rect_rel]
                        valid_candidates.append({
                            'top_left': (i, j),
                            'square_cells': frozenset(abs_sq),
                            'rect_cells': frozenset(abs_rect),
                            'all_cells': frozenset(abs_sq + abs_rect)
                        })

        # ---------------------------------------------------------------------
        # Now search for a combination (of 2 or 3 candidates) whose union is
        # exactly S. (There are very few building blocks, so an exhaustive search
        # is acceptable.)
        # ---------------------------------------------------------------------
        selected_candidates = None
        for r in range(2, 4):  # try 2 or 3 candidates
            for combo in itertools.combinations(valid_candidates, r):
                union_cells = frozenset().union(*(c['all_cells'] for c in combo))
                if union_cells == S:
                    selected_candidates = combo
                    break
            if selected_candidates is not None:
                break

        if selected_candidates is None:
            # Fallback: if no valid combination is found, return an all-zero grid.
            return output_grid

        # ---------------------------------------------------------------------
        # PHASE 3: Produce the output using the selected candidates.
        # For each candidate, assign its 2×2 square cells the square color and
        # its rectangle cells the rectangle color—if a cell is claimed by any square,
        # that color takes precedence.
        # ---------------------------------------------------------------------
        cell_color = {}
        for cand in selected_candidates:
            for cell in cand['rect_cells']:
                if cell not in cell_color:
                    cell_color[cell] = color_2
            for cell in cand['square_cells']:
                cell_color[cell] = color_1

        for (r, c), col in cell_color.items():
            output_grid[r, c] = col

        return output_grid

    def create_grids(self):
        taskvars = {
            'rows': random.randint(9, 30),
            'columns': random.randint(9, 30),
            'input_color': random.randint(1, 9),
            'color_1': random.randint(1, 9),
            'color_2': random.randint(1, 9)
        }
        while len({taskvars['input_color'], taskvars['color_1'], taskvars['color_2']}) < 3:
            taskvars['color_2'] = random.randint(1, 9)
        
        train_pairs = []
        for _ in range(random.randint(3, 4)):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
    



