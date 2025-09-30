import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, find_connected_objects
from input_library import create_object, retry

class ARCAGITaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has dimensions {vars['rows']} X {vars['rows']}",
            "The grid contains exactly one 8-way connected object colored {color('input_color')}",
            "This object consists of two distinct building blocks",
            "The first building block is positioned at the grids center and is either a 2 X 2 square or a single cell",
            "The second building block is a 4-way connected object composed of squares and rectangles",
            "The second building block connects to the first block at exactly three points: top, right, and bottom",
            "For a 2x2 first block, connections are made at the first cell along each edge when moving clockwise",
            "For a single cell first block, connections are made directly at the top, right, and bottom of the center cell",
            "All remaining grid cells are empty (value 0)"
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First identify the 8-way connected object.",
            "Then identify the first building block at the center of the input grid (either a 2 X 2 square or a single cell).",
            "Next, identify the 4-way connected building block that connects to the top of the first building block.",
            "Rotate this connected building block 90 degrees counterclockwise.",
            "If the first block is a 2x2 square: Place the rotated building block to connect to the bottom-left cell of the square, coloring it with {color('output_color')}.",
            "If the first block is a single cell: Place the rotated building block to connect directly to the left of the center cell, coloring it with {color('output_color')}.",
            "The remaining cells are empty (value 0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        input_color = taskvars['input_color']
        grid = np.zeros((rows, rows), dtype=int)
        
        # Create first block (either a 2x2 square or a single cell)
        use_square = random.choice([True, False])
        if use_square:
            first_block = np.full((2, 2), input_color)
        else:
            first_block = np.array([[input_color]])
        
        # Place first block at the center of the grid
        center = rows // 2
        start_x = center - first_block.shape[0] // 2
        start_y = center - first_block.shape[1] // 2
        grid[start_x:start_x + first_block.shape[0], start_y:start_y + first_block.shape[1]] = first_block
        
        # ------------------------------------
        # Helper functions for placement
        # ------------------------------------
        def can_place(grid, pattern, x, y):
            if x < 0 or y < 0 or x + pattern.shape[0] > rows or y + pattern.shape[1] > rows:
                return False
            # Check for overlap with already filled cells
            overlap = grid[x:x + pattern.shape[0], y:y + pattern.shape[1]] != 0
            return not overlap.any()
        
        def place_pattern(grid, pattern, x, y):
            grid[x:x + pattern.shape[0], y:y + pattern.shape[1]] = pattern
        
        # -----------------------------------------------------------
        # Build a simple 4-way object that consists of two parts:
        #  - "Body": a square or rectangle (each dimension no greater than 4)
        #  - "Hand": a straight line of 3–4 cells
        #
        # In its base (vertical) orientation the body sits at the top,
        # and the hand is below it. The connection cell (used for attaching)
        # is defined as the bottom cell of the hand.
        # -----------------------------------------------------------
        def create_4way_object():
            hand_length = random.choice([1,2,3, 4])
            # Randomly select a style for the body: square, rectangle, or small non-rectangular shape.
            shape_type = random.choice(["square", "rectangle", "small"])
            if shape_type == "square":
                # Square body: side length between 2 and 4.
                side = random.randint(1, 4)
                body = np.full((side, side), input_color)
            elif shape_type == "rectangle":
                # Rectangle body: either horizontal (2 rows, 3–4 columns) or vertical (3–4 rows, 2 columns).
                if random.choice([True, False]):
                    body = np.full((2, random.randint(1, 4)), input_color)
                else:
                    body = np.full((random.randint(1, 4), 2), input_color)
            elif shape_type == "small":
                # Create a small 4-way connected object as the body.
                # Option 1: A 3-cell L-shape.
                if random.choice([True, False]):
                    body = np.array([[input_color, input_color],
                                     [0,      input_color       ]])
                else:
                    # Option 2: A 4-cell T-shape.
                    body = np.array([[      0      , input_color,       0      ],
                                     [input_color, input_color, input_color]])
    
            body_height, body_width = body.shape
            overall_height = body_height + hand_length
            overall_width = body_width  # overall width defined by the body's bounding box
            object_pattern = np.zeros((overall_height, overall_width), dtype=int)
            
            # Place the body at the top.
            object_pattern[:body_height, :body_width] = body
            
            # The hand is drawn as a vertical 1-cell-wide line in the center of the body's width.
            hand_col = body_width // 2
            for i in range(body_height, overall_height):
                object_pattern[i, hand_col] = input_color
    
            # The connection cell (used later to attach the object) is the bottom cell of the hand.
            connect_offset = (overall_height - 1, hand_col)
            return object_pattern, connect_offset
        
        # -----------------------------------------------------------
        # Helper to rotate both the object and its connection offset.
        # When rotating an array 90° anticlockwise (k=1), a point (r, c)
        # in an R x C array maps to (C - 1 - c, r); similar formulas are used
        # for 180° and 270° rotations.
        # -----------------------------------------------------------
        def rotate_pattern_and_offset(pattern, connect_offset, k):
            k = k % 4  # Effective rotation
            R, C = pattern.shape
            r, c = connect_offset
            if k == 0:
                new_offset = (r, c)
            elif k == 1:
                new_offset = (C - 1 - c, r)
            elif k == 2:
                new_offset = (R - 1 - r, C - 1 - c)
            elif k == 3:
                new_offset = (c, R - 1 - r)
            return np.rot90(pattern, k=k), new_offset
        
        # Create our base 4-way object
        base_object, base_connect = create_4way_object()
        
        # -----------------------------------------------------------
        # Place the 4-way objects. Attach them via their hand connection cell.
        # The placement depends on the type of first block:
        # 
        # For a square block, we use its top-right cell as reference (i, j):
        #   • Top: attach where hand touches (i-1, j)
        #   • Right: attach at (i, j+2)
        #   • Bottom: attach at (i+2, j)
        #
        # For a single cell block (at (i, j)):
        #   • Top: attach at (i-1, j)
        #   • Right: attach at (i, j+1)
        #   • Bottom: attach at (i+1, j)
        # -----------------------------------------------------------
        if use_square:
            # For a square, the top-right cell is at (start_x, start_y+1)
            # Let (i, j) = (start_x, start_y+1)
            
            # Top pattern: attach at (i-1, j)
            desired_top_conn = (start_x - 1, start_y)
            top_pattern, top_conn = rotate_pattern_and_offset(base_object, base_connect, k=0)
            top_x = desired_top_conn[0] - top_conn[0]
            top_y = desired_top_conn[1] - top_conn[1]
            if can_place(grid, top_pattern, top_x, top_y):
                place_pattern(grid, top_pattern, top_x, top_y)
            
            # Right pattern: attach at (i, j+1) to ensure direct connection (fixes gap)
            desired_right_conn = (start_x, start_y + 2)
            right_pattern, right_conn = rotate_pattern_and_offset(base_object, base_connect, k=3)
            right_x = desired_right_conn[0] - right_conn[0]
            right_y = desired_right_conn[1] - right_conn[1]
            if can_place(grid, right_pattern, right_x, right_y):
                place_pattern(grid, right_pattern, right_x, right_y)
            
            # Bottom pattern: attach at (i+2, j), i.e., (start_x+2, start_y+1)
            desired_bottom_conn = (start_x + 2, start_y + 1)
            bottom_pattern, bottom_conn = rotate_pattern_and_offset(base_object, base_connect, k=2)
            bottom_x = desired_bottom_conn[0] - bottom_conn[0]
            bottom_y = desired_bottom_conn[1] - bottom_conn[1]
            if can_place(grid, bottom_pattern, bottom_x, bottom_y):
                place_pattern(grid, bottom_pattern, bottom_x, bottom_y)
            
            # (Note: For the square case, no left pattern is added.)
        
        else:
            # For a single cell first block at (start_x, start_y), set (i, j) = (start_x, start_y).
            
            # Top pattern: attach at (i-1, j)
            desired_top_conn = (start_x - 1, start_y)
            top_pattern, top_conn = rotate_pattern_and_offset(base_object, base_connect, k=0)
            top_x = desired_top_conn[0] - top_conn[0]
            top_y = desired_top_conn[1] - top_conn[1]
            if can_place(grid, top_pattern, top_x, top_y):
                place_pattern(grid, top_pattern, top_x, top_y)
            
            # Right pattern: attach at (i, j+1)
            desired_right_conn = (start_x, start_y + 1)
            right_pattern, right_conn = rotate_pattern_and_offset(base_object, base_connect, k=3)
            right_x = desired_right_conn[0] - right_conn[0]
            right_y = desired_right_conn[1] - right_conn[1]
            if can_place(grid, right_pattern, right_x, right_y):
                place_pattern(grid, right_pattern, right_x, right_y)
            
            # Bottom pattern: attach at (i+1, j)
            desired_bottom_conn = (start_x + 1, start_y)
            bottom_pattern, bottom_conn = rotate_pattern_and_offset(base_object, base_connect, k=2)
            bottom_x = desired_bottom_conn[0] - bottom_conn[0]
            bottom_y = desired_bottom_conn[1] - bottom_conn[1]
            if can_place(grid, bottom_pattern, bottom_x, bottom_y):
                place_pattern(grid, bottom_pattern, bottom_x, bottom_y)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        input_color, output_color = taskvars['input_color'], taskvars['output_color']
        rows = grid.shape[0]
        center = rows // 2

        # Determine whether the main block is a square (2x2) or a single cell
        is_square = grid[center - 1, center - 1] == input_color
        
        # Set up main block parameters based on type
        if is_square:
            main_i, main_j = center - 1, center - 1
            main_block_coords = {(main_i, main_j),
                               (main_i, main_j + 1),
                               (main_i + 1, main_j),
                               (main_i + 1, main_j + 1)}
            start_r, start_c = main_i - 1, main_j
            connectivity = [(-1, 0), (1, 0), (0, -1), (0, 1),
                          (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            main_i, main_j = center, center
            main_block_coords = {(main_i, main_j)}
            start_r, start_c = main_i - 1, main_j
            connectivity = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Flood fill to extract the attached block
        to_explore = [(start_r, start_c)]
        explored = set()
        obj_coords = set()

        while to_explore:
            r, c = to_explore.pop()
            if (r, c) in explored:
                continue
            if not (0 <= r < rows and 0 <= c < rows):
                continue
            if grid[r, c] != input_color:
                continue
            if (r, c) in main_block_coords:
                continue
            
            explored.add((r, c))
            obj_coords.add((r, c))
            
            # Continue flood fill
            for dr, dc in connectivity:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < rows:
                    to_explore.append((nr, nc))

        # Rotate and place the object
        coords_list = list(obj_coords)
        coords_array = np.array([[coord[0] for coord in coords_list],
                                [coord[1] for coord in coords_list]])
        rotated = np.rot90(coords_array, k=3)
    
        # Apply the transformed coordinates
        for i in range(rotated.shape[0]):
            r, c = rotated[i][0], rotated[i][1]
            if 0 <= r < rows and 0 <= c < rows:
                if is_square:
                    output_grid[r + 1, c] = output_color  # +1 to connect to bottom of 2x2 block

                else:
                    output_grid[r, c] = output_color

        return output_grid
    
    
    def create_grids(self):
        rows = random.choice([r for r in range(14, 31, 2)])
        input_color, output_color = random.sample(range(1, 10), 2)
        
        taskvars = {'rows': rows, 'input_color': input_color, 'output_color': output_color}
        
        train_test_data = {
            'train': [{'input': (inp := self.create_input(taskvars, {})), 'output': self.transform_input(inp, taskvars)} for _ in range(3)],
            'test': [{'input': (inp := self.create_input(taskvars, {})), 'output': self.transform_input(inp, taskvars)}]
        }
        
        return taskvars, train_test_data
