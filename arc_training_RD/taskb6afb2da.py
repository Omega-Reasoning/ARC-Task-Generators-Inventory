# from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
# import numpy as np
# import random
# from transformation_library import find_connected_objects  # Temporarily disabled

# class BlockBoundaryColoringTaskGenerator(ARCTaskGenerator):
#     def __init__(self):
#         input_reasoning_chain = [
#             "Input grids are of size MXM.", 
#             "The grid consists of exactly square or rectangular blocks of grey color.", 
#             "The blocks are usually greater than or equal to 4x4 for a square block and 4x5 or 5x4 for a rectangular block.",
#             "They are spaced uniformly.",
#             "There surely exists at least one square block and one rectangular block in each grid."
#         ]
        
#         transformation_reasoning_chain = [
#             "The output grid is copied from the input grid.",
#             "The blocks are split into two parts: the outermost cells and the inner cells.",
#             "Outer boundary cells are colored red or blue depending on position.",
#             "Inner cells are colored yellow."
#         ]
        
#         super().__init__(input_reasoning_chain, transformation_reasoning_chain)

#     def create_input(self, gridvars=None):
#         if gridvars is None:
#             gridvars = {}

#         object_color = gridvars.get('object_color', 5)
#         grid_size = gridvars.get('grid_size', random.randint(12, 20))
#         grid = np.zeros((grid_size, grid_size), dtype=int)
#         occupied = np.zeros_like(grid, dtype=bool)

#         max_blocks = max(2, (grid_size * grid_size) // 80)
#         num_blocks = min(gridvars.get('num_blocks', random.randint(2, 4)), max_blocks)
#         block_types = ['square', 'rectangle'] + ['random'] * (num_blocks - 2)
#         random.shuffle(block_types)

#         min_size = 4
#         placed_blocks = 0

#         for i, block_type in enumerate(block_types[:num_blocks]):
#             if block_type == 'square':
#                 size = random.randint(min_size, min(7, grid_size // 3))
#                 height = width = size
#             else:
#                 height = random.randint(min_size, min(7, grid_size // 3))
#                 width = random.randint(min_size, min(7, grid_size // 3))
#                 while block_type == 'rectangle' and height == width:
#                     width = random.randint(min_size, min(7, grid_size // 3))

#             for attempt in range(50):
#                 max_row = grid_size - height - 1
#                 max_col = grid_size - width - 1
#                 if max_row <= 1 or max_col <= 1:
#                     break

#                 row = random.randint(1, max_row)
#                 col = random.randint(1, max_col)
#                 r0, r1 = max(0, row - 1), min(grid_size, row + height + 1)
#                 c0, c1 = max(0, col - 1), min(grid_size, col + width + 1)

#                 if np.any(occupied[r0:r1, c0:c1]):
#                     continue

#                 grid[row:row+height, col:col+width] = object_color
#                 occupied[row:row+height, col:col+width] = True
#                 placed_blocks += 1
#                 break

#         if placed_blocks < 2:
#             grid = np.zeros((12, 12), dtype=int)
#             grid[2:6, 2:6] = object_color
#             grid[2:6, 8:12] = object_color

#         return grid


#     def transform_input(self, input_grid, gridvars=None):
#         if gridvars is None:
#             gridvars = {}

#         object_color = gridvars.get('object_color', 5)
#         output_grid = input_grid.copy()

#         try:
#             blocks = find_connected_objects(input_grid, diagonal_connectivity=False, background=0, monochromatic=True)
#             print(f"Found {len(blocks)} blocks")
#         except Exception as e:
#             print(f"Error in find_connected_objects: {e}")
#             return input_grid.copy()

#         return output_grid


#     def create_grids(self):
#         colors = random.sample(range(1, 10), 4)
#         gridvars = {
#             'object_color': colors[0],
#             'bound_color1': colors[1],
#             'bound_color2': colors[2],
#             'fill_color': colors[3]
#         }

#         num_train = 1  # Keep it minimal for testing
#         train_pairs = []

#         for i in range(num_train):
#             print(f"Creating training example {i+1}")
#             example_vars = gridvars.copy()
#             example_vars['grid_size'] = random.randint(12, 20)
#             example_vars['num_blocks'] = random.randint(2, 4)

#             input_grid = self.create_input(example_vars)
#             output_grid = self.transform_input(input_grid, example_vars)

#             train_pairs.append(GridPair(input=input_grid.tolist(), output=output_grid.tolist()))

#         print("Creating test grid...")
#         test_vars = gridvars.copy()
#         test_vars['grid_size'] = random.randint(12, 20)
#         test_vars['num_blocks'] = random.randint(2, 4)

#         test_input = self.create_input(test_vars)
#         test_output = self.transform_input(test_input, test_vars)

#         #test_pair = GridPair(input=test_input, output=test_output)
#         test_pair = GridPair(input=test_input.tolist(), output=test_output.tolist())
#         return gridvars, TrainTestData(train=train_pairs, test=test_pair)

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject
from input_library import Contiguity, create_object, retry

class BlockSplittingTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size MXM.",
            "The grid consists of exactly square or rectangular blocks of red color.",
            "The blocks are usually greater than or equal to 4x4 for a square block and 4x5 or 5x4 for a rectangular block.",
            "They are spaced uniformly.",
            "There surely exists at least one square block and one rectangular block in each grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The blocks are split into two parts such as the outermost cells being one part and the inner cells being the other part.",
            "For each block outermost cells form contribute to form boundaries in specific colors, based on their position:",
            "- If the boundary forming cells are in 4-way connection to inner block, then fill it with grey  color",
            "- If the boundary forming cells are corner cells of the boundary then fill it with yellow color.",
            "And for the inner cells, fill it with blue color."
        ]
        
        taskvars_definitions = {}
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Define grid variables including colors
        gridvars = {}
        
        # Randomly select colors ensuring they are all different
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        gridvars['object_color'] = available_colors[0]  # Red (original block color)
        gridvars['fill_color'] = available_colors[1]    # Blue (inner cells)
        gridvars['bound_color1'] = available_colors[2]  # Yellow (corner cells)
        gridvars['bound_color2'] = available_colors[3]  # Grey (4-way connected boundary cells)
        
        # Number of train examples
        n_train = random.randint(3, 5)
        
        # Create train and test data
        train_pairs = []
        for _ in range(n_train):
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid.copy(), gridvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(gridvars)
        test_output = self.transform_input(test_input.copy(), gridvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)
    
    def create_input(self, gridvars):
        # Create a grid with some square and rectangular blocks
        object_color = gridvars['object_color']
        
        # Choose a random grid size
        grid_size = random.randint(12, 20)
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine number of blocks to place
        num_blocks = random.randint(2, 4)
        
        # Create blocks with specified properties
        blocks = []
        
        # Ensure at least one square and one rectangular block
        block_types = ['square'] + ['rectangle'] + ['random'] * (num_blocks - 2)
        random.shuffle(block_types)
        
        # Track occupied areas to prevent overlaps
        occupied = np.zeros_like(grid, dtype=bool)
        
        for block_type in block_types:
            # Try to place block
            for attempt in range(50):  # Limit attempts to avoid infinite loops
                if block_type == 'square':
                    size = random.randint(4, 6)
                    width, height = size, size
                elif block_type == 'rectangle':
                    if random.choice([True, False]):
                        width, height = random.randint(4, 6), random.randint(5, 7)
                    else:
                        width, height = random.randint(5, 7), random.randint(4, 6)
                else:  # random block
                    if random.choice([True, False]):
                        size = random.randint(4, 6)
                        width, height = size, size
                    else:
                        if random.choice([True, False]):
                            width, height = random.randint(4, 6), random.randint(5, 7)
                        else:
                            width, height = random.randint(5, 7), random.randint(4, 6)
                
                # Choose random position ensuring block fits in grid
                top = random.randint(0, grid_size - height)
                left = random.randint(0, grid_size - width)
                
                # Check if position is valid (not overlapping with existing blocks and has margin)
                margin = 2
                valid = True
                
                # Area to check (with margin)
                check_top = max(0, top - margin)
                check_left = max(0, left - margin)
                check_bottom = min(grid_size, top + height + margin)
                check_right = min(grid_size, left + width + margin)
                
                if np.any(occupied[check_top:check_bottom, check_left:check_right]):
                    valid = False
                
                if valid:
                    # Place the block
                    grid[top:top+height, left:left+width] = object_color
                    occupied[top:top+height, left:left+width] = True
                    blocks.append((top, left, height, width))
                    break
            
            # If we couldn't place a block after many attempts, just continue
            # This could lead to fewer blocks than requested
        
        return grid
    
    def transform_input(self, grid, gridvars):
        object_color = gridvars['object_color']
        fill_color = gridvars['fill_color']
        bound_color1 = gridvars['bound_color1']  # Yellow (corner cells)
        bound_color2 = gridvars['bound_color2']  # Grey (4-way connected boundary cells)
        
        # Find blocks in the grid
        blocks = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Process each block
        for block in blocks:
            if not block.has_color(object_color):
                continue
                
            # Convert block to array format
            block_array = block.to_array()
            r_start, c_start = block.bounding_box[0].start, block.bounding_box[1].start
            
            # Get dimensions
            height, width = block_array.shape
            
            # Create a mask for the inner cells
            inner_mask = np.zeros_like(block_array, dtype=bool)
            if height > 2 and width > 2:
                inner_mask[1:-1, 1:-1] = True
            
            # Create masks for different types of boundary cells
            boundary_mask = (block_array == object_color) & ~inner_mask
            
            # Corner cells are at the corners of the block
            corner_mask = np.zeros_like(block_array, dtype=bool)
            if height > 0 and width > 0:
                corner_mask[0, 0] = True
            if height > 0 and width > 1:
                corner_mask[0, width-1] = True
            if height > 1 and width > 0:
                corner_mask[height-1, 0] = True
            if height > 1 and width > 1:
                corner_mask[height-1, width-1] = True
            
            # 4-way connected boundary cells are the rest of the boundary
            edge_mask = boundary_mask & ~corner_mask
            
            # Apply colors based on masks
            for r in range(height):
                for c in range(width):
                    if inner_mask[r, c]:
                        grid[r + r_start, c + c_start] = fill_color
                    elif corner_mask[r, c]:
                        grid[r + r_start, c + c_start] = bound_color1
                    elif edge_mask[r, c]:
                        grid[r + r_start, c + c_start] = bound_color2
        
        return grid

