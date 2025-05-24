from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject
from input_library import Contiguity, create_object, retry

class BlockSplittingTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size MXM.",
            "The grid consists of exactly square or rectangular blocks of {color('object_color')}.",
            "The blocks are usually greater than or equal to 4x4 for a square block and 4x5 or 5x4 for a rectangular block.",
            "They are spaced uniformly.",
            "There surely exists at least one square block and one rectangular block in each grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The blocks are split into two parts such as the outermost cells being one part and the inner cells being the other part.",
            "For each block outermost cells form contribute to form boundaries in specific colors, based on their position:",
            "- If the boundary forming cells are in 4-way connection to inner block, then fill it with {color('bound_color2')}",
            "- If the boundary forming cells are corner cells of the boundary then fill it with {color('bound_color1')}.",
            "And for the inner cells, fill it with {color('fill_color')}."
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
        # Create a grid with some square and rectangular blocks
        object_color = taskvars['object_color']
        
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
    
    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        fill_color = taskvars['fill_color']
        bound_color1 = taskvars['bound_color1']  # Corner cells
        bound_color2 = taskvars['bound_color2']  # 4-way connected boundary cells
        
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
    
    def create_grids(self):
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        # Randomly select colors ensuring they are all different
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        taskvars = {
            'object_color': available_colors[0],  # Original block color
            'fill_color': available_colors[1],    # Inner cells
            'bound_color1': available_colors[2],  # Corner cells
            'bound_color2': available_colors[3]   # 4-way connected boundary cells
        }
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Replace color placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('object_color')}", color_fmt('object_color'))
                 .replace("{color('fill_color')}", color_fmt('fill_color'))
                 .replace("{color('bound_color1')}", color_fmt('bound_color1'))
                 .replace("{color('bound_color2')}", color_fmt('bound_color2'))
            for chain in self.input_reasoning_chain
        ]
        
        self.transformation_reasoning_chain = [
            chain.replace("{color('object_color')}", color_fmt('object_color'))
                 .replace("{color('fill_color')}", color_fmt('fill_color'))
                 .replace("{color('bound_color1')}", color_fmt('bound_color1'))
                 .replace("{color('bound_color2')}", color_fmt('bound_color2'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Create train and test data
        for _ in range(num_train_pairs):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid.copy(), taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input.copy(), taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
