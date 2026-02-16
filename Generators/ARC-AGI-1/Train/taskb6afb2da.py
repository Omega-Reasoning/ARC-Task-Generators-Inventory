from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.transformation_library import find_connected_objects, GridObject
from Framework.input_library import Contiguity, create_object, retry

class Taskb6afb2daGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grid is of size {vars['n']} x {vars['n']} and are square grids.",
            "The grid consists of exactly square or rectangular blocks of {color('input_color')} color.",
            "The blocks are usually greater than or equal to 4x4 for a square block and 4x5 or 5x4 for a rectangular block.",
            "They are spaced uniformly.",
            "There surely exists at least one square block and one rectangular block in each grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is transformed from the input grid.",
            "The blocks are split into two parts such as the outermost cells being one part and the inner cells being the other part.",
            "For each block outermost cells form contribute to form boundaries in specific colors, based on their position:",
            "If the boundary forming cells are corner cells of the boundary then fill it with {color('fill_color')}.",
            "If the boundary forming cells are in 4-way connection to inner block, then fill it with blue (1) color",
            "And for the inner cells, fill it with {color('bound_color2')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with square and rectangular blocks."""
        grid_size = taskvars['n']  # Use task variable instead of gridvars
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Use input_color for all input blocks
        input_color = taskvars['input_color']
        
        # Determine number of blocks to place
        num_blocks = random.randint(2, 4)
        
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
                    # Place the block in input_color
                    grid[top:top+height, left:left+width] = input_color
                    occupied[top:top+height, left:left+width] = True
                    break
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by coloring block boundaries and interiors with different colors."""
        # Start with a copy of the input grid
        output_grid = grid.copy()
        
        input_color = taskvars['input_color']  # Use task variable
        
        # Based on the expected output pattern:
        # Corners: fill_color
        # Edges: blue (1) - FIXED COLOR
        # Interior: bound_color2
        
        corner_color = taskvars['fill_color']      # Corners get fill_color
        edge_color = 1                             # Edges get blue (1) - FIXED
        interior_color = taskvars['bound_color2']  # Interior gets bound_color2
        
        # Find blocks in the input grid
        blocks = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Process each block and transform it in the output
        for block in blocks:
            if not block.has_color(input_color):
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
            boundary_mask = (block_array == input_color) & ~inner_mask
            
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
            
            # Apply transformation colors ONLY in the output grid
            for r in range(height):
                for c in range(width):
                    if block_array[r, c] == input_color:  # Only transform input_color cells
                        if corner_mask[r, c]:
                            output_grid[r + r_start, c + c_start] = corner_color
                        elif edge_mask[r, c]:
                            output_grid[r + r_start, c + c_start] = edge_color
                        elif inner_mask[r, c]:
                            output_grid[r + r_start, c + c_start] = interior_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Randomly select colors ensuring they are all different (excluding blue=1 which is fixed)
        available_colors = [c for c in range(1, 10) if c != 1]  # Exclude blue (1) as it's fixed for edges
        random.shuffle(available_colors)

        # Generate grid size
        grid_size = random.randint(12, 20)

        # Store task variables - edge color is always blue (1), so not included
        taskvars = {
            'n': grid_size,                        # Grid size
            'input_color': available_colors[0],    # Color of input blocks
            'object_color': available_colors[1],   # Not used in transform but kept for compatibility
            'fill_color': available_colors[2],     # Corners  
            'bound_color1': available_colors[3],   # Not used in current pattern
            'bound_color2': available_colors[4],   # Interior
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for i in range(num_train_examples):
            # Input grid: blocks in input_color
            input_grid = self.create_input(taskvars, {})
            
            # Output grid: Transformed with colors
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }