from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, retry
from Framework.transformation_library import find_connected_objects

class Task5614dbcfGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain 3x3 same-colored (1-9) and empty (0) blocks.",
            "The entire grid is first divided into {vars['grid_size']}, 3x3 subgrids, where some subgrids are completely filled with same-colored (1–9) cells, forming colored blocks, while others remain entirely empty (0), forming empty blocks.",
            "After placing these 3x3 blocks, {vars['grid_size'] - 3} {color('cell_color')} cells are added to the grid, which are randomly distributed.",
            "These {color('cell_color')} cells may sometimes overlap with some existing cells of the colored blocks.",
            "Blocks within the same grid may have different colors, which vary across examples, and can never be {color('cell_color')}."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {vars['grid_size']//3}x{vars['grid_size']//3}.",
            "They are constructed by identifying the {vars['grid_size']}, 3x3 blocks in the input grid, where each block is either completely colored or completely empty.",
            "There are also {vars['grid_size'] - 3} {color('cell_color')} cells placed over the blocks in the input grid, but they should be ignored, as the transformation only depends on the 3x3 blocks.",
            "Once the blocks have been identified, add a single cell to the output grid to represent each 3x3 colored block from the input grid, with its color and position matching that of the respective block.",
            "The transformation preserves both the positions and the colors of the 3x3 blocks."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Choose grid size from allowed values
        grid_size = random.choice([6, 9, 12, 15, 18, 21, 24, 27, 30])
        
        # Choose cell_color that will be added randomly and ignored in transformation
        cell_color = random.randint(1, 9)
        
        # Create task variables dictionary
        taskvars = {
            'grid_size': grid_size,
            'cell_color': cell_color
        }
        
        # Generate random number of training examples (3-4)
        num_train = random.randint(3, 4)
        
        # Create train and test data
        train_test_data = self.create_grids_default(num_train, 1, taskvars)
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        cell_color = taskvars['cell_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Calculate how many 3x3 blocks we need in each dimension
        blocks_per_dim = grid_size // 3
        total_blocks = blocks_per_dim * blocks_per_dim
        
        # Create a sequence of positions where we'll place colored blocks
        # Ensure at least one 3x3 grid is empty per the constraints
        # And at least 3 blocks are colored
        possible_positions = [(r, c) for r in range(blocks_per_dim) for c in range(blocks_per_dim)]
        
        # Decide how many colored blocks to create (at least 3, at most total-1)
        min_colored_blocks = min(3, total_blocks - 1)  # Ensure we don't exceed possible blocks
        max_colored_blocks = total_blocks - 1  # Keep at least one empty
        num_colored_blocks = random.randint(min_colored_blocks, max_colored_blocks)
        
        colored_positions = random.sample(possible_positions, num_colored_blocks)
        
        # Create a block color assignment ensuring no adjacent blocks have the same color
        block_colors = {}
        
        for block_r, block_c in colored_positions:
            # Get the colors of adjacent blocks
            adjacent_colors = set()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                adj_r, adj_c = block_r + dr, block_c + dc
                if (adj_r, adj_c) in block_colors:
                    adjacent_colors.add(block_colors[(adj_r, adj_c)])
            
            # Available colors (excluding cell_color and adjacent colors)
            available_colors = [i for i in range(1, 10) if i != cell_color and i not in adjacent_colors]
            
            # If no available colors, just use a random color that's not cell_color
            # (This is a fallback but shouldn't happen with typical grid sizes and # of colors)
            if not available_colors:
                available_colors = [i for i in range(1, 10) if i != cell_color]
            
            # Choose a random color from available options
            color = random.choice(available_colors)
            block_colors[(block_r, block_c)] = color
            
            # Fill the 3x3 block with the chosen color
            for r in range(3):
                for c in range(3):
                    grid[block_r*3 + r, block_c*3 + c] = color
        
        # Add random cells with cell_color, ensuring they are not connected
        # Check if cells are connected in 8-way connectivity (diagonals included)
        random_cells_count = min(grid_size - 3, grid_size*grid_size)  # Number specified in reasoning chain
        
        # All possible positions in the grid
        all_positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        random.shuffle(all_positions)
        
        # Keep track of placed cell_color cells
        placed_cells = []
        
        # Try to place the required number of cell_color cells
        for r, c in all_positions:
            # Check if current position is adjacent to any already placed cell_color cell
            is_adjacent = False
            for pr, pc in placed_cells:
                # Check if cells are adjacent (including diagonals)
                if abs(r - pr) <= 1 and abs(c - pc) <= 1:
                    is_adjacent = True
                    break
            
            # If not adjacent, place the cell
            if not is_adjacent:
                grid[r, c] = cell_color
                placed_cells.append((r, c))
                
                # Stop when we've placed the required number of cells
                if len(placed_cells) >= random_cells_count:
                    break
        
        return grid
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        cell_color = taskvars['cell_color']
        
        # Calculate output grid size
        output_size = grid_size // 3
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # For each 3x3 block in the input grid
        for r in range(output_size):
            for c in range(output_size):
                # Extract the 3x3 block
                block = grid[r*3:(r+1)*3, c*3:(c+1)*3]
                
                # Determine if this is a valid colored block
                unique_colors = set(block.flatten())
                
                # Remove cell_color from consideration
                if cell_color in unique_colors:
                    unique_colors.remove(cell_color)
                
                # If there's exactly one remaining color and it's not 0 (empty)
                if len(unique_colors) == 1 and 0 not in unique_colors:
                    # Set the output cell to that color
                    output_grid[r, c] = list(unique_colors)[0]
        
        return output_grid


