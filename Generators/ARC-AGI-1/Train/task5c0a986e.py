from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task5c0a986eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain one {color('block1')} and one {color('block2')} 2x2 square block.",
            "The {color('block1')} and {color('block2')} blocks are positioned such that they never touch any of the edges of the grid and are never 4-way connected to each other.",
            "The {color('block1')} block should never be placed along the diagonal path extending from the bottom-right corner cell of the {color('block2')} block to the grid edge.",
            "The {color('block2')} block should never be placed along the diagonal path extending from the top-left corner cell of the {color('block1')} block to the grid edge.",
            "The positions of the blocks vary across examples while avoiding the restricted areas.",
            "All other cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They are constructed by copying the input grids and extending the {color('block1')} and {color('block2')} blocks diagonally from their respective corners to form two paths.",
            "The {color('block1')} block is extended from its top-left corner diagonally up-left until it reaches the grid edge.",
            "The {color('block2')} block is extended from its bottom-right corner diagonally down-right until it reaches the grid edge.",
            "The colors used for creating each path match the respective color of the original block."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        
        taskvars = {}
        
        # Randomly choose grid size
        taskvars['grid_size'] = random.randint(9, 30)  # Increased minimum size to 9
        
        # Randomly choose block colors, ensuring they are different
        block1_color = random.randint(1, 9)
        block2_color = random.choice([c for c in range(1, 10) if c != block1_color])
        taskvars['block1'] = block1_color
        taskvars['block2'] = block2_color
        
        # Generate 3-4 train examples and 1 test example
        num_train_examples = random.randint(3, 4)
        train_data = []
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        block1_color = taskvars['block1']
        block2_color = taskvars['block2']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Check if blocks overlap or are 4-way connected
        def blocks_overlap_or_connect(r1, c1, r2, c2):
            # Check for any overlap
            for dr1 in range(2):
                for dc1 in range(2):
                    for dr2 in range(2):
                        for dc2 in range(2):
                            if r1+dr1 == r2+dr2 and c1+dc1 == c2+dc2:
                                return True
            
            # Check for 4-way connectivity
            for dr1 in range(2):
                for dc1 in range(2):
                    for dr2 in range(2):
                        for dc2 in range(2):
                            # Check if cells are adjacent (4-way)
                            row1, col1 = r1+dr1, c1+dc1
                            row2, col2 = r2+dr2, c2+dc2
                            if ((abs(row1-row2) == 1 and col1 == col2) or 
                                (row1 == row2 and abs(col1-col2) == 1)):
                                return True
            return False
        
        # Check if a position is on a diagonal path from a corner
        def is_on_diagonal_path(start_r, start_c, check_r, check_c, direction):
            r, c = start_r, start_c
            if direction == "up-left":
                while r > 0 and c > 0:
                    r -= 1
                    c -= 1
                    if r == check_r and c == check_c:
                        return True
            elif direction == "down-right":
                while r < grid_size-1 and c < grid_size-1:
                    r += 1
                    c += 1
                    if r == check_r and c == check_c:
                        return True
            return False
        
        # Check if a block position is valid
        def is_valid_block_position(r1, c1, r2, c2):
            # Check if blocks touch edges
            if (r1 <= 0 or r1+1 >= grid_size-1 or c1 <= 0 or c1+1 >= grid_size-1 or 
                r2 <= 0 or r2+1 >= grid_size-1 or c2 <= 0 or c2+1 >= grid_size-1):
                return False
            
            # Check overlap or 4-way connectivity
            if blocks_overlap_or_connect(r1, c1, r2, c2):
                return False
            
            # Check if block1 is on diagonal path from bottom-right of block2
            br_r, br_c = r2+1, c2+1  # Bottom-right of block2
            for dr1 in range(2):
                for dc1 in range(2):
                    if is_on_diagonal_path(br_r, br_c, r1+dr1, c1+dc1, "up-left"):
                        return False
            
            # Check if block2 is on diagonal path from top-left of block1
            tl_r, tl_c = r1, c1  # Top-left of block1
            for dr2 in range(2):
                for dc2 in range(2):
                    if is_on_diagonal_path(tl_r, tl_c, r2+dr2, c2+dc2, "down-right"):
                        return False
            
            return True
        
        # Try to place blocks with constraints
        max_attempts = 1000
        valid_placement = False
        
        for _ in range(max_attempts):
            # Reset grid
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Generate positions ensuring blocks are separated
            # For better separation in smaller grids:
            r1 = random.randint(1, grid_size//2 - 1)
            c1 = random.randint(1, grid_size//2 - 1)
            
            r2 = random.randint(grid_size//2, grid_size - 3)
            c2 = random.randint(grid_size//2, grid_size - 3)
            
            # Check if placements are valid
            if is_valid_block_position(r1, c1, r2, c2):
                # Place block1
                grid[r1:r1+2, c1:c1+2] = block1_color
                
                # Place block2
                grid[r2:r2+2, c2:c2+2] = block2_color
                
                valid_placement = True
                break
        
        if not valid_placement:
            # Fallback positioning - these should be carefully selected to still follow rules
            grid = np.zeros((grid_size, grid_size), dtype=int)
            # Place blocks at opposing corners with safe distance
            grid[1:3, 1:3] = block1_color
            grid[grid_size-3:grid_size-1, grid_size-3:grid_size-1] = block2_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        grid_size = taskvars['grid_size']
        block1_color = taskvars['block1'] 
        block2_color = taskvars['block2']
        
        # Find block1 (by finding the top-left most occurrence of block1_color)
        block1_coords = np.argwhere(grid == block1_color)
        if len(block1_coords) > 0:
            # Sort to ensure we get the top-left corner
            block1_coords = sorted(block1_coords, key=lambda x: (x[0], x[1]))
            tl_r, tl_c = block1_coords[0]
            
            # Extend block1 diagonally up-left from its top-left corner
            r, c = tl_r, tl_c
            while r > 0 and c > 0:
                r -= 1
                c -= 1
                output_grid[r, c] = block1_color
        
        # Find block2 (by finding the bottom-right most occurrence of block2_color)
        block2_coords = np.argwhere(grid == block2_color)
        if len(block2_coords) > 0:
            # Sort to ensure we get the bottom-right corner
            block2_coords = sorted(block2_coords, key=lambda x: (x[0], x[1]), reverse=True)
            br_r, br_c = block2_coords[0]
            
            # Extend block2 diagonally down-right from its bottom-right corner
            r, c = br_r, br_c
            while r < grid_size - 1 and c < grid_size - 1:
                r += 1
                c += 1
                output_grid[r, c] = block2_color
        
        return output_grid

