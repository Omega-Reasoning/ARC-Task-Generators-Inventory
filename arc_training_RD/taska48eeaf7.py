from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random

class BlockAttachmentTaskGenerator(ARCTaskGenerator):
    
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size NxN.",
            "The grid contains one 2x2 square object of {color('block_color')} color, and this block can be placed anywhere in the grid.",
            "The main block 2x2 square is covered with two types of cells namely empty (0) cells and scattered cells (<5), where the scattered cells are of {color('object_color')} color.",
            "These scattered cells are placed such that they are either horizontal or vertical or diagonal to the 2x2 main block square cells, but NOT touching the main block or each other."
        ]
        
        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid main 2x2 square block.",
            "And the scattered cells around the main block, are attached like tails to the main block with respect to their position. Suppose if one of the scattered cells is horizontally placed across the main block, then it gets attached besides the main block horizontally. The same rule is applied for cells being vertical and diagonally placed.",
            "The attachment should not disturb the main block. The scattered cells must occur just as an extension of the main block in the output grid.",
            "Also, the scattered cell alone must be attached to the main block, it must not form like a series of attachments."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # Generate random colors ensuring they're different
        block_color = random.randint(1, 9)
        object_color = random.choice([c for c in range(1, 10) if c != block_color])
        
        # Store task variables
        taskvars = {
            'block_color': block_color,
            'object_color': object_color,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate grid sizes
        min_size = 5
        max_size = 20
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': all_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': all_sizes[-1]}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        """Create input grid with 2x2 block and scattered cells."""
        block_color = taskvars['block_color']
        object_color = taskvars['object_color']
        grid_size = gridvars['grid_size']
        
        def generate_valid_grid():
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Place 2x2 block randomly
            max_block_row = grid_size - 2
            max_block_col = grid_size - 2
            block_row = random.randint(0, max_block_row)
            block_col = random.randint(0, max_block_col)
            
            # Create 2x2 block
            for r in range(block_row, block_row + 2):
                for c in range(block_col, block_col + 2):
                    grid[r, c] = block_color
            
            # Get block cells for reference
            block_cells = {(r, c) for r in range(block_row, block_row + 2) 
                          for c in range(block_col, block_col + 2)}
            
            # Place 1-4 scattered cells
            num_scattered = random.randint(1, 4)
            scattered_positions = []
            
            for _ in range(num_scattered):
                attempts = 0
                while attempts < 100:
                    # Choose direction: horizontal, vertical, or diagonal
                    direction = random.choice(['horizontal', 'vertical', 'diagonal'])
                    
                    if direction == 'horizontal':
                        # Left or right of block
                        side = random.choice(['left', 'right'])
                        if side == 'left' and block_col > 0:
                            scatter_row = random.randint(block_row, block_row + 1)
                            scatter_col = random.randint(0, block_col - 1)
                        elif side == 'right' and block_col + 2 < grid_size:
                            scatter_row = random.randint(block_row, block_row + 1)
                            scatter_col = random.randint(block_col + 2, grid_size - 1)
                        else:
                            attempts += 1
                            continue
                    
                    elif direction == 'vertical':
                        # Above or below block
                        side = random.choice(['above', 'below'])
                        if side == 'above' and block_row > 0:
                            scatter_row = random.randint(0, block_row - 1)
                            scatter_col = random.randint(block_col, block_col + 1)
                        elif side == 'below' and block_row + 2 < grid_size:
                            scatter_row = random.randint(block_row + 2, grid_size - 1)
                            scatter_col = random.randint(block_col, block_col + 1)
                        else:
                            attempts += 1
                            continue
                    
                    else:  # diagonal
                        # Choose diagonal direction
                        diag_dir = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
                        if diag_dir == 'top-left' and block_row > 0 and block_col > 0:
                            scatter_row = random.randint(0, block_row - 1)
                            scatter_col = random.randint(0, block_col - 1)
                        elif diag_dir == 'top-right' and block_row > 0 and block_col + 2 < grid_size:
                            scatter_row = random.randint(0, block_row - 1)
                            scatter_col = random.randint(block_col + 2, grid_size - 1)
                        elif diag_dir == 'bottom-left' and block_row + 2 < grid_size and block_col > 0:
                            scatter_row = random.randint(block_row + 2, grid_size - 1)
                            scatter_col = random.randint(0, block_col - 1)
                        elif diag_dir == 'bottom-right' and block_row + 2 < grid_size and block_col + 2 < grid_size:
                            scatter_row = random.randint(block_row + 2, grid_size - 1)
                            scatter_col = random.randint(block_col + 2, grid_size - 1)
                        else:
                            attempts += 1
                            continue
                    
                    # Check if position is valid (not touching block or other scattered cells)
                    pos = (scatter_row, scatter_col)
                    if pos in block_cells or pos in scattered_positions:
                        attempts += 1
                        continue
                    
                    # Check not touching block or other scattered cells
                    valid = True
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            check_pos = (scatter_row + dr, scatter_col + dc)
                            if check_pos in block_cells or check_pos in scattered_positions:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        scattered_positions.append(pos)
                        grid[scatter_row, scatter_col] = object_color
                        break
                    
                    attempts += 1
            
            return grid, len(scattered_positions) > 0
        
        grid, has_scattered = retry(
            generate_valid_grid,
            lambda x: x[1],
            max_attempts=100
        )
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        """Transform by attaching scattered cells to the 2x2 block."""
        output_grid = grid.copy()
        block_color = taskvars['block_color']
        object_color = taskvars['object_color']
        
        # Find the 2x2 block
        block_cells = set()
        rows, cols = grid.shape
        
        for r in range(rows - 1):
            for c in range(cols - 1):
                if (grid[r, c] == block_color and grid[r, c+1] == block_color and
                    grid[r+1, c] == block_color and grid[r+1, c+1] == block_color):
                    block_cells = {(r, c), (r, c+1), (r+1, c), (r+1, c+1)}
                    break
            if block_cells:
                break
        
        # Find scattered cells
        scattered_cells = []
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == object_color:
                    scattered_cells.append((r, c))
        
        # Remove all scattered cells from their original positions first
        for scatter_r, scatter_c in scattered_cells:
            output_grid[scatter_r, scatter_c] = 0
        
        # For each scattered cell, attach it to the block
        for scatter_r, scatter_c in scattered_cells:
            # Find the closest block edge cell
            min_dist = float('inf')
            closest_block_cell = None
            
            for block_r, block_c in block_cells:
                dist = abs(scatter_r - block_r) + abs(scatter_c - block_c)
                if dist < min_dist:
                    min_dist = dist
                    closest_block_cell = (block_r, block_c)
            
            if closest_block_cell:
                block_r, block_c = closest_block_cell
                
                # Determine direction from block to scattered cell
                dr = 0 if scatter_r == block_r else (1 if scatter_r > block_r else -1)
                dc = 0 if scatter_c == block_c else (1 if scatter_c > block_c else -1)
                
                # Attach by placing the scattered cell adjacent to the block
                attach_r = block_r + dr
                attach_c = block_c + dc
                
                # Make sure attachment position is valid and not part of the block
                if (0 <= attach_r < rows and 0 <= attach_c < cols and 
                    (attach_r, attach_c) not in block_cells):
                    output_grid[attach_r, attach_c] = object_color
        
        return output_grid