from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry
import numpy as np
import random

class Taskd89b689bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of squared shape and can have different sizes.",
            "Each grid contains a 2×2 square in the central area, colored {color('square_color')}, and four single-colored cells, each with a distinct color.",
            "The four single-colored cells are located in the quadrants defined by the 2×2 square: one in the top-left quadrant, one in the top-right quadrant, one in the bottom-left quadrant, and one in the bottom-right quadrant relative to the 2×2 square.",
            "The four single-colored cells do not overlap the 2×2 square.",
            "All the remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The central square of color {color('square_color')} and all four single-colored cells are identified.",
            "Each cell of the central square is recolored to match the color of the single-colored cell located in the corresponding quadrant relative to the square, so that the top left cell of the square matches the single-colored cell in the top left quadrant, the top right cell matches the single-colored cell in the top right quadrant, the bottom left cell matches the single-colored cell in the bottom left quadrant, and the bottom right cell matches the single-colored cell in the bottom right quadrant.",
            "All four single-colored cells outside the central square are changed to empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        size = gridvars['size']
        grid = np.zeros((size, size), dtype=int)
        
        # Place 2x2 square in central area (not necessarily exact center)
        center = size // 2
        # Allow some offset from exact center
        max_offset = size // 4 - 1
        offset_r = random.randint(-max_offset, max_offset)
        offset_c = random.randint(-max_offset, max_offset)
        
        square_r = center - 1 + offset_r
        square_c = center - 1 + offset_c
        
        # Ensure the 2x2 square stays within bounds
        square_r = max(1, min(square_r, size - 3))
        square_c = max(1, min(square_c, size - 3))
        
        grid[square_r:square_r+2, square_c:square_c+2] = taskvars['square_color']
        
        # Store square position for later use
        gridvars['square_pos'] = (square_r, square_c)
        
        # Define quadrants relative to the 2x2 square
        # Top-left quadrant: rows < square_r, cols < square_c
        tl_positions = [(r, c) for r in range(square_r) for c in range(square_c)]
        if tl_positions:
            tl_pos = random.choice(tl_positions)
            grid[tl_pos] = gridvars['tl_color']
        
        # Top-right quadrant: rows < square_r, cols >= square_c + 2
        tr_positions = [(r, c) for r in range(square_r) for c in range(square_c + 2, size)]
        if tr_positions:
            tr_pos = random.choice(tr_positions)
            grid[tr_pos] = gridvars['tr_color']
        
        # Bottom-left quadrant: rows >= square_r + 2, cols < square_c
        bl_positions = [(r, c) for r in range(square_r + 2, size) for c in range(square_c)]
        if bl_positions:
            bl_pos = random.choice(bl_positions)
            grid[bl_pos] = gridvars['bl_color']
        
        # Bottom-right quadrant: rows >= square_r + 2, cols >= square_c + 2
        br_positions = [(r, c) for r in range(square_r + 2, size) for c in range(square_c + 2, size)]
        if br_positions:
            br_pos = random.choice(br_positions)
            grid[br_pos] = gridvars['br_color']
        
        return grid

    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        size = grid.shape[0]
        
        # Find the 2x2 square position
        square_r, square_c = None, None
        for r in range(size - 1):
            for c in range(size - 1):
                if (grid[r, c] == taskvars['square_color'] and 
                    grid[r+1, c] == taskvars['square_color'] and
                    grid[r, c+1] == taskvars['square_color'] and
                    grid[r+1, c+1] == taskvars['square_color']):
                    square_r, square_c = r, c
                    break
            if square_r is not None:
                break
        
        # Find the quadrant colors by scanning each quadrant relative to the square
        tl_color = None
        tr_color = None  
        bl_color = None
        br_color = None
        
        # Scan top-left quadrant (rows < square_r, cols < square_c)
        for r in range(square_r):
            for c in range(square_c):
                if grid[r, c] != 0 and grid[r, c] != taskvars['square_color']:
                    tl_color = grid[r, c]
                    output_grid[r, c] = 0  # Remove quadrant cell
                    break
            if tl_color is not None:
                break
                
        # Scan top-right quadrant (rows < square_r, cols >= square_c + 2)
        for r in range(square_r):
            for c in range(square_c + 2, size):
                if grid[r, c] != 0 and grid[r, c] != taskvars['square_color']:
                    tr_color = grid[r, c]
                    output_grid[r, c] = 0  # Remove quadrant cell
                    break
            if tr_color is not None:
                break
                
        # Scan bottom-left quadrant (rows >= square_r + 2, cols < square_c)
        for r in range(square_r + 2, size):
            for c in range(square_c):
                if grid[r, c] != 0 and grid[r, c] != taskvars['square_color']:
                    bl_color = grid[r, c]
                    output_grid[r, c] = 0  # Remove quadrant cell
                    break
            if bl_color is not None:
                break
                
        # Scan bottom-right quadrant (rows >= square_r + 2, cols >= square_c + 2)
        for r in range(square_r + 2, size):
            for c in range(square_c + 2, size):
                if grid[r, c] != 0 and grid[r, c] != taskvars['square_color']:
                    br_color = grid[r, c]
                    output_grid[r, c] = 0  # Remove quadrant cell
                    break
            if br_color is not None:
                break
        
        # Recolor center square cells
        if tl_color is not None:
            output_grid[square_r, square_c] = tl_color  # Top-left of square
        if tr_color is not None:
            output_grid[square_r, square_c+1] = tr_color    # Top-right of square
        if bl_color is not None:
            output_grid[square_r+1, square_c] = bl_color    # Bottom-left of square
        if br_color is not None:
            output_grid[square_r+1, square_c+1] = br_color      # Bottom-right of square
            
        return output_grid

    def create_grids(self):
        # Generate task variables
        available_colors = list(range(1, 10))
        square_color = random.choice(available_colors)
        
        taskvars = {
            'square_color': square_color
        }
        
        # Generate train and test examples
        num_train = random.randint(3, 6)
        
        def generate_example():
            # Generate grid-specific variables
            size = random.choice([i for i in range(8, 31) if i % 2 == 0])  # Even sizes, minimum 8 for offset space
            
            # Choose 4 distinct colors for quadrants (different from square color)
            quarter_colors = random.sample([c for c in available_colors if c != square_color], 4)
            
            gridvars = {
                'size': size,
                'tl_color': quarter_colors[0],
                'tr_color': quarter_colors[1], 
                'bl_color': quarter_colors[2],
                'br_color': quarter_colors[3]
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            return {
                'input': input_grid,
                'output': output_grid
            }
        
        train_examples = [generate_example() for _ in range(num_train)]
        test_examples = [generate_example()]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }