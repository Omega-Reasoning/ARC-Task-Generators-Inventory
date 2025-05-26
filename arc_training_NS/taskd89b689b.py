from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import retry
import numpy as np
import random

class Taskd89b689b(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of squared shape and can have different sizes.",
            "Each grid contains a 2×2 square in the center, colored {color('square_color')}, and four single-colored cells, each with a distinct color, located respectively in the top-left, top-right, bottom-left, and bottom-right quarters of the grid.",
            "The four single-colored cells do not overlap the square in the center.",
            "All the remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The central square of color {color('square_color')} and all four single-colored cells are identified.",
            "Each cell of the central square is recolored to match the color of the single-colored cell located in the corresponding quarter of the grid, so that the top left cell matches the single-colored cell in the top left quarter, the top right cell matches the single-colored cell in the top right quarter, the bottom left cell matches the single-colored cell in the bottom left quarter, and the bottom right cell matches the single-colored cell in the bottom right quarter.",
            "All four single-colored cells outside the central square are changed to empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        size = gridvars['size']
        grid = np.zeros((size, size), dtype=int)
        
        # Place 2x2 center square
        center = size // 2
        grid[center-1:center+1, center-1:center+1] = taskvars['square_color']
        
        # Define quarters (avoiding center square)
        half = size // 2
        
        # Top-left quarter
        tl_positions = [(r, c) for r in range(half-1) for c in range(half-1)]
        tl_pos = random.choice(tl_positions)
        grid[tl_pos] = gridvars['tl_color']
        
        # Top-right quarter  
        tr_positions = [(r, c) for r in range(half-1) for c in range(half+1, size)]
        tr_pos = random.choice(tr_positions)
        grid[tr_pos] = gridvars['tr_color']
        
        # Bottom-left quarter
        bl_positions = [(r, c) for r in range(half+1, size) for c in range(half-1)]
        bl_pos = random.choice(bl_positions)
        grid[bl_pos] = gridvars['bl_color']
        
        # Bottom-right quarter
        br_positions = [(r, c) for r in range(half+1, size) for c in range(half-1, size)]
        br_pos = random.choice(br_positions)
        grid[br_pos] = gridvars['br_color']
        
        return grid

    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        size = grid.shape[0]
        center = size // 2
        half = size // 2
        
        # Find the quarter colors by scanning each quarter
        tl_color = None
        tr_color = None  
        bl_color = None
        br_color = None
        
        # Scan top-left quarter
        for r in range(half-1):
            for c in range(half-1):
                if grid[r, c] != 0 and grid[r, c] != taskvars['square_color']:
                    tl_color = grid[r, c]
                    output_grid[r, c] = 0  # Remove quarter cell
                    break
            if tl_color is not None:
                break
                
        # Scan top-right quarter
        for r in range(half-1):
            for c in range(half+1, size):
                if grid[r, c] != 0 and grid[r, c] != taskvars['square_color']:
                    tr_color = grid[r, c]
                    output_grid[r, c] = 0  # Remove quarter cell
                    break
            if tr_color is not None:
                break
                
        # Scan bottom-left quarter
        for r in range(half+1, size):
            for c in range(half-1):
                if grid[r, c] != 0 and grid[r, c] != taskvars['square_color']:
                    bl_color = grid[r, c]
                    output_grid[r, c] = 0  # Remove quarter cell
                    break
            if bl_color is not None:
                break
                
        # Scan bottom-right quarter
        for r in range(half+1, size):
            for c in range(half+1, size):
                if grid[r, c] != 0 and grid[r, c] != taskvars['square_color']:
                    br_color = grid[r, c]
                    output_grid[r, c] = 0  # Remove quarter cell
                    break
            if br_color is not None:
                break
        
        # Recolor center square cells
        if tl_color is not None:
            output_grid[center-1, center-1] = tl_color  # Top-left of center
        if tr_color is not None:
            output_grid[center-1, center] = tr_color    # Top-right of center
        if bl_color is not None:
            output_grid[center, center-1] = bl_color    # Bottom-left of center
        if br_color is not None:
            output_grid[center, center] = br_color      # Bottom-right of center
            
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
            size = random.choice([i for i in range(6, 31) if i % 2 == 0])  # Even sizes only
            
            # Choose 4 distinct colors for quarters (different from square color)
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

