from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, Contiguity, random_cell_coloring
import numpy as np
import random


class Task1d0a4b61Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "Each input grid has a completely filled background of {color('background_color')}, with multi-colored rectangular tiles placed on top.",
            "Within the same grid, the size and pattern of each tile must be identical.",
            "The tiling should always start from position (1,1), leaving exactly one {color('background_color')} row and column as spacing between tiles.",
            "The tile size must be chosen so that the spacing is evenly distributed and the grid remains symmetric.",
            "The overall layout should form a tiled or periodic pattern, where the same design is repeated across the entire {vars['grid_size']} x {vars['grid_size']} grid.",
            "For example, in a 25 x 25 grid: A 5 x 5 tile size would produce 16 tiles. A 3 x 1 tile size would produce 42 tiles. A 2 x 2 tile size would produce 64 tiles.",
            "After the tiling is completed, remove several rectangular sections (sections must be randomly chosen) from the grid to create large empty regions filled with 0 (representing empty cells)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the rectangular tiles in the input grid from the non-empty areas.",
            "The background color is always {color('background_color')}.",
            "The tiles are always separated by exactly one row and one column of {color('background_color')} color.",
            "The tiles start at position (1,1) and are aligned both vertically and horizontally.",
            "The size and pattern of each tile within the same grid are identical.",
            "The output is then completed by filling in the empty sections, using the identified tile pattern, so that the final grid is complete, symmetric, and consists of tiles of the same size and pattern."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        background_color = taskvars['background_color']
        tile_size = gridvars.get('tile_size', None)
        tile_pattern = gridvars.get('tile_pattern', None)
        
        grid = np.full((grid_size, grid_size), background_color, dtype=int)
        
        if tile_size is None:
            possible_sizes = []
            for size in range(1, min(8, grid_size//3)):
                if (grid_size - 1) % (size + 1) == 0:
                    possible_sizes.append(size)
            if not possible_sizes:
                possible_sizes = [1, 2, 3]
            tile_size = random.choice(possible_sizes)
        
        if tile_pattern is None:
            # Sometimes tiles can be background_color (for test cases)
            if random.random() < 0.1:  # 10% chance of background_color tiles
                tile_pattern = np.full((tile_size, tile_size), background_color, dtype=int)
            else:
                colors = [c for c in range(1, 10) if c != background_color]
                tile_pattern = np.full((tile_size, tile_size), random.choice(colors), dtype=int)
                random_cell_coloring(tile_pattern, colors, density=0.5, background=random.choice(colors), overwrite=True)
        
        spacing = 1
        step = tile_size + spacing
        
        for i in range(spacing, grid_size - tile_size + 1, step):
            for j in range(spacing, grid_size - tile_size + 1, step):
                grid[i:i+tile_size, j:j+tile_size] = tile_pattern
        
        num_removals = random.randint(2, 4)
        for _ in range(num_removals):
            removal_width = random.randint(max(3, tile_size + spacing), min(grid_size//2, 3*(tile_size + spacing)))
            removal_height = random.randint(max(3, tile_size + spacing), min(grid_size//2, 3*(tile_size + spacing)))
            
            max_x = grid_size - removal_width
            max_y = grid_size - removal_height
            if max_x > 0 and max_y > 0:
                start_x = random.randint(0, max_x)
                start_y = random.randint(0, max_y)
                grid[start_x:start_x+removal_height, start_y:start_y+removal_width] = 0
        
        gridvars['tile_size'] = tile_size
        gridvars['tile_pattern'] = tile_pattern
        
        return grid
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        background_color = taskvars['background_color']
        
        output_grid = grid.copy()
        
        # First, identify the tile size by finding background_color separators
        # Tiles start at (1,1) and are separated by background_color rows/columns
        spacing = 1
        detected_tile_size = None
        
        # Look for the pattern of background_color rows/columns
        # Check for 1x1 tiles first, then larger tiles
        for tile_size in range(1, min(8, grid_size//3)):
            step = tile_size + spacing
            # Check if this tile size fits the grid pattern
            expected_separators = []
            for pos in range(0, grid_size, step):
                expected_separators.append(pos)
            
            # Verify this is the correct tile size by checking separators
            valid_size = True
            for sep_pos in expected_separators:
                if sep_pos < grid_size:
                    # Check if row is mostly background_color (allowing for missing sections)
                    row = grid[sep_pos, :]
                    non_zero_non_bg = np.sum((row != 0) & (row != background_color))
                    if non_zero_non_bg > tile_size:  # Too many non-background colors
                        valid_size = False
                        break
            
            if valid_size:
                detected_tile_size = tile_size
                break
        
        if detected_tile_size is None:
            detected_tile_size = 1  # Default fallback
        
        # Now find the reference tile pattern from complete tiles
        step = detected_tile_size + spacing
        reference_tile = None
        
        # Look for complete tiles (no empty cells) starting from position (1,1)
        for i in range(spacing, grid_size - detected_tile_size + 1, step):
            for j in range(spacing, grid_size - detected_tile_size + 1, step):
                if i + detected_tile_size <= grid_size and j + detected_tile_size <= grid_size:
                    tile_region = grid[i:i+detected_tile_size, j:j+detected_tile_size]
                    # A complete tile has no empty (0) cells
                    # Note: tiles can be background_color, that's valid
                    if np.all(tile_region != 0):
                        reference_tile = tile_region.copy()
                        break
            if reference_tile is not None:
                break
        
        # If no complete tile found, try to reconstruct from partial tiles
        if reference_tile is None:
            reference_tile = np.full((detected_tile_size, detected_tile_size), background_color, dtype=int)
            # Try to find partial patterns and combine them
            for i in range(spacing, grid_size - detected_tile_size + 1, step):
                for j in range(spacing, grid_size - detected_tile_size + 1, step):
                    if i + detected_tile_size <= grid_size and j + detected_tile_size <= grid_size:
                        tile_region = grid[i:i+detected_tile_size, j:j+detected_tile_size]
                        # Copy non-empty cells to build reference
                        mask = tile_region != 0
                        reference_tile[mask] = tile_region[mask]
        
        # Fill in all tile positions with the reference tile
        for i in range(spacing, grid_size - detected_tile_size + 1, step):
            for j in range(spacing, grid_size - detected_tile_size + 1, step):
                if i + detected_tile_size <= grid_size and j + detected_tile_size <= grid_size:
                    output_grid[i:i+detected_tile_size, j:j+detected_tile_size] = reference_tile
        
        # Fill any remaining empty (0) cells with background_color
        output_grid[output_grid == 0] = background_color
        
        return output_grid
    
    def create_grids(self):
        taskvars = {
            'grid_size': random.randint(15, 30),
            'background_color': random.randint(1, 9)
        }
        
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            gridvars = {}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        test_gridvars = {}
        test_input = self.create_input(taskvars, test_gridvars)
        
        test_tile_pattern = test_gridvars['tile_pattern']
        colors = [c for c in range(1, 10) if c != taskvars['background_color']]
        background_color = taskvars['background_color']
        
        for i in range(test_tile_pattern.shape[0]):
            if random.random() < 0.3:
                test_tile_pattern[i, :] = background_color
        for j in range(test_tile_pattern.shape[1]):
            if random.random() < 0.3:
                test_tile_pattern[:, j] = background_color
        
        test_gridvars['tile_pattern'] = test_tile_pattern
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_pairs = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_pairs,
            'test': test_pairs
        }


