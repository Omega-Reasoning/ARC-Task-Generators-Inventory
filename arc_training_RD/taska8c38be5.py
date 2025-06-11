from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, GridObjects, find_connected_objects
from input_library import create_object, retry, random_cell_coloring
import numpy as np
import random

class Taska8c38be5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "There are nine 3x3 sub-grids in total.",
            "Among them: Four sub-grids each contain a single L-shaped object, rotated differently by 90 degrees (i.e., no two L-shapes share the same orientation).",
            "Four sub-grids each contain a single T-shaped [3 cells head and 1 cell tail] object, also rotated uniquely by 90 degrees (with no repeated rotation).",
            "One sub-grid is completely filled with the sub-grid background color (entire 3x3 area).",
            "All sub-grids share the same background color.",
            "Each object within the sub-grids must have a unique color — no color is reused across objects.",
            "The sub-grids are placed randomly within the larger grid with spacing between them to avoid adjacency.",
            "The main grid uses a different background color from the sub-grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is a 9x9 grid.",
            "All empty cells in the output are filled with the sub-grid background color.",
            "The four L-shaped objects from input are placed at the four corners of the 9x9 grid, maintaining their exact colors.",
            "The four T-shaped objects from input are placed at the middle of each outer edge of the 9x9 grid, maintaining their exact colors.",
            "Each T-shape occupies 4 cells (3 cells head + 1 cell tail) with the tail pointing inward toward the center.",
            "Together, L-shapes and T-shapes form a complete square border where each edge has 7 out of 9 cells filled.",
            "The filled sub-grid (entire 3x3 area with sub-grid background color) is placed at the exact center of the 9x9 grid."
        ]
        
        taskvars_definitions = {}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
        
        self.task_sub_bg = None

    def create_l_shape(self, rotation=0):
        base = np.array([
            [1, 1, 0],
            [1, 0, 0], 
            [0, 0, 0]
        ])
        return np.rot90(base, k=rotation)

    def create_t_shape(self, rotation=0):
        base = np.array([
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 0]
        ])
        return np.rot90(base, k=rotation)

    def get_shape_signature(self, obj_array):
        """Create a unique signature for the shape pattern"""
        if obj_array.size == 0:
            return None
        
        # Trim empty borders
        rows_with_data = np.any(obj_array != 0, axis=1)
        cols_with_data = np.any(obj_array != 0, axis=0)
        if not np.any(rows_with_data) or not np.any(cols_with_data):
            return None
            
        trimmed = obj_array[rows_with_data][:, cols_with_data]
        
        # Get relative coordinates of non-zero cells
        coords = []
        for r in range(trimmed.shape[0]):
            for c in range(trimmed.shape[1]):
                if trimmed[r, c] != 0:
                    coords.append((r, c))
        
        # Normalize to start from (0,0)
        if not coords:
            return None
            
        min_r = min(r for r, c in coords)
        min_c = min(c for r, c in coords)
        normalized_coords = tuple(sorted([(r - min_r, c - min_c) for r, c in coords]))
        
        return normalized_coords

    def create_input(self):
        sub_bg = self.task_sub_bg
        grid_size = random.randint(14, 20)
        main_bg = 0
        
        available_colors = [c for c in range(1, 10) if c != sub_bg]
        random.shuffle(available_colors)
        
        grid = np.full((grid_size, grid_size), main_bg, dtype=int)
        
        # Create reference patterns for all rotations with CORRECTED mappings
        self.l_signatures = {}  # signature -> output_position
        self.t_signatures = {}  # signature -> output_position
        
        # CORRECTED L-shape mappings
        l_positions = [
            'top-left',      # rotation 0
            'bottom-left',   # rotation 1 (CORRECTED: was top-right)
            'bottom-right',  # rotation 2
            'top-right'      # rotation 3 (CORRECTED: was bottom-left)
        ]
        for rotation in range(4):
            shape = self.create_l_shape(rotation)
            signature = self.get_shape_signature(shape)
            if signature:
                self.l_signatures[signature] = l_positions[rotation]
        
        # CORRECTED T-shape mappings
        t_positions = [
            'top',     # rotation 0
            'left',    # rotation 1 (CORRECTED: was right)
            'bottom',  # rotation 2
            'right'    # rotation 3 (CORRECTED: was left)
        ]
        for rotation in range(4):
            shape = self.create_t_shape(rotation)
            signature = self.get_shape_signature(shape)
            if signature:
                self.t_signatures[signature] = t_positions[rotation]
        
        sub_grids = []
        
        # Create L-shapes
        for rotation in range(4):
            shape = self.create_l_shape(rotation)
            sub_grid = np.full((3, 3), sub_bg, dtype=int)
            color = available_colors[rotation]
            sub_grid[shape == 1] = color
            sub_grids.append(sub_grid)
            
        # Create T-shapes
        for rotation in range(4):
            shape = self.create_t_shape(rotation)
            sub_grid = np.full((3, 3), sub_bg, dtype=int)
            color = available_colors[4 + rotation]
            sub_grid[shape == 1] = color
            sub_grids.append(sub_grid)
            
        # Filled sub-grid
        filled_sub_grid = np.full((3, 3), sub_bg, dtype=int)
        sub_grids.append(filled_sub_grid)
        
        def try_place_subgrids():
            test_grid = np.full((grid_size, grid_size), main_bg, dtype=int)
            placed_positions = []
            
            for sub_grid in sub_grids:
                attempts = 0
                placed = False
                while attempts < 200 and not placed:
                    row = random.randint(0, grid_size - 3)
                    col = random.randint(0, grid_size - 3)
                    
                    valid_position = True
                    
                    if not np.all(test_grid[row:row+3, col:col+3] == main_bg):
                        valid_position = False
                    
                    if valid_position:
                        for prev_row, prev_col in placed_positions:
                            row_distance = max(0, max(row - (prev_row + 3), prev_row - (row + 3)))
                            col_distance = max(0, max(col - (prev_col + 3), prev_col - (col + 3)))
                            
                            if row_distance == 0 and col_distance == 0:
                                valid_position = False
                                break
                    
                    if valid_position:
                        test_grid[row:row+3, col:col+3] = sub_grid
                        placed_positions.append((row, col))
                        placed = True
                    
                    attempts += 1
                
                if not placed:
                    return None
                    
            return test_grid
        
        grid = retry(try_place_subgrids, lambda x: x is not None)
        return grid

    def transform_input(self, input_grid):
        sub_bg = self.task_sub_bg
        main_bg = 0
        
        output = np.full((9, 9), sub_bg, dtype=int)
        
        # Find all objects in input
        objects = find_connected_objects(input_grid, diagonal_connectivity=False, background=main_bg, monochromatic=True)
        
        for obj in objects:
            # Skip sub_bg colored objects
            if sub_bg in obj.colors:
                continue
                
            color = list(obj.colors)[0]
            obj_array = obj.to_array()
            signature = self.get_shape_signature(obj_array)
            
            if signature is None:
                continue
            
            # Check if it's an L-shape
            if signature in self.l_signatures:
                position = self.l_signatures[signature]
                
                if position == 'top-left':
                    positions = [(0,0), (0,1), (1,0)]
                elif position == 'top-right':
                    positions = [(0,7), (0,8), (1,8)]
                elif position == 'bottom-right':
                    positions = [(7,8), (8,7), (8,8)]
                elif position == 'bottom-left':
                    positions = [(7,0), (8,0), (8,1)]
                else:
                    continue
                    
                for r, c in positions:
                    output[r, c] = color
            
            # Check if it's a T-shape
            elif signature in self.t_signatures:
                position = self.t_signatures[signature]
                
                if position == 'top':
                    positions = [(0,3), (0,4), (0,5), (1,4)]
                elif position == 'right':
                    positions = [(3,8), (4,8), (5,8), (4,7)]
                elif position == 'bottom':
                    positions = [(8,3), (8,4), (8,5), (7,4)]
                elif position == 'left':
                    positions = [(3,0), (4,0), (5,0), (4,1)]
                else:
                    continue
                    
                for r, c in positions:
                    output[r, c] = color
        
        return output

    def create_grids(self):
        gridvars = {}
        # Set the consistent sub-grid background color for this entire task
        self.task_sub_bg = random.randint(1, 9)
        # Generate training pairs
        train_pairs = []
        num_train = random.randint(3, 5)
        
        for _ in range(num_train):
            input_grid = self.create_input()
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # Generate test pair
        test_input = self.create_input()
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)

# Test the generator
