from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random
from scipy.ndimage import binary_fill_holes

class ClosedShapeDetector(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Grids contain a mixture of: Irregular open patterns (shapes that are not closed), Irregular closed shapes (enclosed areas), Single cells, lines, or very small patterns.",
            "All shapes and patterns are uniformly colored with {color('input_color')} color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is formed by copying the input grid.",
            "Detect all closed shapes — i.e., patterns that form an enclosed loop or area.",
            "Change the color of these closed shapes to {color('close_color')} color.",
            "Leave all other patterns (lines, open shapes, scattered cells) in their original {color('input_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def color_name(self, color: int) -> str:
        color_map = {
            0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
            5: "gray", 6: "magenta", 7: "orange", 8: "cyan", 9: "brown"
        }
        return color_map.get(color, f"color_{color}")
    
    def create_grids(self):
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        input_color = available_colors[0]
        close_color = available_colors[1]
        
        taskvars = {
            "input_color": input_color,
            "close_color": close_color
        }
        
        self.taskvars = taskvars
        
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"
        
        self.input_reasoning_chain = [
            chain.replace("{color('input_color')}", color_fmt('input_color'))
                 .replace("{color('close_color')}", color_fmt('close_color'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('input_color')}", color_fmt('input_color'))
                 .replace("{color('close_color')}", color_fmt('close_color'))
            for chain in self.transformation_reasoning_chain
        ]
        
        num_train_pairs = random.randint(3, 5)
        train_pairs = [self.create_example(taskvars) for _ in range(num_train_pairs)]
        test_pairs = [self.create_example(taskvars)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
    
    def create_example(self, taskvars):
        input_grid = self.create_input(taskvars)
        output_grid = self.transform_input(input_grid.copy(), taskvars)
        return GridPair(input=input_grid, output=output_grid)
    
    def create_input(self, taskvars):
        grid_height = random.randint(15, 20)
        grid_width = random.randint(15, 20)
        
        grid = np.zeros((grid_height, grid_width), dtype=np.int32)
        input_color = taskvars['input_color']
        
        # 6 shapes total:
        # 1. One line (3 cells) - OPEN
        # 2. One single cell - OPEN  
        # 3. At least 2 closed shapes - CLOSED
        # 4. Rest are OPEN shapes
        
        # Create 1 line (OPEN)
        line = self._create_line(input_color)
        self._place_pattern(grid, line, grid_height, grid_width)
        
        # Create 1 single cell (OPEN)
        single = self._create_single_cell(input_color)
        self._place_pattern(grid, single, grid_height, grid_width)
        
        # Create exactly 2 closed shapes (CLOSED)
        for _ in range(2):
            closed_shape = self._create_closed_shape(input_color)
            self._place_pattern(grid, closed_shape, grid_height, grid_width)
        
        # Create 2 more OPEN shapes
        for _ in range(2):
            open_shape = self._create_open_shape(input_color)
            self._place_pattern(grid, open_shape, grid_height, grid_width)
        
        return grid
    
    def _create_line(self, color):
        if random.choice([True, False]):
            pattern = np.zeros((1, 3), dtype=np.int32)
            pattern[0, :] = color
        else:
            pattern = np.zeros((3, 1), dtype=np.int32)
            pattern[:, 0] = color
        return pattern
    
    def _create_single_cell(self, color):
        pattern = np.zeros((1, 1), dtype=np.int32)
        pattern[0, 0] = color
        return pattern
    
    def _create_closed_shape(self, color):
        size = random.randint(4, 6)
        pattern = np.zeros((size, size), dtype=np.int32)
        
        # Create hollow shape that's definitely closed
        pattern[0, :] = color  # Top
        pattern[-1, :] = color  # Bottom
        pattern[:, 0] = color  # Left
        pattern[:, -1] = color  # Right
        
        # Add some irregularity while keeping it closed
        for _ in range(random.randint(1, 2)):
            side = random.choice(['top', 'bottom', 'left', 'right'])
            pos = random.randint(1, size-2)
            if side == 'top':
                pattern[1, pos] = color
            elif side == 'bottom':
                pattern[-2, pos] = color
            elif side == 'left':
                pattern[pos, 1] = color
            elif side == 'right':
                pattern[pos, -2] = color
        
        return pattern
    
    def _create_open_shape(self, color):
        size = random.randint(3, 5)
        pattern = np.zeros((size, size), dtype=np.int32)
        
        # Create different types of OPEN shapes
        shape_type = random.choice(['L_shape', 'T_shape', 'arc', 'branch', 'random_walk'])
        
        if shape_type == 'L_shape':
            # L-shaped pattern (clearly open)
            pattern[:, 0] = color  # Vertical part
            pattern[-1, :size//2+1] = color  # Horizontal part
        
        elif shape_type == 'T_shape':
            # T-shaped pattern (clearly open)
            center = size // 2
            pattern[0, :] = color  # Top horizontal
            pattern[:center+1, center] = color  # Vertical part
        
        elif shape_type == 'arc':
            # Arc pattern (incomplete circle)
            center = size // 2
            for r in range(size):
                for c in range(size):
                    dist = ((r - center) ** 2 + (c - center) ** 2) ** 0.5
                    angle = np.arctan2(r - center, c - center)
                    if abs(dist - center + 0.5) < 0.8 and -np.pi/2 < angle < np.pi/2:
                        pattern[r, c] = color
        
        elif shape_type == 'branch':
            # Branching pattern
            center = size // 2
            pattern[:, center] = color  # Main line
            if size >= 3:
                pattern[center, :center+1] = color  # Branch
        
        else:  # random_walk
            # Random walk (always open)
            r, c = size // 2, size // 2
            pattern[r, c] = color
            
            num_cells = random.randint(3, size * size // 2)
            for _ in range(num_cells - 1):
                dr, dc = random.choice([(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)])
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    pattern[nr, nc] = color
                    r, c = nr, nc
        
        # Double check it's not accidentally closed
        if self._is_closed_shape(pattern):
            # Force it to be open by removing a cell
            cells = np.where(pattern == color)
            if len(cells[0]) > 2:
                idx = random.randint(0, len(cells[0]) - 1)
                pattern[cells[0][idx], cells[1][idx]] = 0
        
        return pattern
    
    def _place_pattern(self, grid, pattern, grid_height, grid_width):
        ph, pw = pattern.shape
        spacing = 2
        
        for _ in range(50):
            r = random.randint(spacing, grid_height - ph - spacing)
            c = random.randint(spacing, grid_width - pw - spacing)
            
            if np.all(grid[r-spacing:r+ph+spacing, c-spacing:c+pw+spacing] == 0):
                grid[r:r+ph, c:c+pw] += pattern
                return True
        return False
    
    def _is_closed_shape(self, obj_array):
        if obj_array.size <= 4:
            return False
        
        non_zero_rows = np.any(obj_array != 0, axis=1)
        non_zero_cols = np.any(obj_array != 0, axis=0)
        
        if not np.any(non_zero_rows) or not np.any(non_zero_cols):
            return False
        
        trimmed = obj_array[non_zero_rows][:, non_zero_cols]
        h, w = trimmed.shape
        
        if h < 3 or w < 3:
            return False
        
        binary_mask = (trimmed != 0)
        filled = binary_fill_holes(binary_mask)
        holes_filled = np.sum(filled) - np.sum(binary_mask)
        
        return holes_filled > 0
    
    def transform_input(self, input_grid, taskvars):
        input_color = taskvars['input_color']
        close_color = taskvars['close_color']
        
        output_grid = input_grid.copy()
        objects = find_connected_objects(input_grid, diagonal_connectivity=False)
        
        for obj in objects:
            if obj.has_color(input_color):
                obj_array = obj.to_array()
                if self._is_closed_shape(obj_array):
                    for r, c, _ in obj:
                        output_grid[r, c] = close_color
        
        return output_grid

