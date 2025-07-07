from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random
from scipy.ndimage import binary_fill_holes

class Taskb2862040Generator(ARCTaskGenerator):
    
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
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a grid with mixture of open and closed shapes."""
        grid_height = random.randint(15, 20)
        grid_width = random.randint(15, 20)
        
        grid = np.zeros((grid_height, grid_width), dtype=np.int32)
        input_color = taskvars['input_color']
        
        # Total shapes: 4-7 (to accommodate variable closed shapes)
        # 1. One line (3 cells) - OPEN
        # 2. One single cell - OPEN  
        # 3. 1-3 closed shapes - CLOSED
        # 4. Rest are OPEN shapes (1-3 additional)
        
        # Create 1 line (OPEN)
        line = self._create_line(input_color)
        self._place_pattern(grid, line, grid_height, grid_width)
        
        # Create 1 single cell (OPEN)
        single = self._create_single_cell(input_color)
        self._place_pattern(grid, single, grid_height, grid_width)
        
        # Create 1-3 closed shapes (CLOSED)
        num_closed_shapes = random.randint(1, 3)
        for _ in range(num_closed_shapes):
            closed_shape = self._create_closed_shape(input_color)
            self._place_pattern(grid, closed_shape, grid_height, grid_width)
        
        # Create 1-3 more OPEN shapes to fill out the grid
        num_open_shapes = random.randint(1, 3)
        for _ in range(num_open_shapes):
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
        size = random.randint(4, 7)
        pattern = np.zeros((size, size), dtype=np.int32)
        
        # Create basic hollow shape that's definitely closed
        pattern[0, :] = color  # Top
        pattern[-1, :] = color  # Bottom
        pattern[:, 0] = color  # Left
        pattern[:, -1] = color  # Right
        
        # Add funny/irregular elements while keeping it closed
        
        # 1. Add random line extenders from the boundary
        for _ in range(random.randint(1, 3)):
            side = random.choice(['top', 'bottom', 'left', 'right'])
            pos = random.randint(1, size-2)
            extend_length = random.randint(1, 2)
            
            if side == 'top':
                for i in range(1, min(extend_length + 1, size-1)):
                    pattern[i, pos] = color
            elif side == 'bottom':
                for i in range(max(1, size-extend_length-1), size-1):
                    pattern[i, pos] = color
            elif side == 'left':
                for i in range(1, min(extend_length + 1, size-1)):
                    pattern[pos, i] = color
            elif side == 'right':
                for i in range(max(1, size-extend_length-1), size-1):
                    pattern[pos, i] = color
        
        # 2. Add random single cell extenders (protruding from boundary)
        for _ in range(random.randint(0, 2)):
            # Find boundary cells
            boundary_cells = []
            for r in range(size):
                for c in range(size):
                    if pattern[r, c] == color:
                        if r == 0 or r == size-1 or c == 0 or c == size-1:
                            boundary_cells.append((r, c))
            
            if boundary_cells:
                br, bc = random.choice(boundary_cells)
                directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
                for dr, dc in random.sample(directions, len(directions)):
                    nr, nc = br + dr, bc + dc
                    if 0 <= nr < size and 0 <= nc < size and pattern[nr, nc] == 0:
                        pattern[nr, nc] = color
                        break
        
        # 3. Add internal structures that don't break the closure
        for _ in range(random.randint(0, 2)):
            if size >= 5:
                internal_r = random.randint(2, size-3)
                internal_c = random.randint(2, size-3)
                pattern[internal_r, internal_c] = color
                if random.choice([True, False]) and internal_c + 1 < size - 1:
                    pattern[internal_r, internal_c + 1] = color
        
        # 4. Add zigzag or wavy patterns to boundaries
        if random.choice([True, False]):
            for c in range(1, size-1):
                if random.choice([True, False, False]):
                    if pattern[1][c] == 0:
                        pattern[1][c] = color
        
        return pattern
    
    def _create_open_shape(self, color):
        size = random.randint(3, 5)
        pattern = np.zeros((size, size), dtype=np.int32)
        
        shape_type = random.choice(['L_shape', 'T_shape', 'arc', 'branch', 'random_walk'])
        
        if shape_type == 'L_shape':
            pattern[:, 0] = color
            pattern[-1, :size//2+1] = color
        elif shape_type == 'T_shape':
            center = size // 2
            pattern[0, :] = color
            pattern[:center+1, center] = color
        elif shape_type == 'arc':
            center = size // 2
            for r in range(size):
                for c in range(size):
                    dist = ((r - center) ** 2 + (c - center) ** 2) ** 0.5
                    angle = np.arctan2(r - center, c - center)
                    if abs(dist - center + 0.5) < 0.8 and -np.pi/2 < angle < np.pi/2:
                        pattern[r, c] = color
        elif shape_type == 'branch':
            center = size // 2
            pattern[:, center] = color
            if size >= 3:
                pattern[center, :center+1] = color
        else:  # random_walk
            r, c = size // 2, size // 2
            pattern[r, c] = color
            
            num_cells = random.randint(3, size * size // 2)
            for _ in range(num_cells - 1):
                dr, dc = random.choice([(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)])
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    pattern[nr, nc] = color
                    r, c = nr, nc
        
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
    
    def transform_input(self, grid, taskvars):
        """Transform input by changing closed shapes to close_color."""
        input_color = taskvars['input_color']
        close_color = taskvars['close_color']
        
        output_grid = grid.copy()
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # Transform closed shapes
        for obj in objects:
            if obj.has_color(input_color):
                obj_array = obj.to_array()
                
                # Inline closed shape detection to avoid function generation issues
                is_closed = False
                
                if obj_array.size > 4:
                    # Get the bounding box of non-zero elements
                    rows_with_data = np.any(obj_array != 0, axis=1)
                    cols_with_data = np.any(obj_array != 0, axis=0)
                    
                    if np.any(rows_with_data) and np.any(cols_with_data):
                        # Trim to only the area containing the object
                        trimmed = obj_array[rows_with_data][:, cols_with_data]
                        h, w = trimmed.shape
                        
                        # Need minimum size to potentially be closed
                        if h >= 3 and w >= 3:
                            # Create binary mask
                            binary_mask = (trimmed != 0).astype(bool)
                            
                            # Method 1: Use binary_fill_holes
                            try:
                                filled = binary_fill_holes(binary_mask)
                                holes_filled = np.sum(filled) - np.sum(binary_mask)
                                if holes_filled > 0:
                                    is_closed = True
                            except:
                                pass
                            
                            if not is_closed:
                                # Method 2: Check if shape has boundary structure typical of closed shapes
                                total_cells = h * w
                                filled_cells = np.sum(binary_mask)
                                density = filled_cells / total_cells
                                
                                # Check coverage on edges
                                top_coverage = np.sum(binary_mask[0, :]) / w
                                bottom_coverage = np.sum(binary_mask[-1, :]) / w
                                left_coverage = np.sum(binary_mask[:, 0]) / h
                                right_coverage = np.sum(binary_mask[:, -1]) / h
                                
                                edge_coverages = [top_coverage, bottom_coverage, left_coverage, right_coverage]
                                edges_with_good_coverage = sum(1 for cov in edge_coverages if cov > 0.3)
                                
                                if (0.25 <= density <= 0.85 and 
                                    edges_with_good_coverage >= 3 and 
                                    filled_cells >= 8):
                                    is_closed = True
                            
                            if not is_closed:
                                # Method 3: Flood fill from edges to detect internal holes
                                padded = np.zeros((h + 2, w + 2), dtype=bool)
                                padded[1:-1, 1:-1] = binary_mask
                                
                                # Flood fill from top-left corner (outside the shape)
                                visited = np.zeros_like(padded, dtype=bool)
                                stack = [(0, 0)]
                                
                                while stack:
                                    r, c = stack.pop()
                                    if (0 <= r < h + 2 and 0 <= c < w + 2 and 
                                        not visited[r, c] and not padded[r, c]):
                                        visited[r, c] = True
                                        stack.extend([(r+1, c), (r-1, c), (r, c+1), (r, c-1)])
                                
                                # Check if there are unvisited empty cells (holes)
                                internal_area = visited[1:-1, 1:-1]
                                holes = ~internal_area & ~binary_mask
                                
                                if np.sum(holes) > 0:
                                    is_closed = True
                
                if is_closed:
                    for r, c, _ in obj:
                        output_grid[r, c] = close_color
        
        return output_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        input_color = available_colors[0]
        close_color = available_colors[1]
        
        taskvars = {
            "input_color": input_color,
            "close_color": close_color
        }
        
        # Generate 3-5 training examples with same task variables
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with same task variables
        test_gridvars = {}
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

# Test code
if __name__ == "__main__":
    generator = Taskb2862040Generator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)