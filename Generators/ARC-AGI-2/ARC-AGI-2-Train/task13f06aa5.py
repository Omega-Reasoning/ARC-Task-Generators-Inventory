from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import retry, create_object, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, List, Tuple, Set

class Task13f06aa5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each grid has a solid background filled with a single color (ranging from 1 to 9) and contains several colored objects.",
            "The colored objects are hat-shaped and follow the pattern [[a, c, c, c, a], [c, c, b, c, c]], where a represents the background color, and b and c form the object, with b being a single distinct cell.",
            "These objects should also appear in rotated forms, such as [[a, c], [c, c], [c, b], [c, c], [a, c]], or similar variations.",
            "There can be at most 4 such objects in a grid.",
            "Each object is associated with one of the four grid borders, meaning that the b cell of each object faces a specific grid edge.",
            "No part of any object touches the grid border.",
            "The placement must ensure that if a straight line is extended from the b cell towards its associated border, it does not intersect or get blocked by any part of another object.",
            "The number of cells in the line between the b cell and the corresponding border must be even.",
            "The colors a, b, and c are not fixed and should vary across grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are created by copying the input grids.",
            "In the copied grid, all colored objects with a hat shape are identified.",
            "The colored objects are hat-shaped and follow the pattern [[a, c, c, c, a], [c, c, b, c, c]], where a represents the background color, and b and c form the object, with b being a single distinct cell.",
            "These objects may also appear in rotated forms, such as [[a, c], [c, c], [c, b], [c, c], [a, c]], or similar variations.",
            "There can be at most 4 such objects in a grid where each object is associated with one of the four grid borders.",
            "The b cell of each object faces its respective grid edge.",
            "A transformation is applied by drawing a dotted line from the b cell to the corresponding grid border.",
            "The dotted line alternates between color b and background color a.",
            "The cell immediately after b in the line is filled with a, then the next with b, then a, and so on.",
            "The line continues in this alternating pattern until it reaches the border.",
            "Once the line reaches the border, the entire respective border is filled with color b.",
            "If two different objects with different b colors are assigned to adjacent borders that share a corner cell, that corner cell is left empty (0) instead of assigning a conflicting color.",
            "The color b is specific to each object and can vary between them.",
            "Colors a, b, and c are not fixed and may vary across different grids."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        rows = random.randint(16, 30)
        cols = random.randint(16, 30)
        taskvars = {'rows': rows, 'cols': cols}

        # We'll try a few times to generate 4 examples whose actual detected counts
        # of hat-objects are all distinct. We pick desired counts from 1..4 and
        # request generation for those counts; if the generator falls back and
        # uniqueness is lost, we retry a few times.
        max_objects = 4
        max_attempts = 8

        for attempt in range(max_attempts):
            desired_counts = random.sample(list(range(1, max_objects + 1)), 4)

            train_grids = []
            success = True

            # create the three training grids
            for i in range(3):
                desired_num = desired_counts[i]
                if i == 1:
                    preferred = ['top', 'bottom', 'left', 'right']
                    specific_borders = preferred[:desired_num]
                    try:
                        input_grid, objects_info = self.create_input_with_info(taskvars, {
                            'num_objects': desired_num,
                            'specific_borders': specific_borders
                        })
                        grid_data = {'input': input_grid, 'output': None, 'objects_info': objects_info}
                    except ValueError:
                        grid_data = self.create_grid_with_fallback(taskvars, max_objects=desired_num)
                else:
                    try:
                        input_grid, objects_info = self.create_input_with_info(taskvars, {'num_objects': desired_num})
                        grid_data = {'input': input_grid, 'output': None, 'objects_info': objects_info}
                    except ValueError:
                        grid_data = self.create_grid_with_fallback(taskvars, max_objects=desired_num)

                grid_data['output'] = self.transform_input_with_info(
                    grid_data['input'], taskvars, grid_data['objects_info']
                )
                grid_data.pop('objects_info', None)
                train_grids.append(grid_data)

            # create the test grid
            desired_test_num = desired_counts[3]
            try:
                test_input, test_objects_info = self.create_input_with_info(taskvars, {'num_objects': desired_test_num})
                test_grid = {'input': test_input, 'output': None, 'objects_info': test_objects_info}
            except ValueError:
                test_grid = self.create_grid_with_fallback(taskvars, max_objects=desired_test_num)

            test_grid['output'] = self.transform_input_with_info(
                test_grid['input'], taskvars, test_grid['objects_info']
            )
            test_grid.pop('objects_info', None)

            # compute actual counts by detecting hat objects
            actual_counts = []
            try:
                for t in train_grids:
                    grid = t['input']
                    bg = self.get_background_color(grid)
                    objs = self.detect_hat_objects(grid, bg, grid.shape[0], grid.shape[1])
                    actual_counts.append(len(objs))

                bg_test = self.get_background_color(test_grid['input'])
                test_count = len(self.detect_hat_objects(test_grid['input'], bg_test, test_grid['input'].shape[0], test_grid['input'].shape[1]))
                actual_counts.append(test_count)
            except Exception:
                # Something went wrong during detection; try again
                success = False

            if not success:
                continue

            if len(set(actual_counts)) == 4:
                # success: all counts are distinct
                train_test_data = {'train': train_grids, 'test': [test_grid]}
                return taskvars, train_test_data

        # If we exit the loop without success, return the last generated set
        train_test_data = {'train': train_grids, 'test': [test_grid]}
        return taskvars, train_test_data


    def create_grid_with_fallback(self, taskvars: Dict[str, Any], max_objects: int) -> Dict[str, Any]:
        """Create a grid with fallback to fewer objects if needed."""
        for num_objects in range(max_objects, 0, -1):
            try:
                input_grid, objects_info = self.create_input_with_info(taskvars, {'num_objects': num_objects})
                return {'input': input_grid, 'output': None, 'objects_info': objects_info}
            except ValueError:
                continue
        
        # If all else fails, create a single object
        input_grid, objects_info = self.create_input_with_info(taskvars, {'num_objects': 1})
        return {'input': input_grid, 'output': None, 'objects_info': objects_info}

    def create_grid_with_specific_borders(self, taskvars: Dict[str, Any], borders: List[str]) -> Dict[str, Any]:
        """Try to create grid with specific borders, fallback to fewer objects."""
        for i in range(len(borders), 0, -1):
            try:
                selected_borders = borders[:i]
                input_grid, objects_info = self.create_input_with_info(taskvars, {
                    'num_objects': len(selected_borders), 
                    'specific_borders': selected_borders
                })
                return {'input': input_grid, 'output': None, 'objects_info': objects_info}
            except ValueError:
                continue
        
        # Fallback to single object
        input_grid, objects_info = self.create_input_with_info(taskvars, {'num_objects': 1})
        return {'input': input_grid, 'output': None, 'objects_info': objects_info}

    def create_input_with_info(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> Tuple[np.ndarray, List]:
        """Create input grid and return both grid and objects info."""
        rows, cols = taskvars['rows'], taskvars['cols']
        num_objects = gridvars['num_objects']
        
        def generate_valid_grid():
            # Choose background color (1-9)
            background = random.randint(1, 9)
            grid = np.full((rows, cols), background, dtype=int)
            
            # Available colors for objects (excluding background)
            available_colors = [c for c in range(1, 10) if c != background]
            
            # Check if we have enough colors
            if len(available_colors) < num_objects * 2:
                return None
            
            # Choose borders for objects
            if 'specific_borders' in gridvars:
                borders = gridvars['specific_borders']
            else:
                borders = random.sample(['top', 'bottom', 'left', 'right'], num_objects)
            
            # Pre-allocate unique colors for each object
            color_assignments = []
            used_colors = set()
            
            for i in range(num_objects):
                # Choose b_color
                available_b = [c for c in available_colors if c not in used_colors]
                if not available_b:
                    return None
                b_color = random.choice(available_b)
                used_colors.add(b_color)
                
                # Choose c_color
                available_c = [c for c in available_colors if c not in used_colors]
                if not available_c:
                    return None
                c_color = random.choice(available_c)
                used_colors.add(c_color)
                
                color_assignments.append((b_color, c_color))
            
            objects_info = []
            occupied_cells = set()
            
            for i, border in enumerate(borders):
                b_color, c_color = color_assignments[i]
                
                # Create hat object based on border
                hat_pattern = self.create_hat_pattern(border, background, b_color, c_color)
                
                # Find valid position for this object
                valid_position = self.find_valid_position(grid, hat_pattern, border, background, occupied_cells, rows, cols)
                if valid_position is None:
                    return None
                
                # Place the object
                r, c = valid_position
                h, w = hat_pattern.shape
                
                # Update grid with non-background cells only
                object_cells = set()
                for i in range(h):
                    for j in range(w):
                        if hat_pattern[i, j] != background:
                            grid[r + i, c + j] = hat_pattern[i, j]
                            object_cells.add((r + i, c + j))
                
                # Find b cell position
                b_pos = self.find_b_position(hat_pattern, (r, c), b_color)
                
                # Get line cells from b to border
                line_cells = self.get_line_to_border(b_pos, border, rows, cols)
                
                # Store object info
                objects_info.append({
                    'border': border,
                    'b_pos': b_pos,
                    'b_color': b_color,
                    'c_color': c_color,
                    'object_cells': object_cells,
                    'line_cells': line_cells
                })
                
                # Update occupied cells (object + line + buffer)
                occupied_cells.update(object_cells)
                occupied_cells.update(line_cells)
                
                # Add buffer around object
                buffer_cells = self.get_buffer_zone(object_cells, 2)
                occupied_cells.update(buffer_cells)
            
            return grid, objects_info
        
        result = retry(generate_valid_grid, lambda x: x is not None, max_attempts=100)
        if result is None:
            raise ValueError("Could not generate valid grid")
        
        grid, objects_info = result
        return grid, objects_info

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Legacy method for compatibility."""
        grid, objects_info = self.create_input_with_info(taskvars, gridvars)
        self.current_objects_info = objects_info
        return grid

    def get_buffer_zone(self, object_cells: Set[Tuple[int, int]], buffer_size: int) -> Set[Tuple[int, int]]:
        """Get buffer zone around object cells."""
        buffer_cells = set()
        for r, c in object_cells:
            for dr in range(-buffer_size, buffer_size + 1):
                for dc in range(-buffer_size, buffer_size + 1):
                    if dr == 0 and dc == 0:
                        continue
                    buffer_cells.add((r + dr, c + dc))
        return buffer_cells

    def create_hat_pattern(self, border: str, background: int, b_color: int, c_color: int) -> np.ndarray:
        """Create hat pattern based on border direction."""
        if border == 'top':
            hat_pattern = np.array([
                [background, c_color, b_color, c_color, background],
                [c_color, c_color, c_color, c_color, c_color]
            ])
        elif border == 'bottom':
            hat_pattern = np.array([
                [c_color, c_color, c_color, c_color, c_color],
                [background, c_color, b_color, c_color, background]
            ])
        elif border == 'left':
            hat_pattern = np.array([
                [background, c_color],
                [c_color, c_color],
                [b_color, c_color],
                [c_color, c_color],
                [background, c_color]
            ])
        elif border == 'right':
            hat_pattern = np.array([
                [c_color, background],
                [c_color, c_color],
                [c_color, b_color],
                [c_color, c_color],
                [c_color, background]
            ])
        
        return hat_pattern

    def get_line_to_border(self, b_pos: Tuple[int, int], border: str, rows: int, cols: int) -> List[Tuple[int, int]]:
        """Get all cells in the line from b cell to border."""
        b_r, b_c = b_pos
        
        if border == 'top':
            return [(i, b_c) for i in range(b_r - 1, -1, -1)]
        elif border == 'bottom':
            return [(i, b_c) for i in range(b_r + 1, rows)]
        elif border == 'left':
            return [(b_r, i) for i in range(b_c - 1, -1, -1)]
        elif border == 'right':
            return [(b_r, i) for i in range(b_c + 1, cols)]

    def get_border_cells(self, border: str, rows: int, cols: int) -> List[Tuple[int, int]]:
        """Get all cells on the specified border."""
        if border == 'top':
            return [(0, c) for c in range(cols)]
        elif border == 'bottom':
            return [(rows - 1, c) for c in range(cols)]
        elif border == 'left':
            return [(r, 0) for r in range(rows)]
        elif border == 'right':
            return [(r, cols - 1) for r in range(rows)]

    def find_valid_position(self, grid: np.ndarray, hat_pattern: np.ndarray, border: str, background: int, occupied_cells: Set[Tuple[int, int]], rows: int, cols: int) -> Tuple[int, int]:
        """Find a valid position for the hat object ensuring no border touching."""
        h, w = hat_pattern.shape
        
        # Ensure objects don't touch borders - more restrictive placement
        min_border_distance = 3  # Minimum distance from any border
        
        if border == 'top':
            row_range = range(min_border_distance, rows // 2 + 2)
            col_range = range(min_border_distance, cols - w - min_border_distance)
        elif border == 'bottom':
            row_range = range(rows // 2 - 2, rows - h - min_border_distance)
            col_range = range(min_border_distance, cols - w - min_border_distance)
        elif border == 'left':
            row_range = range(min_border_distance, rows - h - min_border_distance)
            col_range = range(min_border_distance, cols // 2 + 2)
        elif border == 'right':
            row_range = range(min_border_distance, rows - h - min_border_distance)
            col_range = range(cols // 2 - 2, cols - w - min_border_distance)
        
        # Try all positions in the range
        positions = [(r, c) for r in row_range for c in col_range]
        random.shuffle(positions)
        
        for r, c in positions:
            if self.is_valid_position(grid, hat_pattern, (r, c), border, background, occupied_cells, rows, cols):
                return (r, c)
        
        return None

    def is_valid_position(self, grid: np.ndarray, hat_pattern: np.ndarray, pos: Tuple[int, int], border: str, background: int, occupied_cells: Set[Tuple[int, int]], rows: int, cols: int) -> bool:
        """Check if position is valid ensuring no border touching."""
        r, c = pos
        h, w = hat_pattern.shape
        
        # Check bounds
        if r + h > rows or c + w > cols or r < 0 or c < 0:
            return False
        
        # Check if object would touch any border
        if r == 0 or c == 0 or r + h == rows or c + w == cols:
            return False
        
        # Check if object cells would conflict
        object_cells = set()
        for i in range(h):
            for j in range(w):
                if hat_pattern[i, j] != background:
                    cell_pos = (r + i, c + j)
                    if cell_pos in occupied_cells:
                        return False
                    object_cells.add(cell_pos)
        
        # Find b cell position
        b_pos = self.find_b_position(hat_pattern, pos, None)
        if b_pos is None:
            return False
        
        # Check line to border
        return self.check_line_to_border(b_pos, border, occupied_cells, rows, cols)

    def find_b_position(self, hat_pattern: np.ndarray, offset: Tuple[int, int], b_color: int) -> Tuple[int, int]:
        """Find the position of the b cell in the hat pattern."""
        r_off, c_off = offset
        
        if b_color is None:
            colors, counts = np.unique(hat_pattern, return_counts=True)
            background_idx = np.argmax(counts)
            background = colors[background_idx]
            
            other_colors = [(colors[i], counts[i]) for i in range(len(colors)) if i != background_idx]
            if len(other_colors) < 2:
                return None
            
            other_colors.sort(key=lambda x: x[1])
            b_color = other_colors[0][0]
        
        b_positions = np.where(hat_pattern == b_color)
        if len(b_positions[0]) == 0:
            return None
        
        return (b_positions[0][0] + r_off, b_positions[1][0] + c_off)

    def check_line_to_border(self, b_pos: Tuple[int, int], border: str, occupied_cells: Set[Tuple[int, int]], rows: int, cols: int) -> bool:
        """Check if line from b cell to border is valid."""
        b_r, b_c = b_pos
        
        if border == 'top':
            distance = b_r
            line_cells = [(i, b_c) for i in range(b_r - 1, -1, -1)]
        elif border == 'bottom':
            distance = rows - 1 - b_r
            line_cells = [(i, b_c) for i in range(b_r + 1, rows)]
        elif border == 'left':
            distance = b_c
            line_cells = [(b_r, i) for i in range(b_c - 1, -1, -1)]
        elif border == 'right':
            distance = cols - 1 - b_c
            line_cells = [(b_r, i) for i in range(b_c + 1, cols)]
        
        # Check if distance is even
        if distance % 2 != 0:
            return False
        
        # Check if line is clear
        for cell in line_cells:
            if cell in occupied_cells:
                return False
        
        return True

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input grid by detecting hat objects and drawing lines with border filling."""
        output = grid.copy()
        rows, cols = grid.shape
        
        # Get background color
        background = self.get_background_color(grid)
        
        # Detect hat objects in the grid
        hat_objects = self.detect_hat_objects(grid, background, rows, cols)
        
        # Keep track of border assignments
        border_assignments = {}
        
        # First pass: draw dotted lines
        for obj_info in hat_objects:
            border = obj_info['border']
            b_pos = obj_info['b_pos']
            b_color = obj_info['b_color']
            line_cells = obj_info['line_cells']
            
            # Store border assignment
            border_assignments[border] = b_color
            
            # Draw alternating line
            for i, (r, c) in enumerate(line_cells):
                if i % 2 == 0:
                    color = background
                else:
                    color = b_color
                
                output[r, c] = color
        
        # Second pass: fill borders
        for border, b_color in border_assignments.items():
            border_cells = self.get_border_cells(border, rows, cols)
            
            for r, c in border_cells:
                # Check if this is a corner cell
                is_corner = ((r == 0 or r == rows - 1) and (c == 0 or c == cols - 1))
                
                if is_corner:
                    # Check for conflicts with adjacent borders
                    conflicting_colors = set()
                    
                    # Check all borders that share this corner
                    if r == 0 and c == 0:  # Top-left corner
                        if 'top' in border_assignments:
                            conflicting_colors.add(border_assignments['top'])
                        if 'left' in border_assignments:
                            conflicting_colors.add(border_assignments['left'])
                    elif r == 0 and c == cols - 1:  # Top-right corner
                        if 'top' in border_assignments:
                            conflicting_colors.add(border_assignments['top'])
                        if 'right' in border_assignments:
                            conflicting_colors.add(border_assignments['right'])
                    elif r == rows - 1 and c == 0:  # Bottom-left corner
                        if 'bottom' in border_assignments:
                            conflicting_colors.add(border_assignments['bottom'])
                        if 'left' in border_assignments:
                            conflicting_colors.add(border_assignments['left'])
                    elif r == rows - 1 and c == cols - 1:  # Bottom-right corner
                        if 'bottom' in border_assignments:
                            conflicting_colors.add(border_assignments['bottom'])
                        if 'right' in border_assignments:
                            conflicting_colors.add(border_assignments['right'])
                    
                    # If there are multiple different colors, set to 0
                    if len(conflicting_colors) > 1:
                        output[r, c] = 0
                    else:
                        output[r, c] = b_color
                else:
                    # Non-corner border cell
                    output[r, c] = b_color
        
        return output

    def get_background_color(self, grid: np.ndarray) -> int:
        """Get the background color from the grid."""
        # Background color is the most frequent color
        unique, counts = np.unique(grid, return_counts=True)
        return unique[np.argmax(counts)]

    def detect_hat_objects(self, grid: np.ndarray, background: int, rows: int, cols: int) -> List[Dict[str, Any]]:
        """Detect hat objects in the grid and return their information."""
        hat_objects = []
        
        # Find all non-background connected components
        visited = set()
        
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != background and (r, c) not in visited:
                    # Found a new object, explore it
                    obj_cells = self.get_connected_component(grid, (r, c), background, visited)
                    
                    # Check if this is a hat object
                    hat_info = self.analyze_hat_object(grid, obj_cells, background, rows, cols)
                    if hat_info:
                        hat_objects.append(hat_info)
        
        return hat_objects

    def get_connected_component(self, grid: np.ndarray, start: Tuple[int, int], background: int, visited: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
        """Get all cells in the connected component starting from start."""
        rows, cols = grid.shape
        component = set()
        stack = [start]
        
        while stack:
            r, c = stack.pop()
            if (r, c) in visited or r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if grid[r, c] == background:
                continue
            
            visited.add((r, c))
            component.add((r, c))
            
            # Add neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))
        
        return component

    def analyze_hat_object(self, grid: np.ndarray, obj_cells: Set[Tuple[int, int]], background: int, rows: int, cols: int) -> Dict[str, Any]:
        """Analyze if the object is a hat and return its information."""
        # Get colors in the object
        colors = set()
        for r, c in obj_cells:
            colors.add(grid[r, c])
        
        # Hat should have exactly 2 colors (b and c)
        if len(colors) != 2:
            return None
        
        # Find b_color (appears less frequently) and c_color
        color_counts = {}
        for r, c in obj_cells:
            color = grid[r, c]
            color_counts[color] = color_counts.get(color, 0) + 1
        
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])
        b_color = sorted_colors[0][0]
        c_color = sorted_colors[1][0]
        
        # Find b cell position (should be unique)
        b_positions = [(r, c) for r, c in obj_cells if grid[r, c] == b_color]
        if len(b_positions) != 1:
            return None
        
        b_pos = b_positions[0]
        
        # Determine which border this object faces
        border = self.determine_border_direction(obj_cells, b_pos, rows, cols)
        if not border:
            return None
        
        # Get line cells to border
        line_cells = self.get_line_to_border(b_pos, border, rows, cols)
        
        return {
            'border': border,
            'b_pos': b_pos,
            'b_color': b_color,
            'c_color': c_color,
            'object_cells': obj_cells,
            'line_cells': line_cells
        }

    def determine_border_direction(self, obj_cells: Set[Tuple[int, int]], b_pos: Tuple[int, int], rows: int, cols: int) -> str:
        """Determine which border the hat object faces based on its shape."""
        b_r, b_c = b_pos
        
        # Check the relative position of b cell within the object
        min_r = min(r for r, c in obj_cells)
        max_r = max(r for r, c in obj_cells)
        min_c = min(c for r, c in obj_cells)
        max_c = max(c for r, c in obj_cells)
        
        # Determine direction based on b cell position within the object bounds
        if b_r == min_r:  # b is at the top of the object
            return 'top'
        elif b_r == max_r:  # b is at the bottom of the object
            return 'bottom'
        elif b_c == min_c:  # b is at the left of the object
            return 'left'
        elif b_c == max_c:  # b is at the right of the object
            return 'right'
        
        return None

    def transform_input_with_info(self, grid: np.ndarray, taskvars: Dict[str, Any], objects_info: List) -> np.ndarray:
        """Transform input using provided objects info."""
        output = grid.copy()
        rows, cols = grid.shape
        
        # Get background color
        background = grid[0, 0]
        
        # Keep track of border assignments
        border_assignments = {}
        
        # First pass: draw dotted lines
        for obj_info in objects_info:
            border = obj_info['border']
            b_pos = obj_info['b_pos']
            b_color = obj_info['b_color']
            line_cells = obj_info['line_cells']
            
            # Store border assignment
            border_assignments[border] = b_color
            
            # Draw alternating line
            for i, (r, c) in enumerate(line_cells):
                if i % 2 == 0:
                    color = background
                else:
                    color = b_color
                
                output[r, c] = color
        
        # Second pass: fill borders
        for border, b_color in border_assignments.items():
            border_cells = self.get_border_cells(border, rows, cols)
            
            for r, c in border_cells:
                # Check if this is a corner cell
                is_corner = ((r == 0 or r == rows - 1) and (c == 0 or c == cols - 1))
                
                if is_corner:
                    # Check for conflicts with adjacent borders
                    conflicting_colors = set()
                    
                    # Check all borders that share this corner
                    if r == 0 and c == 0:  # Top-left corner
                        if 'top' in border_assignments:
                            conflicting_colors.add(border_assignments['top'])
                        if 'left' in border_assignments:
                            conflicting_colors.add(border_assignments['left'])
                    elif r == 0 and c == cols - 1:  # Top-right corner
                        if 'top' in border_assignments:
                            conflicting_colors.add(border_assignments['top'])
                        if 'right' in border_assignments:
                            conflicting_colors.add(border_assignments['right'])
                    elif r == rows - 1 and c == 0:  # Bottom-left corner
                        if 'bottom' in border_assignments:
                            conflicting_colors.add(border_assignments['bottom'])
                        if 'left' in border_assignments:
                            conflicting_colors.add(border_assignments['left'])
                    elif r == rows - 1 and c == cols - 1:  # Bottom-right corner
                        if 'bottom' in border_assignments:
                            conflicting_colors.add(border_assignments['bottom'])
                        if 'right' in border_assignments:
                            conflicting_colors.add(border_assignments['right'])
                    
                    # If there are multiple different colors, set to 0
                    if len(conflicting_colors) > 1:
                        output[r, c] = 0
                    else:
                        output[r, c] = b_color
                else:
                    # Non-corner border cell
                    output[r, c] = b_color
        
        return output

