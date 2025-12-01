from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, List, Tuple
from input_library import retry, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task264363fdGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "The entire input grid is filled with color grid_color(between 1 and 9).",
            "There are two to three 4-way connected objects, which are either a rectangle or square.",
            "There is another 4-way object named pattern, which consists of either (1) a single center cell of color center_color(between 1 and 9) or (2) a 3x3 square of color square_color(between 1 and 9) with the center cell of color center_color(between 1 and 9).", 
            "In both the above cases, there are arms extending from the center cell of color arm_color(between 1 and 9) with arm length two, either horizontally(left and right of the center cell) or vertically(top and bottom of center cell) or both(left,top,right and bottom).",
            "In the above 4-way connected objects which are square or rectangle and not pattern object, one or two random cells are colored the same as pattern center cell color."
        ]
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "Identify the pattern object which has a center cell of a specific color, optionally surrounded by a 3x3 square, and arms extending horizontally, vertically, or both in the input grid",
            "The pattern doesnt appear in the output grid.",
            "For each remaining object, find cells that match the patterns center color.",
            "For each matching cell in other objects, if the pattern has a 3x3 square, create a similar square around the center cell within the objects bounds.",
            "Extend arms from each center cell horizontally, vertically, or both (matching the patterns arm direction) until reaching the objects boundaries."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables with random grid size between 20 and 30
        rows = random.randint(20, 30)
        
        taskvars = {
            'rows': rows
        }
        def generate_colors_and_pattern_vars() -> Dict[str, Any]:
            # Generate colors
            colors = list(range(1, 10))
            grid_color = random.choice(colors)
            colors.remove(grid_color)
            in_color = random.choice(colors)
            colors.remove(in_color)
            arm_color = random.choice(colors)
            colors.remove(arm_color)
            obj_color = random.choice(colors)
            colors.remove(obj_color)
            center_color = random.choice(colors)
            
            # Generate pattern variables
            arm_direction = random.choice(['horizontal', 'vertical', 'both'])
            pattern_size = 17 if arm_direction == 'both' else 13  # 3x3 + 8/4 arm cells
            
            return {
                "grid_color": grid_color,
                "in_color": in_color,
                "arm_color": arm_color,
                "center_color": center_color,
                "obj_color": obj_color,
                "arm_direction": arm_direction,
                "pattern_size": pattern_size
            }
        
        # Create train and test data
        num_train = random.randint(3, 4)
        train = []
        for _ in range(num_train):
            grid_vars = generate_colors_and_pattern_vars()
            input_grid = self.create_input(taskvars, grid_vars)
            output_grid = self.transform_input(input_grid, grid_vars)
            train.append({'input': input_grid, 'output': output_grid})
        
        grid_vars = generate_colors_and_pattern_vars()

        test_input = self.create_input(taskvars, grid_vars)
        test_output = self.transform_input(test_input, grid_vars)
        test = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train, 'test': test}

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']  # Use the variable rows from taskvars
        grid_color = gridvars['grid_color']
        in_color = gridvars['in_color']
        arm_color = gridvars['arm_color']
        center_color = gridvars['center_color']
        obj_color = gridvars['obj_color']
        arm_direction = gridvars['arm_direction']
        pattern_size = gridvars['pattern_size']
        
        grid = np.full((rows, rows), grid_color, dtype=int)
        
        # Create and place pattern object
        pattern = self.create_pattern(in_color, center_color, arm_color, arm_direction, grid_color)
        pattern_rows, pattern_cols = pattern.shape
        
        def place_pattern():
            r = random.randint(0, rows - pattern_rows)
            c = random.randint(0, rows - pattern_cols)
            # Check if there's at least one cell spacing around the pattern
            r_start = max(0, r - 1)
            r_end = min(rows, r + pattern_rows + 1)
            c_start = max(0, c - 1)
            c_end = min(rows, c + pattern_cols + 1)
            if np.all(grid[r_start:r_end, c_start:c_end] == grid_color):
                grid[r:r+pattern_rows, c:c+pattern_cols] = pattern
                return True
            return False
        
        retry(place_pattern, lambda x: x, max_attempts=300)
        
        # Create non-pattern objects
        max_dim = rows // 3
        # Set minimum object size based on grid size to avoid issues
        min_obj_size = max(5, max_dim // 2)
        # Ensure max_dim is reasonable
        if max_dim < min_obj_size:
            max_dim = min(rows // 2, rows - 5)
            min_obj_size = max(5, max_dim // 2)
        
        num_objects = random.randint(2, 3)
        
        for _ in range(num_objects):
            while True:
                h = random.randint(min_obj_size, max_dim)
                w = random.randint(min_obj_size, max_dim)
                if h * w > pattern_size:
                    break
            
            obj_rows = h
            obj_cols = w
            
            def place_object():
                r = random.randint(0, rows - obj_rows)
                c = random.randint(0, rows - obj_cols)
                # Check if there's at least one cell spacing around the object
                r_start = max(0, r - 1)
                r_end = min(rows, r + obj_rows + 1)
                c_start = max(0, c - 1)
                c_end = min(rows, c + obj_cols + 1)
                if np.all(grid[r_start:r_end, c_start:c_end] == grid_color):
                    grid[r:r+obj_rows, c:c+obj_cols] = obj_color
                    # Select 1-2 cells to set to center_color, ensuring they are separated
                    cells = [(rr, cc) for rr in range(r + 1, r+obj_rows - 1) for cc in range(c + 1, c+obj_cols - 1)]
                    if not cells:  # If object is too small, use all interior cells
                        cells = [(rr, cc) for rr in range(r, r+obj_rows) for cc in range(c, c+obj_cols)]
                    
                    if cells:
                        first_cell = random.choice(cells)
                        # Remove first cell and cells in same row/column and adjacent cells from candidates
                        remaining_cells = [(rr, cc) for rr, cc in cells 
                                         if (abs(rr - first_cell[0]) > 2 or abs(cc - first_cell[1]) > 2)
                                         and rr != first_cell[0]  # different row
                                         and cc != first_cell[1]
                                         and abs(rr - first_cell[0]) > 1
                                         and abs(cc - first_cell[1]) > 1] # different column
                        
                        selected = [first_cell]
                        if remaining_cells:  # 50% chance to add second cell
                            selected.append(random.choice(remaining_cells))
                        for rr, cc in selected:
                            grid[rr, cc] = center_color
                    return True
                return False
            
            retry(place_object, lambda x: x, max_attempts=300)
        
        return grid
    
    def create_pattern(self, in_color: int, center_color: int, arm_color: int, direction: str, grid_color: int) -> np.ndarray:
        # Randomly decide whether to include the 3x3 square
        has_square = random.choice([True, False])
        
        if has_square:
            # Create 3x3 square as before
            pattern = np.full((3, 3), in_color, dtype=int)
            pattern[1, 1] = center_color
            center_pos = 1  # center position in the 3x3 grid
        else:
            # Create single center cell
            pattern = np.array([[center_color]], dtype=int)
            center_pos = 0  # center position in the 1x1 grid
        
        arms = []
        if direction in ['horizontal', 'both']:
            arms.extend([(center_pos, center_pos - i, arm_color) for i in range(1, 3)])
            arms.extend([(center_pos, center_pos + i, arm_color) for i in range(1, 3)])
        if direction in ['vertical', 'both']:
            arms.extend([(center_pos - i, center_pos, arm_color) for i in range(1, 3)])
            arms.extend([(center_pos + i, center_pos, arm_color) for i in range(1, 3)])
        
        # Determine bounding box
        min_r = max_r = center_pos
        min_c = max_c = center_pos
        for r, c, _ in arms:
            min_r = min(min_r, r)
            max_r = max(max_r, r)
            min_c = min(min_c, c)
            max_c = max(max_c, c)
        
        # Expand to include original pattern
        min_r = min(min_r, 0)
        max_r = max(max_r, pattern.shape[0] - 1)
        min_c = min(min_c, 0)
        max_c = max(max_c, pattern.shape[1] - 1)
        
        pattern_height = max_r - min_r + 1
        pattern_width = max_c - min_c + 1
        pattern_array = np.full((pattern_height, pattern_width), grid_color, dtype=int)
        
        # Fill the original pattern
        pattern_r, pattern_c = pattern.shape
        for r in range(pattern_r):
            for c in range(pattern_c):
                pattern_array[r - min_r, c - min_c] = pattern[r, c]
        
        # Add arms
        for r, c, color in arms:
            pattern_array[r - min_r, c - min_c] = color
        
        return pattern_array

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = np.copy(grid)
        rows, cols = output.shape

        # Find grid_color (most common color)
        grid_color = max(set(output.flatten()), key=lambda x: np.sum(output == x))
        
        # Find all objects
        objects = find_connected_objects(output, diagonal_connectivity=False, background=grid_color, monochromatic=False)
        
        # Find the pattern object (smallest multi-colored object)
        pattern_obj = min(
            [obj for obj in objects.objects if len({color for _, _, color in obj.cells}) > 1],
            key=lambda obj: len(obj.cells)
        )
        
        # Extract pattern properties
        colors_in_pattern = {color for _, _, color in pattern_obj.cells}
        colors_in_pattern.discard(grid_color)
        
        color_counts = {color: sum(1 for _, _, c in pattern_obj.cells if c == color) 
                       for color in colors_in_pattern}
        
        # Center color appears exactly once
        center_color = next(color for color, count in color_counts.items() if count == 1)
        center_r, center_c = [(r, c) for r, c, color in pattern_obj.cells if color == center_color][0]
        
        # Find arm color (appears 4 or 8 times)
        arm_color = next(color for color, count in color_counts.items() 
                        if count in [4, 8] and color != center_color)
        
        # Find in_color if it exists (remaining color with count == 8)
        remaining_colors = colors_in_pattern - {arm_color, center_color}
        in_color = next(iter(remaining_colors)) if remaining_colors else None
        has_square = in_color is not None
        
        # Determine arm directions by checking positions relative to center
        arm_cells = [(r, c) for r, c, color in pattern_obj.cells if color == arm_color]
        
        has_horizontal = any(c != center_c and r == center_r for r, c in arm_cells)
        has_vertical = any(r != center_r and c == center_c for r, c in arm_cells)
        
        # Find obj_color (color of other objects)
        obj_color = next((color for obj in objects.objects 
                         if obj != pattern_obj 
                         for _, _, color in obj.cells 
                         if color not in [grid_color, center_color]), None)
        
        if obj_color is None:
            return output
        
        # Clear the pattern from output
        for r, c, _ in pattern_obj.cells:
            output[r, c] = grid_color
        
        # Re-find objects after clearing pattern
        objects = find_connected_objects(output, diagonal_connectivity=False, background=grid_color, monochromatic=False)
        
        # Process each remaining object
        for obj in objects.objects:
            # Find cells with matching center color
            center_matches = [(r, c) for r, c, color in obj.cells if color == center_color]
            
            slice_r, slice_c = obj.bounding_box
            
            for center_r, center_c in center_matches:
                # Construct 3x3 pattern around center cell if pattern has square
                if has_square and in_color is not None:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:  # Skip center cell
                                continue
                            new_r, new_c = center_r + dr, center_c + dc
                            # Only fill if within object bounds and currently obj_color
                            if (slice_r.start <= new_r < slice_r.stop and 
                                slice_c.start <= new_c < slice_c.stop and
                                (new_r, new_c) in obj.coords and
                                output[new_r, new_c] == obj_color):
                                output[new_r, new_c] = in_color
                
                # Extend arms to object boundaries
                if has_horizontal:
                    # Extend left
                    c = center_c - 1
                    while c >= slice_c.start and (center_r, c) in obj.coords:
                        output[center_r, c] = arm_color
                        c -= 1
                    
                    # Extend right
                    c = center_c + 1
                    while c < slice_c.stop and (center_r, c) in obj.coords:
                        output[center_r, c] = arm_color
                        c += 1
                
                if has_vertical:
                    # Extend up
                    r = center_r - 1
                    while r >= slice_r.start and (r, center_c) in obj.coords:
                        output[r, center_c] = arm_color
                        r -= 1
                    
                    # Extend down
                    r = center_r + 1
                    while r < slice_r.stop and (r, center_c) in obj.coords:
                        output[r, center_c] = arm_color
                        r += 1
        
        return output