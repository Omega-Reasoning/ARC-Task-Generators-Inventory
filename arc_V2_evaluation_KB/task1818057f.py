from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring
import numpy as np
import random

class Task1818057fGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains a completely filled background of {color('background_color')} and several {color('object_color')} objects randomly placed in the grid.",
            "The {color('object_color')} objects may have different arrangements in different grids, but it is guaranteed that each input grid contains at least one {color('object_color')} object into which a cross shape can be fitted.",
            "The cross shape is defined as: [0, c, 0], [c, c, c], [0, c, 0].",
            "The {color('object_color')} objects vary in shape: some are single cells, some are block-shaped, and some resemble a vertically elongated version of the cross shape.",
            "The {color('object_color')} objects are shaped and sized in such a way that, at any given location, there is only one unique way to add the cross shape—no multiple placements are possible at the same spot."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all {color('object_color')} objects.",
            "Once the {color('object_color')} objects have been identified, locate those that can accommodate a cross shape.",
            "The cross shape is defined as: [0,c,0],[c,c,c],[0,c,0].",
            "After identifying the {color('object_color')} objects that can fit a cross shape, add a {color('cross_color')} cross shape at those locations."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        def create_example(taskvars, example_type):
            """Create a single example with specified type"""
            gridvars = {'example_type': example_type}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            return {'input': input_grid, 'output': output_grid}
        
        # Generate random colors ensuring they are all different
        all_colors = list(range(1, 10))
        selected_colors = random.sample(all_colors, 3)
        
        taskvars = {
            'background_color': selected_colors[0],
            'object_color': selected_colors[1], 
            'cross_color': selected_colors[2]
        }
        
        # Create training examples
        train_examples = []
        train_examples.append(create_example(taskvars, 'basic_shapes'))
        train_examples.append(create_example(taskvars, 'elongated_crosses'))
        train_examples.append(create_example(taskvars, 'dense_packing'))
        train_examples.append(create_example(taskvars, 'random'))
        
        if random.choice([True, False]):
            train_examples.append(create_example(taskvars, 'random'))
        
        # Create test example
        test_example = create_example(taskvars, 'network_grid')
        
        return taskvars, {'train': train_examples, 'test': [test_example]}
    
    def create_input(self, taskvars, gridvars):
        def can_place_pattern(grid, pattern, background_color):
            """Check if pattern can be placed with buffer zone"""
            height, width = grid.shape
            
            for pr, pc in pattern:
                if not (0 <= pr < height and 0 <= pc < width) or grid[pr, pc] != background_color:
                    return False
            
            buffer_zone = []
            for pr, pc in pattern:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        br, bc = pr + dr, pc + dc
                        if (br, bc) not in pattern:
                            buffer_zone.append((br, bc))
            
            for br, bc in set(buffer_zone):
                if (0 <= br < height and 0 <= bc < width and 
                    grid[br, bc] != background_color):
                    return False
            
            return True
        
        def is_pattern_isolated(grid, pattern, taskvars, min_distance):
            """Check if pattern can be placed with sufficient isolation"""
            height, width = grid.shape
            
            for pr, pc in pattern:
                if not (0 <= pr < height and 0 <= pc < width):
                    return False
                if grid[pr, pc] != taskvars['background_color']:
                    return False
            
            for pr, pc in pattern:
                for dr in range(-min_distance, min_distance + 1):
                    for dc in range(-min_distance, min_distance + 1):
                        check_r, check_c = pr + dr, pc + dc
                        if (0 <= check_r < height and 0 <= check_c < width and 
                            (check_r, check_c) not in pattern and 
                            grid[check_r, check_c] == taskvars['object_color']):
                            return False
            
            return True
        
        def add_t_shape(grid, taskvars):
            """Add T-shape that allows exactly one cross placement"""
            height, width = grid.shape
            for _ in range(20):
                r = random.randint(2, height-3)
                c = random.randint(2, width-3)
                pattern = [(r-1, c-1), (r-1, c), (r-1, c+1), (r, c), (r+1, c)]
                if can_place_pattern(grid, pattern, taskvars['background_color']):
                    for pr, pc in pattern:
                        grid[pr, pc] = taskvars['object_color']
                    break
        
        def add_vertical_line_with_cross(grid, taskvars):
            """Add 6-cell vertical line with horizontal line crossing it"""
            height, width = grid.shape
            for _ in range(20):
                r_start = random.randint(1, height-7)
                c = random.randint(1, width-2)
                cross_row = random.randint(r_start + 1, r_start + 4)
                
                pattern = [(r_start + i, c) for i in range(6)]
                pattern.extend([(cross_row, c - 1), (cross_row, c + 1)])
                
                if can_place_pattern(grid, pattern, taskvars['background_color']):
                    for pr, pc in pattern:
                        grid[pr, pc] = taskvars['object_color']
                    break
        
        def add_special_l_shape(grid, taskvars):
            """Add special L-shape: [c,0,0],[c,c,0],[c,c,c],[0,c,0]"""
            height, width = grid.shape
            for _ in range(30):
                r = random.randint(1, height-4)
                c = random.randint(0, width-3)
                
                pattern = [
                    (r, c), (r+1, c), (r+1, c+1), (r+2, c), 
                    (r+2, c+1), (r+2, c+2), (r+3, c+1)
                ]
                
                if can_place_pattern(grid, pattern, taskvars['background_color']):
                    for pr, pc in pattern:
                        grid[pr, pc] = taskvars['object_color']
                    return True
            return False
        
        def add_basic_shapes(grid, taskvars):
            """Add single cells, lines, and properly shaped objects"""
            height, width = grid.shape
            obj_color = taskvars['object_color']
            
            # Single cells (cannot fit cross)
            for _ in range(5):
                r, c = random.randint(1, height-2), random.randint(1, width-2)
                if grid[r, c] == taskvars['background_color']:
                    grid[r, c] = obj_color
            
            # Vertical lines (cannot fit cross)
            for _ in range(2):
                if height >= 5:
                    r_start = random.randint(1, height-4)
                    c = random.randint(1, width-2)
                    pattern = [(r_start + i, c) for i in range(3)]
                    if can_place_pattern(grid, pattern, taskvars['background_color']):
                        for pr, pc in pattern:
                            grid[pr, pc] = obj_color
            
            # Horizontal lines (cannot fit cross)
            for _ in range(2):
                if width >= 5:
                    r = random.randint(1, height-2)
                    c_start = random.randint(1, width-4)
                    pattern = [(r, c_start + i) for i in range(3)]
                    if can_place_pattern(grid, pattern, taskvars['background_color']):
                        for pr, pc in pattern:
                            grid[pr, pc] = obj_color
            
            # T-shapes (exactly 1 cross position)
            for _ in range(3):
                add_t_shape(grid, taskvars)
            
            # Vertical lines with cross
            for _ in range(2):
                add_vertical_line_with_cross(grid, taskvars)
            
            # Special L-shapes
            for _ in range(random.randint(3, 5)):
                add_special_l_shape(grid, taskvars)
        
        def add_bottom_shapes(grid, taskvars, start_row):
            """Add various shapes in the bottom portion"""
            height, width = grid.shape
            obj_color = taskvars['object_color']
            
            # Single cells
            for _ in range(random.randint(5, 8)):
                r, c = random.randint(start_row, height-1), random.randint(0, width-1)
                if grid[r, c] == taskvars['background_color']:
                    grid[r, c] = obj_color
            
            # Horizontal lines
            for _ in range(random.randint(3, 5)):
                if width >= 3:
                    r, c = random.randint(start_row, height-1), random.randint(0, width-3)
                    pattern = [(r, c+i) for i in range(3)]
                    if can_place_pattern(grid, pattern, taskvars['background_color']):
                        for pr, pc in pattern:
                            grid[pr, pc] = obj_color
            
            # Vertical lines
            for _ in range(random.randint(3, 5)):
                if height - start_row >= 3:
                    r, c = random.randint(start_row, height-3), random.randint(0, width-1)
                    pattern = [(r+i, c) for i in range(3)]
                    if can_place_pattern(grid, pattern, taskvars['background_color']):
                        for pr, pc in pattern:
                            grid[pr, pc] = obj_color
            
            # T-shapes and other shapes
            for _ in range(random.randint(3, 5)):
                add_t_shape(grid, taskvars)
            
            for _ in range(random.randint(2, 3)):
                add_vertical_line_with_cross(grid, taskvars)
            
            for _ in range(random.randint(3, 5)):
                add_special_l_shape(grid, taskvars)
        
        def add_elongated_crosses(grid, taskvars):
            """Add 2x2 blocks in top rows and elongated crosses in center"""
            height, width = grid.shape
            obj_color = taskvars['object_color']
            
            # 2x2 blocks in top rows
            num_blocks = random.randint(4, 6)
            for _ in range(num_blocks):
                r = random.randint(0, 1)
                c = random.randint(0, width-2)
                pattern = [(r+i, c+j) for i in range(2) for j in range(2)]
                if can_place_pattern(grid, pattern, taskvars['background_color']):
                    for pr, pc in pattern:
                        grid[pr, pc] = obj_color
            
            # Elongated crosses in center
            if height >= 15 and width >= 15:
                num_crosses = random.randint(4, 6)
                placed = 0
                for row_idx in range(2):
                    for col_idx in range(3):
                        if placed >= num_crosses:
                            break
                        
                        base_r = 4 + 3 + row_idx * 9
                        base_c = 2 + 1 + col_idx * 6
                        
                        if base_r + 5 < height and base_c + 1 < width:
                            pattern = [
                                (base_r - 1, base_c), (base_r, base_c - 1), (base_r, base_c), 
                                (base_r, base_c + 1), (base_r + 1, base_c), (base_r + 2, base_c),
                                (base_r + 3, base_c), (base_r + 4, base_c - 1), (base_r + 4, base_c),
                                (base_r + 4, base_c + 1), (base_r + 5, base_c)
                            ]
                            
                            if can_place_pattern(grid, pattern, taskvars['background_color']):
                                for pr, pc in pattern:
                                    grid[pr, pc] = obj_color
                                placed += 1
            
            add_bottom_shapes(grid, taskvars, height * 2 // 3)
        
        def add_dense_packing(grid, taskvars):
            """Add densely packed objects in bottom right corner"""
            height, width = grid.shape
            obj_color = taskvars['object_color']
            start_r, start_c = height * 2 // 3, width * 2 // 3
            
            for _ in range(12):
                shape_type = random.choice(['single', 'h_line', 'v_line', 't_shape', 'vertical_line_cross'])
                
                if shape_type == 'single':
                    r, c = random.randint(start_r, height-1), random.randint(start_c, width-1)
                    if grid[r, c] == taskvars['background_color']:
                        grid[r, c] = obj_color
                elif shape_type == 'h_line' and width - start_c >= 3:
                    r, c = random.randint(start_r, height-1), random.randint(start_c, width-3)
                    pattern = [(r, c+i) for i in range(3)]
                    if can_place_pattern(grid, pattern, taskvars['background_color']):
                        for pr, pc in pattern:
                            grid[pr, pc] = obj_color
                elif shape_type == 'v_line' and height - start_r >= 3:
                    r, c = random.randint(start_r, height-3), random.randint(start_c, width-1)
                    pattern = [(r+i, c) for i in range(3)]
                    if can_place_pattern(grid, pattern, taskvars['background_color']):
                        for pr, pc in pattern:
                            grid[pr, pc] = obj_color
                elif shape_type == 't_shape':
                    add_t_shape(grid, taskvars)
                elif shape_type == 'vertical_line_cross':
                    add_vertical_line_with_cross(grid, taskvars)
            
            for _ in range(random.randint(2, 4)):
                add_special_l_shape(grid, taskvars)
        
        def add_network_grid(grid, taskvars):
            """Add network of lines and isolated cross-fitting shapes"""
            height, width = grid.shape
            obj_color = taskvars['object_color']
            
            # Create network lines
            v_spacing = random.randint(7, 9)
            h_spacing = random.randint(7, 9)
            
            vertical_lines = []
            for c in range(v_spacing, width, v_spacing):
                if c < width:
                    for r in range(height):
                        grid[r, c] = obj_color
                    vertical_lines.append(c)
            
            horizontal_lines = []
            for r in range(h_spacing, height, h_spacing):
                if r < height:
                    for c in range(width):
                        grid[r, c] = obj_color
                    horizontal_lines.append(r)
            
            # Randomly modify lines
            for v_line in vertical_lines:
                for _ in range(random.randint(2, 3)):
                    gap_start = random.randint(1, height-4)
                    gap_length = random.randint(1, 3)
                    for r in range(gap_start, min(gap_start + gap_length, height-1)):
                        grid[r, v_line] = taskvars['background_color']
            
            for h_line in horizontal_lines:
                for _ in range(random.randint(2, 3)):
                    gap_start = random.randint(1, width-4)
                    gap_length = random.randint(1, 3)
                    for c in range(gap_start, min(gap_start + gap_length, width-1)):
                        grid[h_line, c] = taskvars['background_color']
            
            # Add isolated shapes in subgrid areas
            def find_subgrid_areas():
                areas = []
                v_boundaries = [0] + vertical_lines + [width]
                h_boundaries = [0] + horizontal_lines + [height]
                
                for i in range(len(h_boundaries) - 1):
                    for j in range(len(v_boundaries) - 1):
                        area_r_start = h_boundaries[i] + 1
                        area_r_end = h_boundaries[i + 1] - 1
                        area_c_start = v_boundaries[j] + 1
                        area_c_end = v_boundaries[j + 1] - 1
                        
                        area_h = area_r_end - area_r_start + 1
                        area_w = area_c_end - area_c_start + 1
                        
                        if area_h >= 4 and area_w >= 3:
                            empty_count = sum(1 for r in range(area_r_start, area_r_end + 1)
                                            for c in range(area_c_start, area_c_end + 1)
                                            if 0 <= r < height and 0 <= c < width 
                                            and grid[r, c] == taskvars['background_color'])
                            total_count = area_h * area_w
                            
                            if total_count > 0 and empty_count / total_count >= 0.8:
                                areas.append((area_r_start, area_c_start, area_h, area_w))
                
                return areas
            
            def add_isolated_shapes():
                subgrid_areas = find_subgrid_areas()
                max_shapes = min(6, len(subgrid_areas))
                shapes_added = 0
                
                for area in random.sample(subgrid_areas, max_shapes):
                    if shapes_added >= max_shapes:
                        break
                        
                    area_r, area_c, area_h, area_w = area
                    
                    if area_h >= 5 and area_w >= 3:
                        # T-shape
                        for _ in range(10):
                            r = random.randint(area_r + 1, area_r + area_h - 3)
                            c = random.randint(area_c + 1, area_c + area_w - 2)
                            pattern = [(r-1, c-1), (r-1, c), (r-1, c+1), (r, c), (r+1, c)]
                            
                            if is_pattern_isolated(grid, pattern, taskvars, 2):
                                for pr, pc in pattern:
                                    grid[pr, pc] = taskvars['object_color']
                                shapes_added += 1
                                break
                    elif area_h >= 4 and area_w >= 3:
                        # Special L-shape
                        for _ in range(10):
                            r = random.randint(area_r, area_r + area_h - 4)
                            c = random.randint(area_c, area_c + area_w - 3)
                            pattern = [(r, c), (r+1, c), (r+1, c+1), (r+2, c), 
                                      (r+2, c+1), (r+2, c+2), (r+3, c+1)]
                            
                            if is_pattern_isolated(grid, pattern, taskvars, 2):
                                for pr, pc in pattern:
                                    grid[pr, pc] = taskvars['object_color']
                                shapes_added += 1
                                break
                    elif area_h >= 6 and area_w >= 3:
                        # Vertical line with cross
                        for _ in range(10):
                            r_start = random.randint(area_r, area_r + area_h - 6)
                            c = random.randint(area_c + 1, area_c + area_w - 2)
                            cross_row = random.randint(r_start + 1, r_start + 4)
                            
                            pattern = [(r_start + i, c) for i in range(6)]
                            pattern.extend([(cross_row, c - 1), (cross_row, c + 1)])
                            
                            if is_pattern_isolated(grid, pattern, taskvars, 2):
                                for pr, pc in pattern:
                                    grid[pr, pc] = taskvars['object_color']
                                shapes_added += 1
                                break
            
            add_isolated_shapes()
            
            # ENSURE AT LEAST ONE CROSS-FITTABLE SHAPE IS GUARANTEED
            def ensure_cross_fittable_shape():
                """Force add at least one shape that can fit a cross in the test grid"""
                # Try to find a clear area to add a guaranteed T-shape
                for _ in range(50):  # More attempts for network grid
                    r = random.randint(3, height-4)
                    c = random.randint(3, width-4)
                    
                    # T-shape pattern
                    pattern = [(r-1, c-1), (r-1, c), (r-1, c+1), (r, c), (r+1, c)]
                    
                    # Check if we can place it (less strict for guaranteed placement)
                    can_place = True
                    for pr, pc in pattern:
                        if not (0 <= pr < height and 0 <= pc < width):
                            can_place = False
                            break
                        if grid[pr, pc] != taskvars['background_color']:
                            can_place = False
                            break
                    
                    if can_place:
                        for pr, pc in pattern:
                            grid[pr, pc] = taskvars['object_color']
                        return True
                
                # If T-shape fails, try special L-shape
                for _ in range(50):
                    r = random.randint(1, height-5)
                    c = random.randint(1, width-4)
                    
                    pattern = [
                        (r, c), (r+1, c), (r+1, c+1), (r+2, c), 
                        (r+2, c+1), (r+2, c+2), (r+3, c+1)
                    ]
                    
                    can_place = True
                    for pr, pc in pattern:
                        if not (0 <= pr < height and 0 <= pc < width):
                            can_place = False
                            break
                        if grid[pr, pc] != taskvars['background_color']:
                            can_place = False
                            break
                    
                    if can_place:
                        for pr, pc in pattern:
                            grid[pr, pc] = taskvars['object_color']
                        return True
                
                return False
            
            # Ensure at least one cross-fittable shape
            ensure_cross_fittable_shape()
        
        def add_random_objects(grid, taskvars):
            """Add random objects ensuring unique cross placement"""
            height, width = grid.shape
            obj_color = taskvars['object_color']
            
            for _ in range(random.randint(9, 15)):
                shape_type = random.choice(['single', 'h_line', 'v_line', 't_shape', 'vertical_line_cross'])
                
                if shape_type == 'single':
                    r, c = random.randint(0, height-1), random.randint(0, width-1)
                    if grid[r, c] == taskvars['background_color']:
                        grid[r, c] = obj_color
                elif shape_type == 'h_line' and width >= 3:
                    r, c = random.randint(0, height-1), random.randint(0, width-3)
                    pattern = [(r, c+i) for i in range(3)]
                    if can_place_pattern(grid, pattern, taskvars['background_color']):
                        for pr, pc in pattern:
                            grid[pr, pc] = obj_color
                elif shape_type == 'v_line' and height >= 3:
                    r, c = random.randint(0, height-3), random.randint(0, width-1)
                    pattern = [(r+i, c) for i in range(3)]
                    if can_place_pattern(grid, pattern, taskvars['background_color']):
                        for pr, pc in pattern:
                            grid[pr, pc] = obj_color
                elif shape_type == 't_shape':
                    add_t_shape(grid, taskvars)
                elif shape_type == 'vertical_line_cross':
                    add_vertical_line_with_cross(grid, taskvars)
            
            for _ in range(random.randint(3, 5)):
                add_special_l_shape(grid, taskvars)
        
        def ensure_special_l_shape(grid, taskvars):
            """Ensure at least one special L-shape is present in the grid"""
            def is_special_l_shape(obj):
                coords = obj.coords
                if len(coords) != 7:
                    return False
                
                min_r = min(r for r, c in coords)
                max_r = max(r for r, c in coords)
                min_c = min(c for r, c in coords)
                max_c = max(c for r, c in coords)
                
                if max_r - min_r != 3 or max_c - min_c != 2:
                    return False
                
                expected_pattern = {
                    (min_r, min_c), (min_r+1, min_c), (min_r+1, min_c+1),
                    (min_r+2, min_c), (min_r+2, min_c+1), (min_r+2, min_c+2),
                    (min_r+3, min_c+1)
                }
                
                return coords == expected_pattern
            
            objects = find_connected_objects(grid, diagonal_connectivity=False, 
                                           background=taskvars['background_color'])
            
            for obj in objects:
                if obj.has_color(taskvars['object_color']) and is_special_l_shape(obj):
                    return
            
            add_special_l_shape(grid, taskvars)
        
        def validate_all_objects(grid, taskvars):
            """Validate that all objects have unique cross placement"""
            def is_elongated_cross_pattern(obj):
                coords = obj.coords
                if len(coords) != 11:
                    return False
                
                min_r = min(r for r, c in coords)
                max_r = max(r for r, c in coords)
                min_c = min(c for r, c in coords)
                max_c = max(c for r, c in coords)
                
                if max_r - min_r != 6 or max_c - min_c != 2:
                    return False
                
                base_r = min_r + 1
                base_c = min_c + 1
                
                expected_pattern = {
                    (base_r - 1, base_c), (base_r, base_c - 1), (base_r, base_c),
                    (base_r, base_c + 1), (base_r + 1, base_c), (base_r + 2, base_c),
                    (base_r + 3, base_c), (base_r + 4, base_c - 1), (base_r + 4, base_c),
                    (base_r + 4, base_c + 1), (base_r + 5, base_c)
                }
                
                return coords == expected_pattern
            
            def is_vertical_line_with_cross_pattern(obj):
                coords = obj.coords
                if len(coords) != 8:
                    return False
                
                min_r = min(r for r, c in coords)
                max_r = max(r for r, c in coords)
                min_c = min(c for r, c in coords)
                max_c = max(c for r, c in coords)
                
                if max_r - min_r != 5 or max_c - min_c != 2:
                    return False
                
                for cross_row in range(min_r + 1, max_r):
                    center_c = min_c + 1
                    expected_pattern = set()
                    
                    for i in range(6):
                        expected_pattern.add((min_r + i, center_c))
                    
                    expected_pattern.add((cross_row, center_c - 1))
                    expected_pattern.add((cross_row, center_c + 1))
                    
                    if coords == expected_pattern:
                        return True
                
                return False
            
            def validate_unique_cross_placement(obj):
                if is_elongated_cross_pattern(obj) or is_vertical_line_with_cross_pattern(obj):
                    return True
                
                coords = obj.coords
                cross_positions = []
                
                for coord in coords:
                    r, c = coord
                    cross_coords = [(r-1, c), (r, c-1), (r, c), (r, c+1), (r+1, c)]
                    if all(cross_coord in coords for cross_coord in cross_coords):
                        cross_positions.append((r, c))
                
                return len(cross_positions) <= 1
            
            objects = find_connected_objects(grid, diagonal_connectivity=False, 
                                           background=taskvars['background_color'])
            
            for obj in objects:
                if obj.has_color(taskvars['object_color']):
                    if not validate_unique_cross_placement(obj):
                        for r, c in obj.coords:
                            grid[r, c] = taskvars['background_color']
        
        # Main create_input logic
        height = random.randint(15, 30)
        width = random.randint(15, 30)
        grid = np.full((height, width), taskvars['background_color'], dtype=int)
        
        example_type = gridvars.get('example_type', 'random')
        
        if example_type == 'basic_shapes':
            add_basic_shapes(grid, taskvars)
        elif example_type == 'elongated_crosses':
            add_elongated_crosses(grid, taskvars)
        elif example_type == 'dense_packing':
            add_dense_packing(grid, taskvars)
        elif example_type == 'network_grid':
            add_network_grid(grid, taskvars)
        else:
            add_random_objects(grid, taskvars)
        
        if example_type != 'network_grid':
            ensure_special_l_shape(grid, taskvars)
            validate_all_objects(grid, taskvars)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        objects = find_connected_objects(grid, diagonal_connectivity=False, 
                                       background=taskvars['background_color'])
        
        for obj in objects:
            if obj.has_color(taskvars['object_color']):
                # Find all positions where a cross can be placed within the object
                height, width = grid.shape
                coords = obj.coords
                
                for r in range(1, height-1):
                    for c in range(1, width-1):
                        cross_coords = [(r-1, c), (r, c-1), (r, c), (r, c+1), (r+1, c)]
                        if all(coord in coords for coord in cross_coords):
                            output_grid[r-1, c] = taskvars['cross_color']
                            output_grid[r, c-1] = taskvars['cross_color']
                            output_grid[r, c] = taskvars['cross_color']
                            output_grid[r, c+1] = taskvars['cross_color']
                            output_grid[r+1, c] = taskvars['cross_color']
        
        return output_grid

