from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class Taskc8cbb738Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids contain marker groups placed as neat geometric shapes.",
            "The corner marker group are placed as a 2x2 square in one area",
            "The center marker group are placed as a 2x2 square in another area", 
            "The optional edge marker groups are placed as neat shapes (cross, line, etc.) in other areas",
            "Each group has a distinct geometric shape and is well-separated",
            "The background is that all non-marker cells use the background color"
        ]
        
        transformation_reasoning_chain = [
            "Assemble the scattered marker groups into one organized square pattern.",
            "Firstly, take the corner marker group and place them at the corners of a square",
            "Secondly, take the center marker group and place them at the centers of each edge", 
            "Moving on, take edge marker groups and use them to fill the edges",
            "Finally, Create a compact square grid with all groups assembled together"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars):
        # Create input grid - make it large enough to accommodate all groups
        input_rows = 25
        input_cols = 25
        
        background_color = taskvars['background_color']
        corner_color = taskvars['corner_color']
        center_color = taskvars['center_color']
        h_edge_color = taskvars.get('h_edge_color', background_color)
        v_edge_color = taskvars.get('v_edge_color', background_color)
        has_h_markers = taskvars.get('has_h_markers', False)
        has_v_markers = taskvars.get('has_v_markers', False)
        output_size = taskvars['output_size']
        
        # Initialize grid with background color
        grid = np.full((input_rows, input_cols), background_color, dtype=int)
        
        # Calculate areas for each group - well separated and non-overlapping
        areas = []
        
        # Top area for corner group
        areas.append((2, 2))
        
        # Right area for center group (if needed)
        if output_size % 2 == 1:
            areas.append((2, 12))
        
        # Bottom-left area for h-edge group
        if has_h_markers:
            areas.append((12, 2))
        
        # Bottom-right area for v-edge group  
        if has_v_markers:
            areas.append((12, 12))
        
        area_index = 0
        
        # 1. Create corner marker group - EXACT spacing as output corners
        if area_index < len(areas):
            base_r, base_c = areas[area_index]
            # Place corners with EXACT spacing as they will appear in output
            grid[base_r, base_c] = corner_color                                    # Top-left corner
            grid[base_r, base_c + (output_size - 1)] = corner_color               # Top-right corner  
            grid[base_r + (output_size - 1), base_c] = corner_color               # Bottom-left corner
            grid[base_r + (output_size - 1), base_c + (output_size - 1)] = corner_color  # Bottom-right corner
            area_index += 1
        
        # 2. Create center marker group - EXACT spacing as output edge centers
        if output_size % 2 == 1 and area_index < len(areas):
            base_r, base_c = areas[area_index]
            center_offset = output_size // 2
            # Place edge centers with EXACT spacing
            grid[base_r, base_c + center_offset] = center_color                    # Top center
            grid[base_r + (output_size - 1), base_c + center_offset] = center_color  # Bottom center
            grid[base_r + center_offset, base_c] = center_color                    # Left center
            grid[base_r + center_offset, base_c + (output_size - 1)] = center_color  # Right center
            area_index += 1
        
        # 3. Create horizontal edge marker group - EXACT layout as output edges
        if has_h_markers and area_index < len(areas):
            base_r, base_c = areas[area_index]
            # Create the EXACT horizontal edge pattern
            for i in range(1, output_size - 1):
                # Skip center position for odd sizes (already occupied by center markers)
                if output_size % 2 == 1 and i == output_size // 2:
                    continue
                # Top edge
                grid[base_r, base_c + i] = h_edge_color
                # Bottom edge  
                grid[base_r + (output_size - 1), base_c + i] = h_edge_color
            area_index += 1
        
        # 4. Create vertical edge marker group - EXACT layout as output edges
        if has_v_markers and area_index < len(areas):
            base_r, base_c = areas[area_index]
            # Create the EXACT vertical edge pattern
            for i in range(1, output_size - 1):
                # Skip center position for odd sizes (already occupied by center markers)
                if output_size % 2 == 1 and i == output_size // 2:
                    continue
                # Left edge
                grid[base_r + i, base_c] = v_edge_color
                # Right edge
                grid[base_r + i, base_c + (output_size - 1)] = v_edge_color
            area_index += 1
        
        return grid
    
    def transform_input(self, grid, taskvars):
        background_color = taskvars['background_color']
        corner_color = taskvars['corner_color']
        center_color = taskvars['center_color']
        h_edge_color = taskvars.get('h_edge_color', background_color)
        v_edge_color = taskvars.get('v_edge_color', background_color)
        output_size = taskvars['output_size']
        
        # Create square output grid
        output_grid = np.full((output_size, output_size), background_color, dtype=int)
        
        # Find all marker groups from input using connected components
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=background_color)
        
        # Identify groups by color and place them exactly as they were arranged in input
        has_corner_group = False
        has_center_group = False
        has_h_edge_group = False
        has_v_edge_group = False
        
        for obj in objects:
            if obj.has_color(corner_color):
                has_corner_group = True
            elif obj.has_color(center_color):
                has_center_group = True
            elif obj.has_color(h_edge_color) and h_edge_color != background_color:
                has_h_edge_group = True
            elif obj.has_color(v_edge_color) and v_edge_color != background_color:
                has_v_edge_group = True
        
        # Place corner markers at the four corners
        if has_corner_group:
            output_grid[0, 0] = corner_color
            output_grid[0, output_size-1] = corner_color
            output_grid[output_size-1, 0] = corner_color
            output_grid[output_size-1, output_size-1] = corner_color
        
        # Place center markers at edge centers (for odd sizes)
        if has_center_group and output_size % 2 == 1:
            center = output_size // 2
            output_grid[0, center] = center_color        # Top center
            output_grid[output_size-1, center] = center_color   # Bottom center
            output_grid[center, 0] = center_color        # Left center
            output_grid[center, output_size-1] = center_color   # Right center
        
        # Fill horizontal edges
        if has_h_edge_group:
            for i in range(1, output_size - 1):
                # Skip center positions for odd sizes
                if output_size % 2 == 1 and i == output_size // 2:
                    continue
                output_grid[0, i] = h_edge_color         # Top edge
                output_grid[output_size-1, i] = h_edge_color    # Bottom edge
        
        # Fill vertical edges
        if has_v_edge_group:
            for i in range(1, output_size - 1):
                # Skip center positions for odd sizes
                if output_size % 2 == 1 and i == output_size // 2:
                    continue
                output_grid[i, 0] = v_edge_color         # Left edge
                output_grid[i, output_size-1] = v_edge_color    # Right edge
        
        return output_grid
    
    def create_grids(self):
        # Choose base colors
        background_color = random.randint(0, 2)
        available_colors = [c for c in range(1, 10) if c != background_color]
        random.shuffle(available_colors)
        
        corner_color = available_colors[0]
        center_color = available_colors[1]
        
        # Shared taskvars for consistent behavior
        taskvars = {
            'background_color': background_color,
            'corner_color': corner_color,
            'center_color': center_color,
        }
        
        # Generate training examples
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        for i in range(num_train_pairs):
            # Each grid gets individual decisions
            has_h_markers = random.choice([True, False])
            has_v_markers = random.choice([True, False])
            output_size = random.choice([5, 7, 9])
            
            # Individual taskvars for this specific grid
            grid_taskvars = taskvars.copy()
            grid_taskvars.update({
                'has_h_markers': has_h_markers,
                'has_v_markers': has_v_markers,
                'output_size': output_size
            })
            
            # Add edge colors only if markers exist
            color_index = 2
            if has_h_markers:
                grid_taskvars['h_edge_color'] = available_colors[color_index]
                color_index += 1
            
            if has_v_markers and color_index < len(available_colors):
                grid_taskvars['v_edge_color'] = available_colors[color_index]
            
            input_grid = self.create_input(grid_taskvars)
            output_grid = self.transform_input(input_grid, grid_taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test example
        test_has_h = random.choice([True, False])
        test_has_v = random.choice([True, False])
        test_output_size = random.choice([5, 7, 9])
        
        test_taskvars = taskvars.copy()
        test_taskvars.update({
            'has_h_markers': test_has_h,
            'has_v_markers': test_has_v,
            'output_size': test_output_size
        })
        
        color_index = 2
        if test_has_h:
            test_taskvars['h_edge_color'] = available_colors[color_index]
            color_index += 1
        
        if test_has_v and color_index < len(available_colors):
            test_taskvars['v_edge_color'] = available_colors[color_index]
        
        test_input = self.create_input(test_taskvars)
        test_output = self.transform_input(test_input, test_taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)

