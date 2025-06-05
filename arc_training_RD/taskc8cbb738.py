from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class SquarePatternGenerator(ARCTaskGenerator):
    def __init__(self):
        self.input_reasoning_chain = [
            "Input grids contain marker groups placed as neat geometric shapes:",
            "1. Corner marker group: Placed as a 2x2 square in one area",
            "2. Center marker group: Placed as a 2x2 square in another area", 
            "3. Optional edge marker groups: Placed as neat shapes (cross, line, etc.) in other areas",
            "4. Each group has a distinct geometric shape and is well-separated",
            "5. Background: All non-marker cells use the background color"
        ]
        
        transformation_reasoning_chain = [
            "Assemble the scattered marker groups into one organized square pattern:",
            "1. Take the corner marker group and place them at the corners of a square",
            "2. Take the center marker group and place them at the centers of each edge", 
            "3. Take edge marker groups and use them to fill the edges",
            "4. Create a compact square grid with all groups assembled together"
        ]
        
        super().__init__(self.input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        # Create input grid
        input_rows = 18
        input_cols = 18
        
        background_color = taskvars['background_color']
        corner_color = taskvars['corner_color']
        center_color = taskvars['center_color']
        h_edge_color = taskvars.get('h_edge_color', background_color)
        v_edge_color = taskvars.get('v_edge_color', background_color)
        has_h_markers = taskvars.get('has_h_markers', False)
        has_v_markers = taskvars.get('has_v_markers', False)
        
        # Initialize grid with background color
        grid = np.full((input_rows, input_cols), background_color, dtype=int)
        
        # Define areas for placing groups (well separated)
        areas = [
            (2, 2),    # Top-left
            (2, 12),   # Top-right  
            (10, 2),   # Bottom-left
            (10, 12)   # Bottom-right
        ]
        
        random.shuffle(areas)
        area_index = 0
        
        # Place corner marker group as 2x2 square
        if area_index < len(areas):
            r, c = areas[area_index]
            grid[r:r+2, c:c+2] = corner_color
            area_index += 1
        
        # Place center marker group as 2x2 square  
        if area_index < len(areas):
            r, c = areas[area_index]
            grid[r:r+2, c:c+2] = center_color
            area_index += 1
        
        # Place horizontal edge marker group as cross/plus shape
        if has_h_markers and area_index < len(areas):
            r, c = areas[area_index]
            # Create a cross/plus shape
            grid[r+1, c:c+3] = h_edge_color    # Horizontal bar
            grid[r:r+3, c+1] = h_edge_color    # Vertical bar  
            area_index += 1
        
        # Place vertical edge marker group as L-shape or line
        if has_v_markers and area_index < len(areas):
            r, c = areas[area_index]
            # Create an L-shape
            grid[r:r+3, c] = v_edge_color      # Vertical part
            grid[r+2, c:c+3] = v_edge_color    # Horizontal part
            area_index += 1
        
        return grid
    
    def transform_input(self, input_grid, taskvars):
        background_color = taskvars['background_color']
        corner_color = taskvars['corner_color']
        center_color = taskvars['center_color']
        h_edge_color = taskvars.get('h_edge_color', background_color)
        v_edge_color = taskvars.get('v_edge_color', background_color)
        output_size = taskvars['output_size']
        
        # Create square output grid
        output_grid = np.full((output_size, output_size), background_color, dtype=int)
        
        # Place corner markers at the four corners
        output_grid[0, 0] = corner_color
        output_grid[0, output_size-1] = corner_color
        output_grid[output_size-1, 0] = corner_color
        output_grid[output_size-1, output_size-1] = corner_color
        
        # Place center markers at edge centers (for odd sizes)
        if output_size % 2 == 1:
            center = output_size // 2
            output_grid[0, center] = center_color        # Top center
            output_grid[output_size-1, center] = center_color   # Bottom center
            output_grid[center, 0] = center_color        # Left center
            output_grid[center, output_size-1] = center_color   # Right center
        
        # Fill edges based on available edge colors
        for i in range(1, output_size - 1):
            # Skip center positions for odd sizes
            if output_size % 2 == 1 and i == output_size // 2:
                continue
                
            # Fill horizontal edges if h_edge_color exists
            if h_edge_color != background_color:
                output_grid[0, i] = h_edge_color         # Top edge
                output_grid[output_size-1, i] = h_edge_color    # Bottom edge
            
            # Fill vertical edges if v_edge_color exists
            if v_edge_color != background_color:
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
        
        # Generate training examples
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        for i in range(num_train_pairs):
            # Each grid gets individual decisions for h/v markers
            has_h_markers = random.choice([True, False])
            has_v_markers = random.choice([True, False])
            
            # Varying output sizes across grids
            output_size = random.choice([5, 7, 9])
            
            # Individual taskvars for this specific grid
            grid_taskvars = {
                'background_color': background_color,
                'corner_color': corner_color,
                'center_color': center_color,
                'has_h_markers': has_h_markers,
                'has_v_markers': has_v_markers,
                'output_size': output_size
            }
            
            # Add edge colors only if markers exist
            color_index = 2
            if has_h_markers:
                grid_taskvars['h_edge_color'] = available_colors[color_index]
                color_index += 1
            
            if has_v_markers:
                grid_taskvars['v_edge_color'] = available_colors[color_index]
            
            input_grid = self.create_input(grid_taskvars, {})
            output_grid = self.transform_input(input_grid, grid_taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test example
        test_has_h = random.choice([True, False])
        test_has_v = random.choice([True, False])
        test_output_size = random.choice([5, 7, 9])
        
        test_taskvars = {
            'background_color': background_color,
            'corner_color': corner_color,
            'center_color': center_color,
            'has_h_markers': test_has_h,
            'has_v_markers': test_has_v,
            'output_size': test_output_size
        }
        
        color_index = 2
        if test_has_h:
            test_taskvars['h_edge_color'] = available_colors[color_index]
            color_index += 1
        
        if test_has_v:
            test_taskvars['v_edge_color'] = available_colors[color_index]
        
        test_input = self.create_input(test_taskvars, {})
        test_output = self.transform_input(test_input, test_taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return {}, TrainTestData(train=train_pairs, test=test_pairs)

# Test the generator
if __name__ == "__main__":
    generator = SquarePatternGenerator()
    taskvars, train_test_data = generator.create_grids()
    ARCTaskGenerator.visualize_train_test_data(train_test_data)