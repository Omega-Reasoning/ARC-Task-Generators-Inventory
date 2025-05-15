from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class SquarePatternGenerator(ARCTaskGenerator):
    def __init__(self):
        self.input_reasoning_chain = [
            "Input grids can be any size.",
            "The grid contains markers in specific positions with different colors:",
            "1. Edge markers: Exactly 4 markers defining the corners of a square",
            "2. Center markers: Exactly 4 cells placed at the center of each edge of the square",
            "3. Optional markers: May contain horizontal (2 top, 2 bottom) or vertical (2 left, 2 right) markers",
            "4. Background: All non-marker cells use the background color"
        ]
        
        self.transformation_reasoning_chain = [
            "The output grid is square, with size determined by the edge markers.",
            "1. Edge markers: Copy the corner colors to the output grid corners",
            "2. Center markers: Place center marker color at the middle of each edge",
            "3. If horizontal/vertical markers exist:",
            "   - Fill the edges with their respective colors",
            "4. If no horizontal/vertical markers:",
            "   - Fill those edges with background color",
            "5. Fill all remaining inner cells with background color"
        ]
        
        taskvars_definitions = {
            "background_color": "The background color (0-9)",
            "edge_color": "The color (1-9) of the edge markers",
            "center_color": "The color (1-9) of the center markers",
            "h_edge_color": "Optional color (1-9) of horizontal markers",
            "v_edge_color": "Optional color (1-9) of vertical markers",
            "has_hv_markers": "Boolean indicating if horizontal/vertical markers exist"
        }
        
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)
    
    def create_input(self, gridvars=None):
        if gridvars is None:
            gridvars = {}
        
        # Define colors
        background_color = gridvars.get('background_color', random.randint(0, 3))
        corner_color = gridvars.get('corner_color', random.choice([i for i in range(4, 10) if i != background_color]))  # Changed from edge_color
        center_color = gridvars.get('center_color', random.choice([i for i in range(4, 10) if i not in [background_color, corner_color]]))
        
        # Decide if we'll have horizontal/vertical markers (50% chance)
        has_hv_markers = gridvars.get('has_hv_markers', random.choice([True, False]))
        
        if has_hv_markers:
            # Ensure h/v colors are different from corner color
            h_edge_color = gridvars.get('h_edge_color', random.choice([i for i in range(4, 10) 
                                       if i not in [background_color, corner_color, center_color]]))
            v_edge_color = gridvars.get('v_edge_color', random.choice([i for i in range(4, 10) 
                                       if i not in [background_color, corner_color, center_color, h_edge_color]]))
        else:
            h_edge_color = background_color
            v_edge_color = background_color
        
        # Create grid with larger size to ensure spacing
        output_size = gridvars.get('output_size', random.choice([7, 9, 11, 13, 15]))
        input_rows = random.randint(output_size + 2, output_size + 5)
        input_cols = random.randint(output_size + 2, output_size + 5)
        
        grid = np.full((input_rows, input_cols), background_color, dtype=int)
        
        # Place edge markers (corners)
        min_row = random.randint(1, input_rows - output_size - 1)
        min_col = random.randint(1, input_cols - output_size - 1)
        max_row = min_row + output_size - 1
        max_col = min_col + output_size - 1
        
        # Place corner markers
        corner_positions = [(min_row, min_col), (min_row, max_col), 
                           (max_row, min_col), (max_row, max_col)]
        for r, c in corner_positions:
            grid[r, c] = corner_color  # Changed from edge_color
        
        # Place exactly 4 center markers at middle points (no series)
        mid_row = min_row + output_size // 2
        mid_col = min_col + output_size // 2
        
        # Calculate exact center positions for each edge
        center_positions = [
            (min_row, min_col + (output_size - 1) // 2),     # Top middle exact
            (max_row, min_col + (output_size - 1) // 2),     # Bottom middle exact
            (min_row + (output_size - 1) // 2, min_col),     # Left middle exact
            (min_row + (output_size - 1) // 2, max_col)      # Right middle exact
        ]
        
        # Place exactly 4 center markers, no additional ones
        for r, c in center_positions:
            grid[r, c] = center_color
            # Ensure no adjacent cells get colored to prevent series formation
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in center_positions:  # Only keep the exact center positions
                    continue
        
        # If using h/v markers, place them with proper spacing
        if has_hv_markers:
            third = output_size // 3
            h_positions = [
                (min_row, min_col + third),
                (min_row, min_col + 2*third),
                (max_row, min_col + third),
                (max_row, min_col + 2*third)
            ]
            v_positions = [
                (min_row + third, min_col),
                (min_row + 2*third, min_col),
                (min_row + third, max_col),
                (min_row + 2*third, max_col)
            ]
            for r, c in h_positions:
                grid[r, c] = h_edge_color
            for r, c in v_positions:
                grid[r, c] = v_edge_color

        gridvars.update({
            'background_color': background_color,
            'corner_color': corner_color,  # Changed from edge_color
            'center_color': center_color,
            'h_edge_color': h_edge_color,
            'v_edge_color': v_edge_color,
            'has_hv_markers': has_hv_markers,
            'output_size': output_size
        })
        
        return grid, gridvars
    
    def transform_input(self, input_grid, gridvars=None):
        # Get grid dimensions
        rows, cols = input_grid.shape
        
        if gridvars is None:
            # If no gridvars provided, we need to infer them from the input grid
            objects = find_connected_objects(input_grid, diagonal_connectivity=False)
            
            # Find all unique colors in the grid
            all_colors = set()
            for obj in objects:
                all_colors.update(obj.colors)
                
            # The background color is the most common color
            background_color = np.bincount(input_grid.flatten()).argmax()
            all_colors.discard(background_color)
            
            # Find corner positions by looking for the four outermost colored cells
            colored_positions = []
            for r in range(input_grid.shape[0]):
                for c in range(input_grid.shape[1]):
                    if input_grid[r, c] != background_color:
                        colored_positions.append((r, c))
            
            # Sort by row and column to find extremes
            sorted_by_row = sorted(colored_positions, key=lambda pos: pos[0])
            sorted_by_col = sorted(colored_positions, key=lambda pos: pos[1])
            
            min_row = sorted_by_row[0][0]
            max_row = sorted_by_row[-1][0]
            min_col = sorted_by_col[0][1]
            max_col = sorted_by_col[-1][1]
            
            corner_positions = [
                (min_row, min_col),  # Top-left
                (min_row, max_col),  # Top-right
                (max_row, min_col),  # Bottom-left
                (max_row, max_col)   # Bottom-right
            ]
            
            # Get corner color from actual input grid corner position
            corner_color = input_grid[min_row, min_col]  # Use actual color from input
            
            # Calculate output size
            output_size = max(max_row - min_row + 1, max_col - min_col + 1)
            
            # Find horizontal edge markers
            h_edge_positions = []
            for c in range(min_col + 1, max_col):
                if input_grid[min_row, c] != background_color and input_grid[min_row, c] != corner_color:
                    h_edge_positions.append((min_row, c))
                if input_grid[max_row, c] != background_color and input_grid[max_row, c] != corner_color:
                    h_edge_positions.append((max_row, c))
            
            # Find vertical edge markers
            v_edge_positions = []
            for r in range(min_row + 1, max_row):
                if input_grid[r, min_col] != background_color and input_grid[r, min_col] != corner_color:
                    v_edge_positions.append((r, min_col))
                if input_grid[r, max_col] != background_color and input_grid[r, max_col] != corner_color:
                    v_edge_positions.append((r, max_col))
            
            # Get edge colors if they exist
            h_edge_color = input_grid[h_edge_positions[0]] if h_edge_positions else None
            v_edge_color = input_grid[v_edge_positions[0]] if v_edge_positions else None
            
            # Find center edge markers (exactly 4)
            center_positions = []
            mid_row = (min_row + max_row) // 2
            mid_col = (min_col + max_col) // 2
            
            # Check middle positions on all four edges
            center_positions = [
                (min_row, mid_col),    # Top edge center
                (max_row, mid_col),    # Bottom edge center
                (mid_row, min_col),    # Left edge center
                (mid_row, max_col)     # Right edge center
            ]
            center_color = input_grid[center_positions[0]]  # All center markers have same color
            
            # Check if horizontal/vertical markers exist
            has_hv_markers = False
            h_edge_color = v_edge_color = background_color
            
            for r, c in h_edge_positions:
                if input_grid[r, c] != background_color and input_grid[r, c] != corner_color:
                    has_hv_markers = True
                    h_edge_color = input_grid[r, c]
                    break
                    
            for r, c in v_edge_positions:
                if input_grid[r, c] != background_color and input_grid[r, c] != corner_color:
                    has_hv_markers = True
                    v_edge_color = input_grid[r, c]
                    break
            
            gridvars = {
                'background_color': background_color,
                'corner_color': corner_color,
                'h_edge_color': h_edge_color,
                'v_edge_color': v_edge_color,
                'center_color': center_color,
                'output_size': output_size,
                'corner_positions': corner_positions
            }
        else:
            # Use the output_size from gridvars
            output_size = gridvars['output_size']
            background_color = gridvars['background_color']
            corner_color = gridvars['corner_color']
            center_color = gridvars['center_color']
            h_edge_color = gridvars['h_edge_color']
            v_edge_color = gridvars['v_edge_color']
        
        # Make sure output_size is odd for true center
        if output_size % 2 == 0:
            output_size += 1
    
        # Create square output grid
        output_grid = np.full((output_size, output_size), background_color, dtype=int)
        
        # Calculate true center
        exact_center = (output_size - 1) // 2
        
        # Fill corners
        output_grid[0, 0] = corner_color
        output_grid[0, -1] = corner_color
        output_grid[-1, 0] = corner_color
        output_grid[-1, -1] = corner_color
        
        # Fill exact center positions
        output_grid[0, exact_center] = center_color      # Top
        output_grid[-1, exact_center] = center_color     # Bottom
        output_grid[exact_center, 0] = center_color      # Left
        output_grid[exact_center, -1] = center_color     # Right
        
        # Fill edges based on h/v markers
        for i in range(1, output_size - 1):
            if i != exact_center:  # Skip center positions
                output_grid[0, i] = h_edge_color    # Top edge
                output_grid[-1, i] = h_edge_color   # Bottom edge
                output_grid[i, 0] = v_edge_color    # Left edge
                output_grid[i, -1] = v_edge_color   # Right edge
        
        return output_grid
    
    def create_grids(self):
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train_pairs):
            # Ensure different colors for each type of marker
            available_colors = list(range(4, 10))
            random.shuffle(available_colors)
            background_color = random.randint(0, 3)
            
            # Always use different colors for corners and center
            corner_color = available_colors[0]
            center_color = available_colors[1]
            
            has_hv_markers = random.choice([True, False])
            if has_hv_markers:
                h_edge_color = available_colors[2]  # Different from corner color
                v_edge_color = available_colors[3]  # Different from corner color
            else:
                h_edge_color = background_color
                v_edge_color = background_color
            
            gridvars = {
                'background_color': background_color,
                'corner_color': corner_color,
                'center_color': center_color,
                'h_edge_color': h_edge_color,
                'v_edge_color': v_edge_color,
                'has_hv_markers': has_hv_markers,
                'output_size': random.choice([7, 9, 11, 13, 15])  # Only odd sizes for true center
            }
            
            input_grid, updated_gridvars = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid, updated_gridvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair with different colors but same rules
        available_colors = list(range(4, 10))
        random.shuffle(available_colors)
        background_color = random.randint(0, 3)
        
        test_gridvars = {
            'background_color': background_color,
            'corner_color': available_colors[0],
            'center_color': available_colors[1],
            'has_hv_markers': random.choice([True, False])
        }
        
        # Set h/v colors based on has_hv_markers
        if test_gridvars['has_hv_markers']:
            test_gridvars.update({
                'h_edge_color': available_colors[2],  # Different from corner and center
                'v_edge_color': available_colors[3]   # Different from all others
            })
        else:
            test_gridvars.update({
                'h_edge_color': background_color,
                'v_edge_color': background_color
            })
        
        test_gridvars['output_size'] = random.randint(7, 15)
        
        test_input, _ = self.create_input(test_gridvars)
        test_output = self.transform_input(test_input, test_gridvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)

