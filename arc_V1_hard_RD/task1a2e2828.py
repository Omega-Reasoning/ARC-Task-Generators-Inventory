from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random

class Task1a2e2828Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "There are up to three horizontal bars (height = 1–3 cells) and up to three vertical bars (width = 1–3 cells).",
            "Among these, you will always find at least one 2-cell-tall horizontal bar and one 2-cell-wide vertical bar, plus exactly one 1-cell-thick stripe (either horizontal or vertical).",
            "All bars are spaced evenly so that wherever a horizontal and a vertical bar cross, you get a neat rectangular intersection."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is a single cell (a 1×1 grid).",
            "Identify every cell in the input where a horizontal bar and a vertical bar overlap.",
            "Among those intersection areas, pick the one with the smallest overlap region (i.e. the narrowest crossing).",
            "Color the output cell with that intersectioning color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        """Create an input grid with horizontal and vertical bars that intersect."""
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Create base grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate more bars for variety (2-3 horizontal, 2-3 vertical)
        num_horizontal = random.randint(2, 3)
        num_vertical = random.randint(2, 3)
        
        # Available colors - ensure we have enough unique colors
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        horizontal_bars = []
        vertical_bars = []
        color_index = 0
        
        # Generate horizontal bars with proper spacing
        has_2_thick_horizontal = False
        has_1_thick_global = False
        
        # Calculate good spacing for horizontal bars
        min_spacing = max(3, rows // (num_horizontal + 1))
        
        for i in range(num_horizontal):
            if not has_2_thick_horizontal:
                thickness = 2
                has_2_thick_horizontal = True
            elif not has_1_thick_global and random.random() < 0.3:
                thickness = 1
                has_1_thick_global = True
            else:
                thickness = random.choice([1, 2, 3])
                if thickness == 1 and not has_1_thick_global:
                    has_1_thick_global = True
                elif thickness == 1 and has_1_thick_global:
                    thickness = random.choice([2, 3])
            
            # Find well-spaced position
            attempts = 0
            placed = False
            
            while attempts < 100 and not placed:
                max_start_row = rows - thickness
                if max_start_row < 0:
                    break
                
                start_row = random.randint(0, max_start_row)
                
                valid = True
                for other_start, other_thickness, _ in horizontal_bars:
                    required_spacing = max(other_thickness, thickness) + min_spacing
                    if abs(start_row - other_start) < required_spacing:
                        valid = False
                        break
                
                if valid:
                    color = available_colors[color_index]
                    color_index += 1
                    horizontal_bars.append((start_row, thickness, color))
                    placed = True
                
                attempts += 1
            
            if not placed and attempts >= 100:
                for reduced_spacing in [min_spacing - 1, min_spacing - 2, 1]:
                    if reduced_spacing < 1:
                        continue
                        
                    for attempt in range(50):
                        start_row = random.randint(0, max(0, rows - thickness))
                        
                        valid = True
                        for other_start, other_thickness, _ in horizontal_bars:
                            if abs(start_row - other_start) < max(other_thickness, thickness) + reduced_spacing:
                                valid = False
                                break
                        
                        if valid:
                            color = available_colors[color_index]
                            color_index += 1
                            horizontal_bars.append((start_row, thickness, color))
                            placed = True
                            break
                    
                    if placed:
                        break
        
        # Generate vertical bars with proper spacing
        has_2_thick_vertical = False
        min_v_spacing = max(3, cols // (num_vertical + 1))
        
        for i in range(num_vertical):
            if not has_2_thick_vertical:
                thickness = 2
                has_2_thick_vertical = True
            elif not has_1_thick_global and random.random() < 0.3:
                thickness = 1
                has_1_thick_global = True
            else:
                thickness = random.choice([1, 2, 3])
                if thickness == 1 and not has_1_thick_global:
                    has_1_thick_global = True
                elif thickness == 1 and has_1_thick_global:
                    thickness = random.choice([2, 3])
            
            attempts = 0
            placed = False
            
            while attempts < 100 and not placed:
                max_start_col = cols - thickness
                if max_start_col < 0:
                    break
                
                start_col = random.randint(0, max_start_col)
                
                valid = True
                for other_start, other_thickness, _ in vertical_bars:
                    required_spacing = max(other_thickness, thickness) + min_v_spacing
                    if abs(start_col - other_start) < required_spacing:
                        valid = False
                        break
                
                if valid:
                    color = available_colors[color_index]
                    color_index += 1
                    vertical_bars.append((start_col, thickness, color))
                    placed = True
                
                attempts += 1
            
            if not placed:
                for reduced_spacing in [min_v_spacing - 1, min_v_spacing - 2, 1]:
                    if reduced_spacing < 1:
                        continue
                        
                    for attempt in range(50):
                        start_col = random.randint(0, max(0, cols - thickness))
                        
                        valid = True
                        for other_start, other_thickness, _ in vertical_bars:
                            if abs(start_col - other_start) < max(other_thickness, thickness) + reduced_spacing:
                                valid = False
                                break
                        
                        if valid:
                            color = available_colors[color_index]
                            color_index += 1
                            vertical_bars.append((start_col, thickness, color))
                            placed = True
                            break
                    
                    if placed:
                        break
        
        # Add final 1-thick stripe if needed
        if not has_1_thick_global and color_index < len(available_colors):
            if random.choice([True, False]):
                for attempt in range(30):
                    start_row = random.randint(0, rows - 1)
                    valid = True
                    for other_start, other_thickness, _ in horizontal_bars:
                        if abs(start_row - other_start) < other_thickness + 2:
                            valid = False
                            break
                    if valid:
                        color = available_colors[color_index]
                        horizontal_bars.append((start_row, 1, color))
                        break
            else:
                for attempt in range(30):
                    start_col = random.randint(0, cols - 1)
                    valid = True
                    for other_start, other_thickness, _ in vertical_bars:
                        if abs(start_col - other_start) < other_thickness + 2:
                            valid = False
                            break
                    if valid:
                        color = available_colors[color_index]
                        vertical_bars.append((start_col, 1, color))
                        break
        
        # Store for intersection analysis
        self.horizontal_bars = horizontal_bars
        self.vertical_bars = vertical_bars
        
        # Draw horizontal bars first
        for start_row, thickness, color in horizontal_bars:
            for r in range(start_row, min(start_row + thickness, rows)):
                for c in range(cols):
                    grid[r, c] = color
        
        # Draw vertical bars (they will overwrite at intersections)
        for start_col, thickness, color in vertical_bars:
            for c in range(start_col, min(start_col + thickness, cols)):
                for r in range(rows):
                    grid[r, c] = color
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input to 1x1 output with color of smallest intersection."""
        
        # Find all intersections by checking where horizontal and vertical bars cross
        intersections = []
        
        # For each horizontal bar, check intersections with vertical bars
        for h_start_row, h_thickness, h_color in self.horizontal_bars:
            h_end_row = h_start_row + h_thickness
            
            for v_start_col, v_thickness, v_color in self.vertical_bars:
                v_end_col = v_start_col + v_thickness
                
                # Check if they actually intersect
                intersection_exists = False
                for r in range(h_start_row, min(h_end_row, grid.shape[0])):
                    for c in range(v_start_col, min(v_end_col, grid.shape[1])):
                        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                            intersection_exists = True
                            break
                    if intersection_exists:
                        break
                
                if intersection_exists:
                    # Calculate intersection area
                    intersection_area = h_thickness * v_thickness
                    
                    # Get the actual color that appears in the intersection
                    # Sample from the middle of the intersection
                    sample_r = h_start_row + h_thickness // 2
                    sample_c = v_start_col + v_thickness // 2
                    
                    if (0 <= sample_r < grid.shape[0] and 0 <= sample_c < grid.shape[1]):
                        actual_color = grid[sample_r, sample_c]
                        intersections.append((intersection_area, actual_color, sample_r, sample_c))
        
        if intersections:
            # Remove duplicates and sort by area (smallest first)
            unique_intersections = {}
            for area, color, r, c in intersections:
                key = (area, color)
                if key not in unique_intersections:
                    unique_intersections[key] = (area, color, r, c)
            
            intersections = list(unique_intersections.values())
            intersections.sort(key=lambda x: (x[0], x[1]))  # Sort by area, then color
            
            return np.array([[intersections[0][1]]], dtype=int)
        
        # Enhanced fallback: Look for intersections by analyzing the grid directly
        intersection_candidates = []
        
        # Find cells that could be intersections (non-zero cells that have neighbors in both directions)
        for r in range(1, grid.shape[0] - 1):
            for c in range(1, grid.shape[1] - 1):
                if grid[r, c] != 0:
                    cell_color = grid[r, c]
                    
                    # Check if this cell has horizontal extent (left or right neighbors of same color)
                    has_horizontal = (grid[r, c-1] == cell_color or grid[r, c+1] == cell_color or
                                    any(grid[r, cc] == cell_color for cc in range(grid.shape[1]) if cc != c))
                    
                    # Check if this cell has vertical extent (up or down neighbors of same color)  
                    has_vertical = (grid[r-1, c] == cell_color or grid[r+1, c] == cell_color or
                                  any(grid[rr, c] == cell_color for rr in range(grid.shape[0]) if rr != r))
                    
                    # If it has both horizontal and vertical extent, it's likely an intersection
                    if has_horizontal and has_vertical:
                        # Calculate approximate area by counting connected cells in both directions
                        h_extent = sum(1 for cc in range(grid.shape[1]) if grid[r, cc] == cell_color)
                        v_extent = sum(1 for rr in range(grid.shape[0]) if grid[rr, c] == cell_color)
                        
                        if h_extent >= 2 and v_extent >= 2:
                            approx_area = min(h_extent, v_extent)  # Use minimum for "smallest" intersection
                            intersection_candidates.append((approx_area, cell_color, r, c))
        
        if intersection_candidates:
            # Remove duplicates and sort
            unique_candidates = {}
            for area, color, r, c in intersection_candidates:
                key = (area, color)
                if key not in unique_candidates:
                    unique_candidates[key] = (area, color, r, c)
            
            candidates = list(unique_candidates.values())
            candidates.sort(key=lambda x: (x[0], x[1]))
            return np.array([[candidates[0][1]]], dtype=int)
        
        # Final fallback
        colors = np.unique(grid)
        colors = colors[colors != 0]
        if len(colors) > 0:
            return np.array([[colors[0]]], dtype=int)
        
        return np.array([[0]], dtype=int)

    def create_grids(self):
        """Create train and test grids with consistent variables."""
        
        # Generate grid dimensions
        grid_rows = random.randint(10, 18)
        grid_cols = random.randint(10, 18)
        
        # Store task variables
        taskvars = {
            'rows': grid_rows,
            'cols': grid_cols,
        }
        
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {}  # No additional grid variables needed
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
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

