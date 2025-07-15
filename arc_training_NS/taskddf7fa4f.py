from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class ColoredCellRectangleGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "In each grid, three randomly selected cells in the first row are each filled with a randomly chosen color.",
            "The remaining part of the grid (excluding the first row) contains three distinct rectangles, each corresponding to one of the colored cells in the first row.",
            "Each colored cell corresponds to exactly one rectangle, and each rectangle corresponds to exactly one colored cell.",
            "The rectangle corresponding to a colored cell is positioned such that its column range includes the column of the colored cell.",
            "The rectangles have different sizes and orientations.",
            "None of the rectangles touches any of the colored cells in the first row.",
            "The rectangles are separated and do not touch each other.",
            "All rectangles in the input grid are of the color {color('color_rectangle')}."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All colored cells in the first row and the three rectangles in the remaining grid are identified.",
            "For each colored cell: Its corresponding rectangle is found by checking which rectangle spans the same column as the cell. All cells within that rectangle are then filled with the same color as its corresponding colored cell.",
            "All other cells remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        
        def generate_valid_grid():
            grid = np.zeros((n, n), dtype=int)
            
            # Step 1: Randomly select 3 different colors for the first row cells
            available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            # Remove rectangle color from available colors
            available_colors = [c for c in available_colors if c != taskvars['color_rectangle']]
            
            if len(available_colors) < 3:
                return None  # Not enough colors available
            
            colors = random.sample(available_colors, 3)
            cols = random.sample(range(n), 3)
            
            for i, col in enumerate(cols):
                grid[0, col] = colors[i]
            
            # Step 2: Create rectangles for each colored cell
            rectangles = []
            rect_color = taskvars['color_rectangle']  # Use single color for all rectangles
            
            for i, (col, corresponding_color) in enumerate(zip(cols, colors)):
                # Generate rectangle that spans the column of the colored cell
                max_attempts = 500
                for attempt in range(max_attempts):
                    # Rectangle dimensions - keep them smaller to ensure separation
                    width = random.randint(2, min(4, n // 5))
                    height = random.randint(2, min(4, n // 5))
                    
                    # Rectangle position - must include the column of colored cell
                    min_start_col = max(0, col - width + 1)
                    max_start_col = min(n - width, col)
                    
                    if min_start_col <= max_start_col:
                        start_col = random.randint(min_start_col, max_start_col)
                        # Ensure rectangle doesn't touch first row
                        min_start_row = 2
                        max_start_row = n - height
                        
                        if min_start_row <= max_start_row:
                            start_row = random.randint(min_start_row, max_start_row)
                            
                            rect = {
                                'start_row': start_row,
                                'end_row': start_row + height,
                                'start_col': start_col,
                                'end_col': start_col + width,
                                'corresponding_color': corresponding_color,
                                'rect_color': rect_color
                            }
                            
                            # Check if rectangle is completely separated from existing rectangles
                            valid = True
                            for existing in rectangles:
                                if not self._rectangles_completely_separated(rect, existing):
                                    valid = False
                                    break
                            
                            if valid:
                                rectangles.append(rect)
                                break
                else:
                    return None  # Failed to place rectangle
            
            if len(rectangles) != 3:
                return None
            
            # Verify each colored cell corresponds to exactly one rectangle
            for col_idx, col in enumerate(cols):
                corresponding_rects = []
                for rect in rectangles:
                    if rect['start_col'] <= col < rect['end_col']:
                        corresponding_rects.append(rect)
                
                if len(corresponding_rects) != 1:
                    return None  # Should be exactly one rectangle per colored cell
            
            # Place rectangles in grid using the single rectangle color
            for rect in rectangles:
                for r in range(rect['start_row'], rect['end_row']):
                    for c in range(rect['start_col'], rect['end_col']):
                        grid[r, c] = rect_color
            
            return grid
        
        return retry(generate_valid_grid, lambda x: x is not None)
    
    def _rectangles_completely_separated(self, rect1: Dict, rect2: Dict) -> bool:
        """Check if two rectangles are completely separated (do not touch at all)"""
        
        # Get the actual boundaries (exclusive end coordinates)
        r1_top, r1_bottom = rect1['start_row'], rect1['end_row']
        r1_left, r1_right = rect1['start_col'], rect1['end_col']
        r2_top, r2_bottom = rect2['start_row'], rect2['end_row']
        r2_left, r2_right = rect2['start_col'], rect2['end_col']
        
        # Check if rectangles are completely separated with no touching
        # For complete separation, there must be at least one empty cell between them
        
        # Check if they're separated vertically
        vertical_separation = (r1_bottom < r2_top) or (r2_bottom < r1_top)
        
        # Check if they're separated horizontally  
        horizontal_separation = (r1_right < r2_left) or (r2_right < r1_left)
        
        # If separated in either dimension, they don't touch
        if vertical_separation or horizontal_separation:
            return True
        
        # If we reach here, they would touch or overlap, which is not allowed
        return False
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        
        # Find colored cells in first row
        colored_cells = []
        for c in range(grid.shape[1]):
            if grid[0, c] != 0:
                colored_cells.append((0, c, grid[0, c]))
        
        # Find rectangles (connected components excluding first row)
        remaining_grid = grid[1:, :]  # Exclude first row
        objects = find_connected_objects(remaining_grid, diagonal_connectivity=False, background=0)
        
        # For each colored cell, find its corresponding rectangle and fill it
        for cell_row, cell_col, cell_color in colored_cells:
            # Find rectangle that spans this column
            for obj in objects:
                # Convert object coordinates back to full grid coordinates
                obj_coords = [(r + 1, c) for r, c, _ in obj.cells]  # Add 1 to row because we excluded first row
                cols_in_obj = set(c for r, c in obj_coords)
                
                if cell_col in cols_in_obj:
                    # Fill this rectangle with the cell's color
                    for r, c in obj_coords:
                        output_grid[r, c] = cell_color
                    break
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables - only grid size and rectangle color are fixed
        taskvars = {
            'n': random.randint(18, 28),  # Larger grid to accommodate non-touching constraint
            'color_rectangle': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        # Create train and test examples
        num_train = random.randint(3, 6)
        train_data = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}

