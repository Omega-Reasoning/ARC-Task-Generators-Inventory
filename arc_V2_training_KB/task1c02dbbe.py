from arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task1c02dbbe(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",

            "Each input grid contains a large {color('object_color')} rectangular block placed approximately at the center of the grid, occupying at least two-thirds of the grid cells. This block is surrounded by 1, 2, or 3 empty rows or columns.",
            
            "Additionally, three marked cells are used to indicate the corners of sub-rectangles that will appear within the main rectangle in the output grid. These marks define the boundaries of a smaller sub-rectangle: one cell corresponds to a corner of the main rectangle, while the other two are placed on the first outer layer of the rectangle—one defining its width and the other its length.",
            
            "To define a sub-rectangle of size m x n (where m = number of rows, n = number of columns) to be placed in the top-left corner: Fill the top-left corner cell of the {color('object_color')} rectangular block with the desired sub-rectangle color. Add one cell in the row above the first row of the {color('object_color')} rectangle, aligned with column index (n - 1). Add one cell in the column to the left of the first column of the {color('object_color')} rectangle, aligned with row index (m - 1). These three marked cells together specify the three corners of the sub-rectangle.",
            
            "Similarly, more cells can be added to define additional sub-rectangles. Since the {color('object_color')} rectangle has four corners, at most four sub-rectangles can be placed. It must be ensured that the length and width of each sub-rectangle are properly marked so that no sub-rectangles overlap.",
            
            "There can be 1, 2, 3, or 4 sub-rectangles in the output. Accordingly, 3, 6, 9, or 12 marked cells are required to define these sub-rectangles.",
            
            "Each sub-rectangle must have a distinct color. Therefore, the 3 cells marking a single sub-rectangle must share the same color, while different sub-rectangles are marked with different colors."
        ]

        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids, identifying the {color('object_color')} rectangular block, and all the single colored cells that define the three corners of the sub-rectangles to be formed in the output.",
            
            "Each sub-rectangle is specified by three marked cells: one of the corner cells of the {color('object_color')} rectangular block (which will also be one of the corner cells of the sub-rectangle), and two additional cells placed on the outer boundary of the {color('object_color')} rectangle.",
            
            "These three marks are then used to complete and fill the corresponding sub-rectangles in the output.",
            
            "If a sub-rectangle of size m x n (with m rows and n columns) is to be placed in the top-left corner, then there will be three defining cells: one in the top-left corner cell of the {color('object_color')} rectangular block, filled with the desired sub-rectangle color; one in the row above the first row of the {color('object_color')} rectangle, aligned with column index (n - 1); and one in the column to the left of the first column of the {color('object_color')} rectangle, aligned with row index (m - 1).",
            
            "Note that two of the marked cells are placed outside the {color('object_color')} rectangular block, but the resulting sub-rectangle must lie entirely within the {color('object_color')} rectangle.",
            
            "Fill all the cells within the defined boundaries to form the sub-rectangle of size m x n using the same color.",
            
            "Repeat this process for all sub-rectangles.",
            "Finally, remove all the marked cells which are outside the {color('object_color')} rectangular block (set them back to the background color, 0)."
        ]

        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Calculate rectangle dimensions (at least 2/3 of grid)
        min_rect_size = int(grid_size * 2/3)
        max_rect_size = grid_size - 2  # Leave space for markers
        
        rect_height = random.randint(min_rect_size, max_rect_size)
        rect_width = random.randint(min_rect_size, max_rect_size)
        
        # Calculate position to center the rectangle
        start_row = (grid_size - rect_height) // 2
        start_col = (grid_size - rect_width) // 2
        
        # Ensure we have space for markers (at least 1 empty row/column around)
        start_row = max(1, start_row)
        start_col = max(1, start_col)
        start_row = min(start_row, grid_size - rect_height - 1)
        start_col = min(start_col, grid_size - rect_width - 1)
        
        # Fill the main rectangle
        grid[start_row:start_row+rect_height, start_col:start_col+rect_width] = object_color
        
        # Store rectangle bounds for marker placement
        rect_bounds = {
            'top': start_row,
            'bottom': start_row + rect_height - 1,
            'left': start_col,
            'right': start_col + rect_width - 1
        }
        
        # Get sub-rectangles from gridvars
        sub_rectangles = gridvars.get('sub_rectangles', [])
        
        # Place markers for each sub-rectangle
        for sub_rect in sub_rectangles:
            corner = sub_rect['corner']
            width = sub_rect['width']
            height = sub_rect['height']
            color = sub_rect['color']
            
            # Determine corner position
            if corner == 'top_left':
                corner_row, corner_col = rect_bounds['top'], rect_bounds['left']
                width_marker_row = rect_bounds['top'] - 1
                width_marker_col = rect_bounds['left'] + width - 1
                height_marker_row = rect_bounds['top'] + height - 1
                height_marker_col = rect_bounds['left'] - 1
            elif corner == 'top_right':
                corner_row, corner_col = rect_bounds['top'], rect_bounds['right']
                width_marker_row = rect_bounds['top'] - 1
                width_marker_col = rect_bounds['right'] - width + 1
                height_marker_row = rect_bounds['top'] + height - 1
                height_marker_col = rect_bounds['right'] + 1
            elif corner == 'bottom_left':
                corner_row, corner_col = rect_bounds['bottom'], rect_bounds['left']
                width_marker_row = rect_bounds['bottom'] + 1
                width_marker_col = rect_bounds['left'] + width - 1
                height_marker_row = rect_bounds['bottom'] - height + 1
                height_marker_col = rect_bounds['left'] - 1
            else:  # bottom_right
                corner_row, corner_col = rect_bounds['bottom'], rect_bounds['right']
                width_marker_row = rect_bounds['bottom'] + 1
                width_marker_col = rect_bounds['right'] - width + 1
                height_marker_row = rect_bounds['bottom'] - height + 1
                height_marker_col = rect_bounds['right'] + 1
            
            # Place the three marker cells
            grid[corner_row, corner_col] = color
            if 0 <= width_marker_row < grid_size and 0 <= width_marker_col < grid_size:
                grid[width_marker_row, width_marker_col] = color
            if 0 <= height_marker_row < grid_size and 0 <= height_marker_col < grid_size:
                grid[height_marker_row, height_marker_col] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        object_color = taskvars['object_color']
        output_grid = grid.copy()
        
        # Find the main rectangle bounds
        object_cells = np.where(grid == object_color)
        if len(object_cells[0]) == 0:
            return output_grid
            
        rect_top = np.min(object_cells[0])
        rect_bottom = np.max(object_cells[0])
        rect_left = np.min(object_cells[1])
        rect_right = np.max(object_cells[1])
        
        # Find all marker colors (non-zero, non-object_color)
        unique_colors = np.unique(grid)
        marker_colors = [c for c in unique_colors if c != 0 and c != object_color]
        
        # Process each marker color (represents one sub-rectangle)
        external_markers_to_remove = []  # Track external markers to remove later
        
        for color in marker_colors:
            marker_cells = np.where(grid == color)
            if len(marker_cells[0]) != 3:
                continue  # Should have exactly 3 markers per sub-rectangle
            
            marker_positions = list(zip(marker_cells[0], marker_cells[1]))
            
            # Find which marker is the corner (inside the main rectangle)
            corner_pos = None
            external_markers = []
            
            for pos in marker_positions:
                r, c = pos
                if rect_top <= r <= rect_bottom and rect_left <= c <= rect_right:
                    corner_pos = pos
                else:
                    external_markers.append(pos)
            
            if corner_pos is None or len(external_markers) != 2:
                continue
            
            # Store external markers for removal
            external_markers_to_remove.extend(external_markers)
            
            corner_r, corner_c = corner_pos
            
            # Determine which corner this is and calculate sub-rectangle bounds
            if corner_r == rect_top and corner_c == rect_left:
                # Top-left corner
                width_marker = None
                height_marker = None
                for r, c in external_markers:
                    if r == rect_top - 1:
                        width_marker = (r, c)
                    elif c == rect_left - 1:
                        height_marker = (r, c)
                
                if width_marker and height_marker:
                    width = width_marker[1] - rect_left + 1
                    height = height_marker[0] - rect_top + 1
                    # Fill sub-rectangle
                    for r in range(rect_top, rect_top + height):
                        for c in range(rect_left, rect_left + width):
                            if rect_top <= r <= rect_bottom and rect_left <= c <= rect_right:
                                output_grid[r, c] = color
                                
            elif corner_r == rect_top and corner_c == rect_right:
                # Top-right corner
                width_marker = None
                height_marker = None
                for r, c in external_markers:
                    if r == rect_top - 1:
                        width_marker = (r, c)
                    elif c == rect_right + 1:
                        height_marker = (r, c)
                
                if width_marker and height_marker:
                    width = rect_right - width_marker[1] + 1
                    height = height_marker[0] - rect_top + 1
                    # Fill sub-rectangle
                    for r in range(rect_top, rect_top + height):
                        for c in range(rect_right - width + 1, rect_right + 1):
                            if rect_top <= r <= rect_bottom and rect_left <= c <= rect_right:
                                output_grid[r, c] = color
                                
            elif corner_r == rect_bottom and corner_c == rect_left:
                # Bottom-left corner
                width_marker = None
                height_marker = None
                for r, c in external_markers:
                    if r == rect_bottom + 1:
                        width_marker = (r, c)
                    elif c == rect_left - 1:
                        height_marker = (r, c)
                
                if width_marker and height_marker:
                    width = width_marker[1] - rect_left + 1
                    height = rect_bottom - height_marker[0] + 1
                    # Fill sub-rectangle
                    for r in range(rect_bottom - height + 1, rect_bottom + 1):
                        for c in range(rect_left, rect_left + width):
                            if rect_top <= r <= rect_bottom and rect_left <= c <= rect_right:
                                output_grid[r, c] = color
                                
            else:  # bottom-right corner
                width_marker = None
                height_marker = None
                for r, c in external_markers:
                    if r == rect_bottom + 1:
                        width_marker = (r, c)
                    elif c == rect_right + 1:
                        height_marker = (r, c)
                
                if width_marker and height_marker:
                    width = rect_right - width_marker[1] + 1
                    height = rect_bottom - height_marker[0] + 1
                    # Fill sub-rectangle
                    for r in range(rect_bottom - height + 1, rect_bottom + 1):
                        for c in range(rect_right - width + 1, rect_right + 1):
                            if rect_top <= r <= rect_bottom and rect_left <= c <= rect_right:
                                output_grid[r, c] = color
        
        # Remove external marker cells (set them back to background)
        for r, c in external_markers_to_remove:
            output_grid[r, c] = 0
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'grid_size': random.randint(17, 30),
            'object_color': random.randint(1, 9)
        }
        
        # Generate diverse train and test examples
        train_examples = []
        test_examples = []
        
        num_train = random.randint(3, 5)
        
        for i in range(num_train + 1):  # +1 for test example
            # Generate sub-rectangles for this example
            num_sub_rects = random.randint(1, 4)
            available_colors = [c for c in range(1, 10) if c != taskvars['object_color']]
            random.shuffle(available_colors)
            
            corners = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
            random.shuffle(corners)
            
            sub_rectangles = []
            for j in range(num_sub_rects):
                # Ensure sub-rectangles don't get too large
                max_size = min(8, taskvars['grid_size'] // 4)
                width = random.randint(2, max_size)
                height = random.randint(2, max_size)
                
                sub_rectangles.append({
                    'corner': corners[j],
                    'width': width,
                    'height': height,
                    'color': available_colors[j]
                })
            
            gridvars = {'sub_rectangles': sub_rectangles}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            example = {'input': input_grid, 'output': output_grid}
            
            if i < num_train:
                train_examples.append(example)
            else:
                test_examples.append(example)
        
        return taskvars, {'train': train_examples, 'test': test_examples}


