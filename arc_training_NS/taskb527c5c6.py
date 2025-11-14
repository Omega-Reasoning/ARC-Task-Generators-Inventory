from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class taskb527c5c6(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squared and have different sizes.",
            "Each input grid contains exactly two rectangles: one vertical and one horizontal.",
            "The rectangles do not touch one another.",
            "The length of each rectangle is at least one cell greater than twice its width (length ≥ 2 × width + 1).",
            "All cells of a rectangle are filled with {color('rectangle_color')}, except for one distinct single_cell at position (i, j), which is colored {color('single_color')}.",
            " For a vertical rectangle, the single_cell is located on either the leftmost or the rightmost column and the row index (j) of the single_cell lies within the length – 2(width – 1) middle rows.",
            "For a horizontal rectangle, the single_cell is located on either the topmost or the bottommost row and the column index (i) of the single_cell lies within the length – 2(width – 1) middle columns.",
            "If the single_cell of a vertical rectangle lies in the leftmost column, then for all rows in the range [j – (width – 1), j + (width – 1)], the cells in the columns to the left of the rectangle (up to the left boundary of the grid) must be empty.",
            "If the single_cell of a vertical rectangle lies in the rightmost column, then for all rows in the range [j – (width – 1), j + (width – 1)], the cells in the columns to the right of the rectangle (up to the right boundary of the grid) must be empty.",
            "If the single_cell of a horizontal rectangle lies in the bottommost row, then for all columns in the range [i – (width – 1), i + (width – 1)], the cells in the rows below the rectangle (up to the bottom boundary of the grid) must be empty.",
            "If the single_cell of a horizontal rectangle lies in the topmost row, then for all columns in the range [i – (width – 1), i + (width – 1)], the cells in the rows above the rectangle (up to the top boundary of the grid) must be empty.",
            "For any two rectangles in the grid, their respective constraint regions must be disjoint. The constraint region for a rectangle is defined as the area that must remain empty according to the single_cell position and rectangle orientation rules."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the two rectangles and their differently colored cells, hereafter referred to as single_cell.",
            "If the single_cell of a vertical rectangle lies in the leftmost column, then for all rows in the range [j – (width – 1), j + (width – 1)], the cells in the columns to the left of the rectangle (up to the left boundary of the grid) must be filled as follows: the cells in row j take the same color as the single_cell, while the cells in all other rows of this range take the same color as the overall rectangle.",
            "If the single_cell of a vertical rectangle lies in the rightmost column, then for all rows in the range [j – (width – 1), j + (width – 1)], the cells in the columns to the right of the rectangle (up to the right boundary of the grid) must be filled as follows: the cells in row j take the same color as the single_cell, while the cells in all other rows of this range take the same color as the overall rectangle.",
            "If the single_cell of a horizontal rectangle lies in the bottommost row, then for all columns in the range [i – (width – 1), i + (width – 1)], the cells in the rows below the rectangle (up to the bottom boundary of the grid) must be filled as follows: the cells in column i take the same color as the single_cell, while the cells in all other columns of this range take the same color as the overall rectangle.",
            "If the single_cell of a horizontal rectangle lies in the topmost row, then for all columns in the range [i – (width – 1), i + (width – 1)], the cells in the rows above the rectangle (up to the top boundary of the grid) must be filled as follows: the cells in column i take the same color as the single_cell, while the cells in all other columns of this range take the same color as the overall rectangle."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'rectangle_color': random.randint(1, 9),
            'single_color': random.randint(1, 9)
        }
        
        # Ensure colors are different
        while taskvars['single_color'] == taskvars['rectangle_color']:
            taskvars['single_color'] = random.randint(1, 9)
        
        # Generate train examples (3-6)
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

    def _get_rectangle_params(self, size: int) -> dict:
        """Get appropriate rectangle parameters based on grid size with maximum variation."""
        # Allow full range of widths from 2 to a reasonable maximum
        max_width = min(6, size // 4)  # Cap at 6 or quarter of grid size
        
        # Allow full range of lengths up to a significant portion of the grid
        max_length = size - 4  # Leave some margin for positioning
        
        return {
            'width_range': (2, max(2, max_width)),  # At least (2, 2)
            'max_length': max(5, max_length)  # At least 5
        }

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        def generate_valid_grid():
            # More balanced grid size distribution
            size_ranges = [
                (8, 12),   # Small grids
                (13, 18),  # Medium grids  
                (19, 25),  # Large grids
                (26, 30)   # Extra large grids
            ]
            
            # Pick a size range randomly (equal probability)
            min_size, max_size = random.choice(size_ranges)
            size = random.randint(min_size, max_size)
            
            grid = np.zeros((size, size), dtype=int)
            
            rectangle_color = taskvars['rectangle_color']
            single_color = taskvars['single_color']
            
            # Get appropriate rectangle parameters for this grid size
            rect_params = self._get_rectangle_params(size)
            
            # Generate vertical rectangle with full width variation
            v_width = random.randint(*rect_params['width_range'])
            min_v_length = 2 * v_width + 1  # Constraint: length ≥ 2 × width + 1
            max_v_length = min(rect_params['max_length'], size - 4)
            
            if min_v_length > max_v_length:
                return None
            
            # Allow full length variation
            v_length = random.randint(min_v_length, max_v_length)
            
            # Calculate minimum margins needed for constraints
            min_margin = max(v_width + 1, 2)
            
            # Position vertical rectangle with adequate margins
            min_row = min_margin
            max_row = size - v_length - min_margin
            min_col = min_margin
            max_col = size - v_width - min_margin
            
            if min_row > max_row or min_col > max_col:
                return None
                
            v_start_row = random.randint(min_row, max_row)
            v_start_col = random.randint(min_col, max_col)
            
            # Fill vertical rectangle
            for r in range(v_start_row, v_start_row + v_length):
                for c in range(v_start_col, v_start_col + v_width):
                    grid[r, c] = rectangle_color
            
            # Place single cell in vertical rectangle
            v_single_side = random.choice(['left', 'right'])
            
            # Calculate valid range for single cell position (middle rows)
            middle_start = v_width - 1
            middle_end = v_length - v_width
            
            if middle_start > middle_end:
                return None
                
            v_single_row_offset = random.randint(middle_start, middle_end)
            v_single_row = v_start_row + v_single_row_offset
            
            if v_single_side == 'left':
                v_single_col = v_start_col
            else:
                v_single_col = v_start_col + v_width - 1
            
            grid[v_single_row, v_single_col] = single_color
            
            # Calculate vertical rectangle's constraint region (transformation area)
            v_constraint_region = set()
            v_range_start = max(0, v_single_row - (v_width - 1))
            v_range_end = min(size - 1, v_single_row + (v_width - 1))
            
            if v_single_side == 'left':
                # Transformation area: columns to the left of vertical rectangle
                for r in range(v_range_start, v_range_end + 1):
                    for c in range(0, v_start_col):
                        v_constraint_region.add((r, c))
            else:  # right side
                # Transformation area: columns to the right of vertical rectangle
                for r in range(v_range_start, v_range_end + 1):
                    for c in range(v_start_col + v_width, size):
                        v_constraint_region.add((r, c))
            
            # Generate horizontal rectangle with independent width variation
            h_width = random.randint(*rect_params['width_range'])
            min_h_length = 2 * h_width + 1
            max_h_length = min(rect_params['max_length'], size - 4)
            
            if min_h_length > max_h_length:
                return None
            
            # Allow full length variation independently
            h_length = random.randint(min_h_length, max_h_length)
            
            # Try to place horizontal rectangle
            max_attempts = 300
            placed = False
            
            for _ in range(max_attempts):
                h_min_margin = max(h_width + 1, 2)
                h_min_row = h_min_margin
                h_max_row = size - h_width - h_min_margin
                h_min_col = h_min_margin
                h_max_col = size - h_length - h_min_margin
                
                if h_min_row > h_max_row or h_min_col > h_max_col:
                    continue
                    
                h_start_row = random.randint(h_min_row, h_max_row)
                h_start_col = random.randint(h_min_col, h_max_col)
                
                # Check that rectangles don't touch
                v_boundary = set()
                for r in range(v_start_row - 1, v_start_row + v_length + 1):
                    for c in range(v_start_col - 1, v_start_col + v_width + 1):
                        if 0 <= r < size and 0 <= c < size:
                            v_boundary.add((r, c))
                
                h_area = set()
                for r in range(h_start_row, h_start_row + h_width):
                    for c in range(h_start_col, h_start_col + h_length):
                        h_area.add((r, c))
                
                if h_area.intersection(v_boundary):
                    continue  # Rectangles would touch
                
                # Check that horizontal rectangle doesn't violate vertical rectangle's constraints
                if h_area.intersection(v_constraint_region):
                    continue
                
                # Calculate horizontal rectangle's constraint region
                h_single_side = random.choice(['top', 'bottom'])
                
                h_middle_start = h_width - 1
                h_middle_end = h_length - h_width
                
                if h_middle_start > h_middle_end:
                    continue
                    
                h_single_col_offset = random.randint(h_middle_start, h_middle_end)
                h_single_col = h_start_col + h_single_col_offset
                
                if h_single_side == 'top':
                    h_single_row = h_start_row
                else:
                    h_single_row = h_start_row + h_width - 1
                
                # Calculate horizontal rectangle's constraint region (transformation area)
                h_constraint_region = set()
                h_range_start = max(0, h_single_col - (h_width - 1))
                h_range_end = min(size - 1, h_single_col + (h_width - 1))
                
                if h_single_side == 'top':
                    # Transformation area: rows above horizontal rectangle
                    for c in range(h_range_start, h_range_end + 1):
                        for r in range(0, h_start_row):
                            h_constraint_region.add((r, c))
                else:  # bottom side
                    # Transformation area: rows below horizontal rectangle
                    for c in range(h_range_start, h_range_end + 1):
                        for r in range(h_start_row + h_width, size):
                            h_constraint_region.add((r, c))
                
                # Check that constraint regions are disjoint
                if v_constraint_region.intersection(h_constraint_region):
                    continue  # Constraint regions overlap, invalid configuration
                
                # Temporarily place horizontal rectangle to verify all constraints
                temp_grid = grid.copy()
                for r in range(h_start_row, h_start_row + h_width):
                    for c in range(h_start_col, h_start_col + h_length):
                        temp_grid[r, c] = rectangle_color
                
                temp_grid[h_single_row, h_single_col] = single_color
                
                # Verify that both constraint regions are empty in the temporary grid
                v_constraint_valid = all(temp_grid[r, c] == 0 for r, c in v_constraint_region)
                h_constraint_valid = all(temp_grid[r, c] == 0 for r, c in h_constraint_region)
                
                if v_constraint_valid and h_constraint_valid:
                    # Valid placement found
                    grid = temp_grid
                    placed = True
                    break
                    
            if not placed:
                return None
            
            return grid
        
        return retry(generate_valid_grid, lambda x: x is not None, max_attempts=800)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        rectangle_color = taskvars['rectangle_color']
        single_color = taskvars['single_color']
        
        # Analyze grid to find rectangles and their properties
        rectangles_info = self._analyze_rectangles(grid, rectangle_color, single_color)
        
        for rect_info in rectangles_info:
            single_row = rect_info['single_row']
            single_col = rect_info['single_col']
            single_side = rect_info['single_side']
            orientation = rect_info['orientation']
            width = rect_info['width']
            start_row = rect_info['start_row']
            start_col = rect_info['start_col']
            
            if orientation == 'vertical':
                # Calculate the range [j - (width-1), j + (width-1)]
                range_start = max(0, single_row - (width - 1))
                range_end = min(grid.shape[0] - 1, single_row + (width - 1))
                
                if single_side == 'left':
                    # Fill columns to the left of rectangle
                    for r in range(range_start, range_end + 1):
                        for c in range(0, start_col):
                            if r == single_row:
                                # Row j gets single_cell color
                                output_grid[r, c] = single_color
                            else:
                                # Other rows get rectangle color
                                output_grid[r, c] = rectangle_color
                                
                else:  # right side
                    # Fill columns to the right of rectangle
                    for r in range(range_start, range_end + 1):
                        for c in range(start_col + width, grid.shape[1]):
                            if r == single_row:
                                # Row j gets single_cell color
                                output_grid[r, c] = single_color
                            else:
                                # Other rows get rectangle color
                                output_grid[r, c] = rectangle_color
                                
            else:  # horizontal rectangle
                # Calculate the range [i - (width-1), i + (width-1)]
                range_start = max(0, single_col - (width - 1))
                range_end = min(grid.shape[1] - 1, single_col + (width - 1))
                
                if single_side == 'top':
                    # Fill rows above rectangle
                    for c in range(range_start, range_end + 1):
                        for r in range(0, start_row):
                            if c == single_col:
                                # Column i gets single_cell color
                                output_grid[r, c] = single_color
                            else:
                                # Other columns get rectangle color
                                output_grid[r, c] = rectangle_color
                                
                else:  # bottom side
                    # Fill rows below rectangle
                    for c in range(range_start, range_end + 1):
                        for r in range(start_row + width, grid.shape[0]):
                            if c == single_col:
                                # Column i gets single_cell color
                                output_grid[r, c] = single_color
                            else:
                                # Other columns get rectangle color
                                output_grid[r, c] = rectangle_color
        
        return output_grid

    def _analyze_rectangles(self, grid: np.ndarray, rectangle_color: int, single_color: int) -> list:
        """Analyze grid to extract rectangle information."""
        rectangles_info = []
        
        # Find all single-colored cells
        single_positions = np.where(grid == single_color)
        
        for i in range(len(single_positions[0])):
            single_r, single_c = single_positions[0][i], single_positions[1][i]
            
            # Find rectangle bounds by expanding from single cell
            min_r = max_r = single_r
            min_c = max_c = single_c
            
            # Expand to find all connected rectangle cells
            visited = set()
            stack = [(single_r, single_c)]
            
            while stack:
                cr, cc = stack.pop()
                if (cr, cc) in visited:
                    continue
                if (cr < 0 or cr >= grid.shape[0] or cc < 0 or cc >= grid.shape[1]):
                    continue
                if grid[cr, cc] != rectangle_color and grid[cr, cc] != single_color:
                    continue
                    
                visited.add((cr, cc))
                min_r = min(min_r, cr)
                max_r = max(max_r, cr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                
                # Add neighbors
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) not in visited:
                        stack.append((nr, nc))
            
            # Determine rectangle properties
            height = max_r - min_r + 1
            width = max_c - min_c + 1
            
            if height > width:  # Vertical rectangle
                side = 'left' if single_c == min_c else 'right'
                rectangles_info.append({
                    'start_row': min_r, 'start_col': min_c,
                    'width': width, 'length': height,
                    'single_row': single_r, 'single_col': single_c,
                    'single_side': side, 'orientation': 'vertical'
                })
            else:  # Horizontal rectangle
                side = 'top' if single_r == min_r else 'bottom'
                rectangles_info.append({
                    'start_row': min_r, 'start_col': min_c,
                    'width': height, 'length': width,
                    'single_row': single_r, 'single_col': single_c,
                    'single_side': side, 'orientation': 'horizontal'
                })
        
        return rectangles_info