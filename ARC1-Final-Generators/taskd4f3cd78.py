from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject
from input_library import retry
import numpy as np
import random

class Taskd4f3cd78Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of squared shape and can have different sizes.",
            "Each grid contains exactly one colored rectangle shape, composed of colored cells of color {color('border_color')}.",
            "The colored rectangle shape has a border made of colored cells surrounding internal cells that are empty (0), with at least one such internal cell.",
            "Exactly one specific cell on the border of the colored rectangle is empty (0), creating a single, intentional gap in the otherwise continuous colored border.",
            "All the remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All cells of color {color('border_color')} are identified.",
            "All empty (0) internal cells enclosed by the colored border are filled with color {color('internal_color')}.",
            "The single empty (0) cell that creates a gap in the border is also filled with color {color('internal_color')}.",
            "Based on the location of the gap, all empty (0) cells aligned with it and extending outside the rectangle in the same direction are filled with {color('internal_color')}: if the gap is on the top side, fill upward in the same column; if on the bottom side, fill downward in the same column; if on the left side, fill leftward in the same row; and if on the right side, fill rightward in the same row."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        border_color = taskvars['border_color']
        size = random.randint(8, 30)  # Square grids between 8x8 and 30x30
        
        def generate_rectangle_with_gap():
            grid = np.zeros((size, size), dtype=int)
            
            # Create rectangle dimensions (at least 4x4 to have internal cells and gap flexibility)
            min_rect_size = 4
            # Leave enough margin for positioning and extension
            margin = 2
            max_rect_size = min(size - 2 * margin, size // 2 + 3)  # Allow larger rectangles for bigger grids
            
            # Ensure we have valid bounds
            if max_rect_size < min_rect_size:
                min_rect_size = 3
                margin = 1
                max_rect_size = max(min_rect_size, size - 2 * margin)
            
            rect_width = random.randint(min_rect_size, max_rect_size)
            rect_height = random.randint(min_rect_size, max_rect_size)
            
            # Position rectangle with margin
            max_start_row = size - rect_height - margin
            max_start_col = size - rect_width - margin
            
            # Ensure valid positioning bounds
            if max_start_row < margin:
                margin = max(0, max_start_row)
                max_start_row = size - rect_height - margin
            if max_start_col < margin:
                margin = max(0, max_start_col)
                max_start_col = size - rect_width - margin
            
            # Final safety check
            if max_start_row < margin or max_start_col < margin:
                margin = 0
                max_start_row = max(0, size - rect_height)
                max_start_col = max(0, size - rect_width)
            
            start_row = random.randint(margin, max_start_row) if max_start_row >= margin else margin
            start_col = random.randint(margin, max_start_col) if max_start_col >= margin else margin
            
            # Draw rectangle border
            # Top and bottom borders
            for c in range(start_col, start_col + rect_width):
                grid[start_row, c] = border_color  # Top
                grid[start_row + rect_height - 1, c] = border_color  # Bottom
            
            # Left and right borders
            for r in range(start_row, start_row + rect_height):
                grid[r, start_col] = border_color  # Left
                grid[r, start_col + rect_width - 1] = border_color  # Right
            
            # Create a gap in the border (not at corners)
            gap_side = random.choice(['top', 'bottom', 'left', 'right'])
            
            if gap_side == 'top' and rect_width > 2:
                gap_col = random.randint(start_col + 1, start_col + rect_width - 2)
                grid[start_row, gap_col] = 0
                gap_pos = (start_row, gap_col)
            elif gap_side == 'bottom' and rect_width > 2:
                gap_col = random.randint(start_col + 1, start_col + rect_width - 2)
                grid[start_row + rect_height - 1, gap_col] = 0
                gap_pos = (start_row + rect_height - 1, gap_col)
            elif gap_side == 'left' and rect_height > 2:
                gap_row = random.randint(start_row + 1, start_row + rect_height - 2)
                grid[gap_row, start_col] = 0
                gap_pos = (gap_row, start_col)
            elif gap_side == 'right' and rect_height > 2:
                gap_row = random.randint(start_row + 1, start_row + rect_height - 2)
                grid[gap_row, start_col + rect_width - 1] = 0
                gap_pos = (gap_row, start_col + rect_width - 1)
            else:
                # Fallback: create gap on top side
                if rect_width > 2:
                    gap_col = start_col + 1
                    grid[start_row, gap_col] = 0
                    gap_pos = (start_row, gap_col)
                else:
                    gap_pos = None
            
            return grid, gap_pos, gap_side, (start_row, start_col, rect_height, rect_width)
        
        # Use retry to ensure we get a valid rectangle
        def has_internal_cells():
            try:
                grid, gap_pos, gap_side, rect_info = generate_rectangle_with_gap()
                start_row, start_col, rect_height, rect_width = rect_info
                
                # Check if there are internal cells (at least one empty cell inside)
                internal_count = 0
                if rect_height > 2 and rect_width > 2:  # Must be big enough to have internals
                    for r in range(start_row + 1, start_row + rect_height - 1):
                        for c in range(start_col + 1, start_col + rect_width - 1):
                            if grid[r, c] == 0:
                                internal_count += 1
                
                return internal_count > 0 and gap_pos is not None
            except (ValueError, IndexError):
                return False
        
        result = retry(generate_rectangle_with_gap, lambda x: has_internal_cells())
        return result[0]  # Return just the grid
    
    def transform_input(self, grid, taskvars):
        border_color = taskvars['border_color']
        internal_color = taskvars['internal_color']
        output_grid = grid.copy()
        
        # Find the rectangle by locating border cells
        border_cells = set()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == border_color:
                    border_cells.add((r, c))
        
        if not border_cells:
            return output_grid
        
        # Find bounding box of rectangle
        rows = [r for r, c in border_cells]
        cols = [c for r, c in border_cells]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)
        
        # Find the gap by checking which border position is missing
        gap_pos = None
        gap_side = None
        
        # Check top border
        for c in range(min_col, max_col + 1):
            if (min_row, c) not in border_cells:
                gap_pos = (min_row, c)
                gap_side = 'top'
                break
        
        # Check bottom border
        if gap_pos is None:
            for c in range(min_col, max_col + 1):
                if (max_row, c) not in border_cells:
                    gap_pos = (max_row, c)
                    gap_side = 'bottom'
                    break
        
        # Check left border
        if gap_pos is None:
            for r in range(min_row, max_row + 1):
                if (r, min_col) not in border_cells:
                    gap_pos = (r, min_col)
                    gap_side = 'left'
                    break
        
        # Check right border
        if gap_pos is None:
            for r in range(min_row, max_row + 1):
                if (r, max_col) not in border_cells:
                    gap_pos = (r, max_col)
                    gap_side = 'right'
                    break
        
        # Fill internal cells
        for r in range(min_row + 1, max_row):
            for c in range(min_col + 1, max_col):
                if output_grid[r, c] == 0:
                    output_grid[r, c] = internal_color
        
        # Fill the gap
        if gap_pos:
            output_grid[gap_pos[0], gap_pos[1]] = internal_color
            
            # Fill extension based on gap side
            gap_row, gap_col = gap_pos
            
            if gap_side == 'top':
                # Fill upward in the same column
                for r in range(gap_row - 1, -1, -1):
                    if output_grid[r, gap_col] == 0:
                        output_grid[r, gap_col] = internal_color
                    else:
                        break
                        
            elif gap_side == 'bottom':
                # Fill downward in the same column
                for r in range(gap_row + 1, output_grid.shape[0]):
                    if output_grid[r, gap_col] == 0:
                        output_grid[r, gap_col] = internal_color
                    else:
                        break
                        
            elif gap_side == 'left':
                # Fill leftward in the same row
                for c in range(gap_col - 1, -1, -1):
                    if output_grid[gap_row, c] == 0:
                        output_grid[gap_row, c] = internal_color
                    else:
                        break
                        
            elif gap_side == 'right':
                # Fill rightward in the same row
                for c in range(gap_col + 1, output_grid.shape[1]):
                    if output_grid[gap_row, c] == 0:
                        output_grid[gap_row, c] = internal_color
                    else:
                        break
        
        return output_grid
    
    def create_grids(self):
        # Choose colors that are different
        all_colors = list(range(1, 10))  # Exclude 0 (background)
        border_color = random.choice(all_colors)
        remaining_colors = [c for c in all_colors if c != border_color]
        internal_color = random.choice(remaining_colors)
        
        taskvars = {
            'border_color': border_color,
            'internal_color': internal_color
        }
        
        # Generate 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        train_test_data = self.create_grids_default(num_train, 1, taskvars)
        
        return taskvars, train_test_data
