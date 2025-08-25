from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, random_cell_coloring
from transformation_library import find_connected_objects
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task1a6449f1Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "Each input grid contains multiple randomly distributed multi-colored cells, with some standing alone and others being 4-way connected to neighboring colored cells. The remaining cells are empty (0).",
            "On top of these multi-colored and empty (0) cells, there are 2 or 3 one-cell-wide rectangular frames placed, enclosing some of the previously placed multi-colored and empty cells.",
            "All frames vary in size and color, with the smallest being at least 3x4, and the others being larger."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by identifying the largest one-cell-wide rectangular frame in the input grid that encloses several multi-colored and empty cells.",
            "Then, simply copy all the interior cells of this largest rectangle and paste them into the output grid.",
            "The size of the output must exactly match the size of the interior of the largest frame."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def _draw_frame(self, grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
        """Draw a rectangular frame on the grid."""
        # Top and bottom edges
        grid[top, left:left+width] = color
        grid[top+height-1, left:left+width] = color
        # Left and right edges
        grid[top:top+height, left] = color
        grid[top:top+height, left+width-1] = color
    
    def _get_frame_interior_coords(self, top: int, left: int, height: int, width: int) -> List[Tuple[int, int]]:
        """Get coordinates of interior cells of a frame."""
        interior_coords = []
        for r in range(top+1, top+height-1):
            for c in range(left+1, left+width-1):
                interior_coords.append((r, c))
        return interior_coords
    
    def _has_sufficient_content(self, grid: np.ndarray, interior_coords: List[Tuple[int, int]], min_colored: int = 3) -> bool:
        """Check if frame interior has sufficient colored content."""
        colored_count = sum(1 for r, c in interior_coords if grid[r, c] != 0)
        return colored_count >= min_colored
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)
        
        # First, place random multi-colored cells
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        random_cell_coloring(grid, available_colors, density=0.2)
        
        def generate_valid_frames():
            frame_data = []
            num_frames = random.choice([2, 3])
            
            # More conservative approach to frame generation
            # Start with smaller frames and ensure they fit
            max_frame_h = min(rows - 4, rows // 2)  # Leave room for placement
            max_frame_w = min(cols - 4, cols // 2)
            
            if max_frame_h < 5 or max_frame_w < 6:  # Minimum frame size check
                return None
                
            # Generate different sized frames
            frame_sizes = []
            
            # First frame (smallest) - ensure minimum interior of 3x4
            min_h, min_w = 5, 6
            h1 = random.randint(min_h, min(max_frame_h, 7))
            w1 = random.randint(min_w, min(max_frame_w, 8))
            frame_sizes.append((h1, w1))
            
            # Additional frames (larger)
            for i in range(1, num_frames):
                # Make each subsequent frame larger than previous
                prev_h, prev_w = frame_sizes[-1]
                min_h = max(prev_h + 1, 6)
                min_w = max(prev_w + 1, 7)
                
                # Ensure we have valid range
                if min_h > max_frame_h or min_w > max_frame_w:
                    # Try smaller increments
                    min_h = prev_h + random.randint(0, 1)  
                    min_w = prev_w + random.randint(0, 1)
                    
                if min_h > max_frame_h or min_w > max_frame_w:
                    return None
                
                h = random.randint(min_h, max_frame_h)
                w = random.randint(min_w, max_frame_w)
                frame_sizes.append((h, w))
            
            # Ensure exactly one largest frame by area
            areas = [h*w for h, w in frame_sizes]
            max_area = max(areas)
            largest_count = areas.count(max_area)
            if largest_count != 1:
                return None
            
            # Try to place frames without overlap
            used_colors = set()
            for i, (h, w) in enumerate(frame_sizes):
                max_attempts = 50
                placed = False
                for _ in range(max_attempts):
                    top = random.randint(1, rows - h - 1)
                    left = random.randint(1, cols - w - 1)
                    
                    # Check for overlap with existing frames
                    overlap = False
                    for prev_top, prev_left, prev_h, prev_w, _ in frame_data:
                        if not (top + h <= prev_top or top >= prev_top + prev_h or 
                               left + w <= prev_left or left >= prev_left + prev_w):
                            overlap = True
                            break
                    
                    if not overlap:
                        # Choose frame color
                        frame_color = random.choice([c for c in available_colors if c not in used_colors])
                        used_colors.add(frame_color)
                        
                        # Check interior content
                        interior_coords = self._get_frame_interior_coords(top, left, h, w)
                        if self._has_sufficient_content(grid, interior_coords):
                            frame_data.append((top, left, h, w, frame_color))
                            placed = True
                            break
                
                if not placed:
                    return None
            
            return frame_data
        
        # Generate valid frame configuration
        frame_data = retry(generate_valid_frames, lambda x: x is not None, max_attempts=20)
        
        # Draw all frames
        for top, left, h, w, color in frame_data:
            self._draw_frame(grid, top, left, h, w, color)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find all rectangular frames by detecting frame colors
        frame_candidates = []
        
        # Try different potential frame colors
        for color in range(1, 10):
            if color not in grid:
                continue
                
            # Find potential rectangular structures of this color
            color_positions = set(zip(*np.where(grid == color)))
            
            # Try to identify rectangular frames
            rows_with_color = sorted(set(r for r, c in color_positions))
            cols_with_color = sorted(set(c for r, c in color_positions))
            
            if len(rows_with_color) < 3 or len(cols_with_color) < 3:
                continue
            
            # Check if this forms a rectangular frame
            for top_row in rows_with_color:
                for bottom_row in rows_with_color:
                    if bottom_row <= top_row + 2:  # Need at least 1 interior row
                        continue
                        
                    for left_col in cols_with_color:
                        for right_col in cols_with_color:
                            if right_col <= left_col + 2:  # Need at least 1 interior col
                                continue
                            
                            # Check if this forms a valid frame
                            expected_frame_cells = set()
                            # Top and bottom edges
                            for c in range(left_col, right_col + 1):
                                expected_frame_cells.add((top_row, c))
                                expected_frame_cells.add((bottom_row, c))
                            # Left and right edges
                            for r in range(top_row, bottom_row + 1):
                                expected_frame_cells.add((r, left_col))
                                expected_frame_cells.add((r, right_col))
                            
                            # Check if all expected frame cells exist and have the right color
                            if expected_frame_cells.issubset(color_positions):
                                h = bottom_row - top_row + 1
                                w = right_col - left_col + 1
                                area = h * w
                                frame_candidates.append((area, top_row, left_col, h, w))
        
        # Find the largest frame
        if not frame_candidates:
            # Fallback: return a small portion of the grid
            return grid[:3, :3]
        
        largest_frame = max(frame_candidates, key=lambda x: x[0])
        _, top, left, h, w = largest_frame
        
        # Extract interior of the largest frame
        interior_top = top + 1
        interior_left = left + 1
        interior_h = h - 2
        interior_w = w - 2
        
        return grid[interior_top:interior_top+interior_h, interior_left:interior_left+interior_w].copy()
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'rows': random.randint(17, 30),
            'cols': random.randint(17, 30)
        }
        
        # Generate training and test examples
        num_train = random.randint(3, 5)
        return taskvars, self.create_grids_default(num_train, 1, taskvars)

