from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taske76a88a6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "In each input grid, there are multiple rectangles of equal size, all colored {color('background_color')}, except for one rectangle, which is divided into two randomly selected colors.",
            "In each input grid, the rectangles are positioned so that they do not touch each other.",
            "The two random colors vary across different input grids in order to ensure diversity."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all the rectangle shapes, including one rectangle that consists of two colors.",
            "The two-color pattern from that rectangle is then applied to the other rectangles that originally had only a single color"
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        background_color = taskvars['background_color']
        color1 = gridvars['color1']
        color2 = gridvars['color2']
        
        # Create empty grid
        grid = np.zeros((n, n), dtype=int)
        
        # Determine rectangle size and number of rectangles
        num_rectangles = random.randint(2, 4)  # 2-4 rectangles total
        
        # Calculate rectangle size based on grid size
        max_rect_size = max(2, min(6, n // 3))  # Reasonable rectangle size
        rect_height = random.randint(2, max_rect_size)
        rect_width = random.randint(2, max_rect_size)
        
        # Place rectangles with spacing
        placed_rectangles = []
        max_attempts = 100
        
        for i in range(num_rectangles):
            attempts = 0
            while attempts < max_attempts:
                # Try to place a rectangle
                start_row = random.randint(0, n - rect_height)
                start_col = random.randint(0, n - rect_width)
                
                # Check if this position conflicts with existing rectangles
                new_rect = (start_row, start_col, start_row + rect_height, start_col + rect_width)
                
                # Check for overlap or touching
                valid = True
                for existing in placed_rectangles:
                    if self._rectangles_overlap_or_touch(new_rect, existing):
                        valid = False
                        break
                
                if valid:
                    placed_rectangles.append(new_rect)
                    break
                    
                attempts += 1
            
            if attempts >= max_attempts:
                # If we can't place more rectangles, break
                break
        
        # Fill rectangles - one with two colors, others with background color
        if len(placed_rectangles) > 0:
            # Choose which rectangle gets the two-color pattern
            special_rect_idx = random.randint(0, len(placed_rectangles) - 1)
            
            for i, (r1, c1, r2, c2) in enumerate(placed_rectangles):
                if i == special_rect_idx:
                    # Create two-color pattern
                    self._fill_rectangle_with_two_colors(grid, r1, c1, r2, c2, color1, color2)
                else:
                    # Fill with background color
                    grid[r1:r2, c1:c2] = background_color
        
        return grid
    
    def _rectangles_overlap_or_touch(self, rect1: Tuple[int, int, int, int], 
                                    rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap or touch (including diagonally)"""
        r1_start, c1_start, r1_end, c1_end = rect1
        r2_start, c2_start, r2_end, c2_end = rect2
        
        # Check if they overlap or touch (no gap between them)
        return not (r1_end < r2_start or r2_end < r1_start or 
                   c1_end < c2_start or c2_end < c1_start)
    
    def _fill_rectangle_with_two_colors(self, grid: np.ndarray, r1: int, c1: int, 
                                       r2: int, c2: int, color1: int, color2: int):
        """Fill a rectangle region with a two-color pattern"""
        # Create a random pattern with two colors
        for r in range(r1, r2):
            for c in range(c1, c2):
                grid[r, c] = random.choice([color1, color2])
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        background_color = taskvars['background_color']
        
        # Find all rectangles by detecting connected components
        rectangles = self._find_rectangles(grid)
        
        if len(rectangles) < 2:
            return output_grid
        
        # Find the rectangle with two colors (pattern rectangle)
        pattern_rectangle = None
        pattern_data = None
        
        for rect_info in rectangles:
            r1, c1, r2, c2 = rect_info
            rect_region = grid[r1:r2, c1:c2]
            unique_colors = np.unique(rect_region)
            
            # Check if this rectangle has exactly 2 colors and none is background (0)
            if len(unique_colors) == 2 and 0 not in unique_colors:
                pattern_rectangle = rect_info
                pattern_data = rect_region.copy()
                break
        
        if pattern_rectangle is None or pattern_data is None:
            return output_grid
        
        # Apply the pattern to other rectangles
        for rect_info in rectangles:
            if rect_info != pattern_rectangle:
                r1, c1, r2, c2 = rect_info
                rect_region = grid[r1:r2, c1:c2]
                
                # Check if this is a single-color rectangle (background color)
                unique_colors = np.unique(rect_region)
                if len(unique_colors) == 1 and background_color in unique_colors:
                    # Apply the pattern
                    if pattern_data.shape == rect_region.shape:
                        output_grid[r1:r2, c1:c2] = pattern_data
                    else:
                        # If sizes don't match, scale the pattern
                        scaled_pattern = self._scale_pattern(pattern_data, rect_region.shape)
                        output_grid[r1:r2, c1:c2] = scaled_pattern
        
        return output_grid
    
    def _find_rectangles(self, grid: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find all rectangular regions in the grid using connected components"""
        rectangles = []
        
        # Find all non-zero connected components
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
        
        for obj in objects:
            # Get bounding box of the object
            bbox = obj.bounding_box
            r1, r2 = bbox[0].start, bbox[0].stop
            c1, c2 = bbox[1].start, bbox[1].stop
            
            # Check if the object completely fills its bounding box (i.e., it's rectangular)
            rect_region = grid[r1:r2, c1:c2]
            if np.all(rect_region != 0):  # All cells in bounding box are non-zero
                rectangles.append((r1, c1, r2, c2))
        
        return rectangles
    
    def _scale_pattern(self, pattern: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Scale a pattern to fit target shape using nearest neighbor"""
        target_h, target_w = target_shape
        pattern_h, pattern_w = pattern.shape
        
        # Simple nearest neighbor scaling
        scaled = np.zeros(target_shape, dtype=int)
        for r in range(target_h):
            for c in range(target_w):
                src_r = min((r * pattern_h) // target_h, pattern_h - 1)
                src_c = min((c * pattern_w) // target_w, pattern_w - 1)
                scaled[r, c] = pattern[src_r, src_c]
        
        return scaled
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Create task variables
        taskvars = {
            'n': random.randint(10, 20),  # Grid size
            'background_color': random.randint(1, 9)  # Background color for single-color rectangles
        }
        
        # Create training examples
        num_train = random.randint(3, 6)
        train_examples = []
        test_examples = []
        
        # Generate examples with different color pairs
        used_color_pairs = set()
        
        for i in range(num_train + 1):  # +1 for test example
            # Generate unique color pair
            attempts = 0
            while attempts < 50:
                color1 = random.randint(1, 9)
                color2 = random.randint(1, 9)
                
                # Make sure colors are different and not background
                if (color1 != color2 and 
                    color1 != taskvars['background_color'] and 
                    color2 != taskvars['background_color'] and
                    (color1, color2) not in used_color_pairs and
                    (color2, color1) not in used_color_pairs):
                    used_color_pairs.add((color1, color2))
                    break
                attempts += 1
            
            # If we couldn't find unique colors, just use any valid pair
            if attempts >= 50:
                color1 = random.randint(1, 9)
                color2 = random.randint(1, 9)
                while color1 == color2 or color1 == taskvars['background_color'] or color2 == taskvars['background_color']:
                    color1 = random.randint(1, 9)
                    color2 = random.randint(1, 9)
            
            gridvars = {'color1': color1, 'color2': color2}
            
            # Retry input generation if it fails
            try:
                input_grid = self.create_input(taskvars, gridvars)
                output_grid = self.transform_input(input_grid, taskvars)
                
                # Only add if transformation actually changed something
                if not np.array_equal(input_grid, output_grid):
                    example = {
                        'input': input_grid,
                        'output': output_grid
                    }
                    
                    if i < num_train:
                        train_examples.append(example)
                    else:
                        test_examples.append(example)
                else:
                    # If no change, retry this example
                    i -= 1
                    continue
                    
            except Exception as e:
                # If generation fails, retry this example
                i -= 1
                continue
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

