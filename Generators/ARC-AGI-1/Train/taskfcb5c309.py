from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Set

class Taskfcb5c309Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Each input grid is of size {vars['n']} × {vars['m']}.",
            "The input grid contains 2 or 3 randomly placed rectangular shapes, all with borders of the same randomly chosen color; the interiors of these rectangles are empty.",
            "The sizes and positions of the rectangles vary within each input grid.",
            "The rectangular shapes are separated from each other.",
            "A random number of single-colored cells in the input grid, excluding the rectangle borders, are colored with a different random color.",
            "At least one of these randomly single-colored cells appears inside one of the rectangular shapes in the input grid.",
            "In each input grid, the number of single-colored cells in each of the rectangular shapes are different."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by first identifying all rectangular shapes present in the input grid.",
            "For each rectangle, the number of single-colored cells (those colored in the secondary color) located within its interior is calculated.",
            "The rectangle containing the greatest number of such single-colored cells is then selected.",
            "The output grid consists exclusively of this selected rectangle and the single-colored cells it encloses.",
            "In the output, the border of the rectangle is recolored to match the color of the enclosed single-colored cells, while the single-colored cells themselves retain their original color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def _create_rectangle_border(self, grid: np.ndarray, top: int, left: int, height: int, width: int, color: int):
        """Create a rectangular border on the grid."""
        # Top and bottom borders
        grid[top, left:left+width] = color
        grid[top+height-1, left:left+width] = color
        # Left and right borders  
        grid[top:top+height, left] = color
        grid[top:top+height, left+width-1] = color
    
    def _get_rectangle_interior(self, top: int, left: int, height: int, width: int) -> Set[Tuple[int, int]]:
        """Get the interior coordinates of a rectangle."""
        interior = set()
        for r in range(top+1, top+height-1):
            for c in range(left+1, left+width-1):
                interior.add((r, c))
        return interior
    
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap or touch (with 1-cell buffer)."""
        top1, left1, height1, width1 = rect1
        top2, left2, height2, width2 = rect2
        
        # Add 1-cell buffer to ensure separation
        return not (top1 + height1 + 1 <= top2 or
                   top2 + height2 + 1 <= top1 or
                   left1 + width1 + 1 <= left2 or
                   left2 + width2 + 1 <= left1)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n, m = taskvars['n'], taskvars['m']
        
        def generate_valid_grid():
            # Choose random colors for this specific grid
            border_color = random.randint(1, 9)
            available_colors = [c for c in range(1, 10) if c != border_color]
            fill_color = random.choice(available_colors)
            
            grid = np.zeros((n, m), dtype=int)
            
            # Create 2 or 3 rectangles with larger minimum sizes
            num_rectangles = random.randint(2, 3)
            rectangles = []
            
            for _ in range(num_rectangles):
                attempts = 0
                while attempts < 100:
                    # Larger rectangle dimensions to ensure good interior space
                    # Minimum 5x5 to have 3x3 interior, up to about 1/3 of grid size
                    min_size = 5
                    max_height = min(n // 2 + 2, n - 2)
                    max_width = min(m // 2 + 2, m - 2)
                    
                    height = random.randint(min_size, max_height)
                    width = random.randint(min_size, max_width)
                    
                    # Position with enough space for borders
                    max_top = n - height
                    max_left = m - width
                    
                    if max_top <= 0 or max_left <= 0:
                        attempts += 1
                        continue
                    
                    top = random.randint(0, max_top)
                    left = random.randint(0, max_left)
                    
                    new_rect = (top, left, height, width)
                    
                    # Check if this rectangle overlaps with existing ones
                    if not any(self._rectangles_overlap(new_rect, existing) for existing in rectangles):
                        rectangles.append(new_rect)
                        self._create_rectangle_border(grid, top, left, height, width, border_color)
                        break
                    
                    attempts += 1
                
                if attempts >= 100:
                    return None  # Failed to place rectangle
            
            if len(rectangles) < 2:
                return None
            
            # Get all interior coordinates for each rectangle
            all_interiors = []
            for rect in rectangles:
                interior = self._get_rectangle_interior(*rect)
                all_interiors.append(interior)
            
            # Define target cell counts for each rectangle (must be different)
            # Create different counts ensuring at least one rectangle has the most
            if num_rectangles == 2:
                target_counts = [1, random.randint(2, 4)]  # e.g., [1, 3]
            else:  # 3 rectangles
                target_counts = [0, 1, random.randint(2, 5)]  # e.g., [0, 1, 3]
            
            # Shuffle to randomize which rectangle gets which count
            random.shuffle(target_counts)
            
            # Place cells according to target counts
            total_cells_needed = sum(target_counts)
            
            # Place the targeted cells in rectangles first
            for i, (interior, target_count) in enumerate(zip(all_interiors, target_counts)):
                if target_count > 0:
                    available_coords = [(r, c) for r, c in interior if grid[r, c] == 0]
                    if len(available_coords) < target_count:
                        return None  # Not enough space
                    
                    coords_to_fill = random.sample(available_coords, target_count)
                    for r, c in coords_to_fill:
                        grid[r, c] = fill_color
            
            # Add more random cells outside rectangles
            non_interior_coords = []
            for r in range(n):
                for c in range(m):
                    if grid[r, c] == 0:  # Empty cell
                        # Check if it's inside any rectangle interior
                        is_interior = False
                        for interior in all_interiors:
                            if (r, c) in interior:
                                is_interior = True
                                break
                        if not is_interior:
                            non_interior_coords.append((r, c))
            
            # Increased number of random cells outside rectangles
            if non_interior_coords:
                # Scale with grid size and total interior cells, but with more generous limits
                grid_size = n * m
                base_extra = max(3, total_cells_needed // 2)  # At least 3, or half of interior cells
                max_extra_cells = min(
                    len(non_interior_coords), 
                    max(8, grid_size // 25),  # At least 8, or ~4% of grid size
                    total_cells_needed + 5    # Or interior cells + 5
                )
                
                num_extra_cells = random.randint(base_extra, max_extra_cells)
                if num_extra_cells > 0:
                    extra_coords = random.sample(non_interior_coords, num_extra_cells)
                    for r, c in extra_coords:
                        grid[r, c] = fill_color
            
            # Verify that each rectangle has a different count
            actual_counts = []
            for interior in all_interiors:
                count = sum(1 for r, c in interior if grid[r, c] == fill_color)
                actual_counts.append(count)
            
            if len(set(actual_counts)) != len(actual_counts):  # Not all different
                return None
            
            if max(actual_counts) == 0:  # No rectangle has any cells
                return None
            
            return grid
        
        return retry(generate_valid_grid, lambda x: x is not None)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n, m = grid.shape

        unique_colors = list(set(grid.flatten()) - {0})
        if len(unique_colors) < 2:
            return np.zeros((5, 5), dtype=int)

        rectangles = []

        # ---- INLINE rectangle detection ----
        for color in unique_colors:
            for r in range(n - 4):
                for c in range(m - 4):
                    if grid[r, c] != color:
                        continue

                    for height in range(5, n - r + 1):
                        for width in range(5, m - c + 1):

                            if r + height > n or c + width > m:
                                continue

                            valid = True

                            # top and bottom
                            for cc in range(c, c + width):
                                if grid[r, cc] != color or grid[r + height - 1, cc] != color:
                                    valid = False
                                    break
                            if not valid:
                                continue

                            # left and right
                            for rr in range(r, r + height):
                                if grid[rr, c] != color or grid[rr, c + width - 1] != color:
                                    valid = False
                                    break

                            if valid:
                                rectangles.append((r, c, height, width, color))

        if not rectangles:
            return np.zeros((5, 5), dtype=int)

        # ---- Deduplicate rectangles ----
        unique_rects = []
        for rect in rectangles:
            if rect not in unique_rects:
                unique_rects.append(rect)

        # ---- Count interior fill cells ----
        best_rect = None
        max_fill = -1
        fill_color = None

        for top, left, height, width, border_color in unique_rects:
            interior_count = {}
            for rr in range(top + 1, top + height - 1):
                for cc in range(left + 1, left + width - 1):
                    val = grid[rr, cc]
                    if val != 0 and val != border_color:
                        interior_count[val] = interior_count.get(val, 0) + 1

            if interior_count:
                candidate_color = max(interior_count, key=interior_count.get)
                count = interior_count[candidate_color]

                if count > max_fill:
                    max_fill = count
                    best_rect = (top, left, height, width, border_color)
                    fill_color = candidate_color

        if best_rect is None:
            return np.zeros((5, 5), dtype=int)

        top, left, height, width, border_color = best_rect

        # ---- Build output ----
        output = np.zeros((height, width), dtype=int)

        for r in range(height):
            for c in range(width):
                val = grid[top + r, left + c]
                if val == border_color:
                    output[r, c] = fill_color
                elif val == fill_color:
                    output[r, c] = fill_color

        return output
    
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables - only grid size, no colors
        taskvars = {
            'n': random.randint(12, 25),
            'm': random.randint(12, 25),
        }
        
        # Generate examples
        num_train = random.randint(3, 6)
        num_test = 1
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        for _ in range(num_test):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_examples.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_examples, 'test': test_examples}

