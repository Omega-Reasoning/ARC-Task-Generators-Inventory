from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random

class Taska8d7556cGenerator(ARCTaskGenerator):
    
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "The input grid is are square grid of the same size.",
            "The grid contains randomly filled cells of {color('cell_color')} color.",
            "These cells spread across the grids and are closely spaced.",
            "While these cells are filling the grid, they leave a couple of squares and rectangles empty(0) cells in the grid."
        ]
        
        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is a copy of the input grid.",
            "The filled cells are preserved.",
            "The squares and rectangles are spotted correctly and are filled with a {color('fill_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # Generate random colors ensuring they're different
        cell_color = random.randint(1, 9)
        fill_color = random.choice([c for c in range(1, 10) if c != cell_color])
        
        # Store task variables
        taskvars = {
            'cell_color': cell_color,
            'fill_color': fill_color,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate grid sizes - need to be large enough to have rectangles
        min_size = 6
        max_size = 10
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': all_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': all_sizes[-1]}
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
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        """Create a completely random grid and check if it has non-overlapping rectangles/squares."""
        cell_color = taskvars['cell_color']
        grid_size = gridvars['grid_size']
        
        def rectangles_overlap(rect1, rect2):
            """Check if two rectangles overlap."""
            r1_top, r1_left = rect1['start_r'], rect1['start_c']
            r1_bottom = r1_top + rect1['height'] - 1
            r1_right = r1_left + rect1['width'] - 1
            
            r2_top, r2_left = rect2['start_r'], rect2['start_c']
            r2_bottom = r2_top + rect2['height'] - 1
            r2_right = r2_left + rect2['width'] - 1
            
            if (r1_right < r2_left or r2_right < r1_left or 
                r1_bottom < r2_top or r2_bottom < r1_top):
                return False
            return True
        
        def select_non_overlapping_rectangles(rectangles):
            """Select the maximum set of non-overlapping rectangles."""
            if not rectangles:
                return []
            
            sorted_rects = sorted(rectangles, 
                                key=lambda r: r['height'] * r['width'], 
                                reverse=True)
            selected = []
            
            for rect in sorted_rects:
                overlaps_with_selected = False
                for selected_rect in selected:
                    if rectangles_overlap(rect, selected_rect):
                        overlaps_with_selected = True
                        break
                
                if not overlaps_with_selected:
                    selected.append(rect)
            
            return selected
        
        def find_all_rectangles_and_squares(grid):
            """Find ALL possible rectangles and squares in the grid."""
            rows, cols = grid.shape
            rectangles = []
            
            for start_r in range(rows):
                for start_c in range(cols):
                    max_height = rows - start_r
                    max_width = cols - start_c
                    
                    for height in range(2, max_height + 1):
                        for width in range(2, max_width + 1):
                            is_valid_rectangle = True
                            
                            for r in range(start_r, start_r + height):
                                for c in range(start_c, start_c + width):
                                    if grid[r, c] != 0:
                                        is_valid_rectangle = False
                                        break
                                if not is_valid_rectangle:
                                    break
                            
                            if is_valid_rectangle:
                                rectangle = {
                                    'start_r': start_r,
                                    'start_c': start_c,
                                    'height': height,
                                    'width': width,
                                    'cells': [(r, c) for r in range(start_r, start_r + height) 
                                             for c in range(start_c, start_c + width)]
                                }
                                rectangles.append(rectangle)
            
            return rectangles
        
        def generate_random_grid():
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Completely random filling
            density = random.uniform(0.4, 0.75)
            random_cell_coloring(grid, cell_color, density=density, background=0, overwrite=False)
            
            # Find ALL possible rectangles and squares
            all_rectangles = find_all_rectangles_and_squares(grid)
            
            # Filter out rectangles that are too small or too large
            good_rectangles = []
            for rect in all_rectangles:
                area = rect['height'] * rect['width']
                if (4 <= area <= 16 and 
                    rect['height'] >= 2 and rect['width'] >= 2):
                    good_rectangles.append(rect)
            
            # Select non-overlapping rectangles
            selected_rectangles = select_non_overlapping_rectangles(good_rectangles)
            
            return grid, len(selected_rectangles) >= 1
        
        # Generate random grids until we find one with good non-overlapping rectangles/squares
        grid, has_good_shapes = retry(
            generate_random_grid,
            lambda x: x[1],
            max_attempts=300
        )
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        """Transform by filling non-overlapping rectangles and squares found in the grid."""
        output_grid = grid.copy()
        fill_color = taskvars['fill_color']
        
        # Find all rectangles and squares directly - NO nested functions with parameters
        rows, cols = grid.shape
        all_rectangles = []
        
        for start_r in range(rows):
            for start_c in range(cols):
                max_height = rows - start_r
                max_width = cols - start_c
                
                for height in range(2, max_height + 1):
                    for width in range(2, max_width + 1):
                        is_valid_rectangle = True
                        
                        for r in range(start_r, start_r + height):
                            for c in range(start_c, start_c + width):
                                if grid[r, c] != 0:
                                    is_valid_rectangle = False
                                    break
                            if not is_valid_rectangle:
                                break
                        
                        if is_valid_rectangle:
                            rectangle = {
                                'start_r': start_r,
                                'start_c': start_c,
                                'height': height,
                                'width': width,
                                'cells': [(r, c) for r in range(start_r, start_r + height) 
                                        for c in range(start_c, start_c + width)]
                            }
                            all_rectangles.append(rectangle)
        
        # Filter rectangles by size
        good_rectangles = []
        for rect in all_rectangles:
            area = rect['height'] * rect['width']
            if (4 <= area <= 16 and 
                rect['height'] >= 2 and rect['width'] >= 2):
                good_rectangles.append(rect)
        
        # Select non-overlapping rectangles inline
        if good_rectangles:
            sorted_rects = sorted(good_rectangles, 
                                key=lambda r: r['height'] * r['width'], 
                                reverse=True)
            selected_rectangles = []
            
            for rect in sorted_rects:
                overlaps_with_selected = False
                for selected_rect in selected_rectangles:
                    # Check overlap inline
                    r1_top, r1_left = rect['start_r'], rect['start_c']
                    r1_bottom = r1_top + rect['height'] - 1
                    r1_right = r1_left + rect['width'] - 1
                    
                    r2_top, r2_left = selected_rect['start_r'], selected_rect['start_c']
                    r2_bottom = r2_top + selected_rect['height'] - 1
                    r2_right = r2_left + selected_rect['width'] - 1
                    
                    if not (r1_right < r2_left or r2_right < r1_left or 
                        r1_bottom < r2_top or r2_bottom < r1_top):
                        overlaps_with_selected = True
                        break
                
                if not overlaps_with_selected:
                    selected_rectangles.append(rect)
        else:
            selected_rectangles = []
        
        # Fill each selected rectangle/square
        for rect in selected_rectangles:
            for r, c in rect['cells']:
                output_grid[r, c] = fill_color
        
        return output_grid