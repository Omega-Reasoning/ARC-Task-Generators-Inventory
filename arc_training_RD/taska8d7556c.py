from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random

class RectangleFillTaskGenerator(ARCTaskGenerator):
    
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
    
    def color_name(self, color: int) -> str:
        color_map = {
            0: "black",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "cyan",
            9: "brown"
        }
        return color_map.get(color, f"color_{color}")
    
    def rectangles_overlap(self, rect1, rect2):
        """Check if two rectangles overlap."""
        # Rectangle 1 boundaries
        r1_top, r1_left = rect1['start_r'], rect1['start_c']
        r1_bottom = r1_top + rect1['height'] - 1
        r1_right = r1_left + rect1['width'] - 1
        
        # Rectangle 2 boundaries  
        r2_top, r2_left = rect2['start_r'], rect2['start_c']
        r2_bottom = r2_top + rect2['height'] - 1
        r2_right = r2_left + rect2['width'] - 1
        
        # Check if they don't overlap (if any of these is true, they don't overlap)
        if (r1_right < r2_left or    # rect1 is completely to the left of rect2
            r2_right < r1_left or    # rect2 is completely to the left of rect1
            r1_bottom < r2_top or    # rect1 is completely above rect2
            r2_bottom < r1_top):     # rect2 is completely above rect1
            return False
        
        return True  # They overlap
    
    def select_non_overlapping_rectangles(self, rectangles):
        """
        Select the maximum set of non-overlapping rectangles.
        Uses a greedy approach: sort by area and pick largest non-overlapping ones.
        """
        if not rectangles:
            return []
        
        # Sort rectangles by area (largest first) for greedy selection
        sorted_rects = sorted(rectangles, 
                            key=lambda r: r['height'] * r['width'], 
                            reverse=True)
        
        selected = []
        
        for rect in sorted_rects:
            # Check if this rectangle overlaps with any already selected
            overlaps_with_selected = False
            for selected_rect in selected:
                if self.rectangles_overlap(rect, selected_rect):
                    overlaps_with_selected = True
                    break
            
            # If no overlap, add it to selected
            if not overlaps_with_selected:
                selected.append(rect)
        
        return selected
    
    def find_all_rectangles_and_squares(self, grid):
        """
        Find ALL possible rectangles and squares in the grid by checking every possible position and size.
        A rectangle/square is valid if ALL cells within it are empty (0).
        """
        rows, cols = grid.shape
        rectangles = []
        
        # Check every possible rectangle starting position
        for start_r in range(rows):
            for start_c in range(cols):
                # Check every possible rectangle size from this position
                max_height = rows - start_r
                max_width = cols - start_c
                
                # Try all possible heights (minimum 2 rows as requested)
                for height in range(2, max_height + 1):
                    # Try all possible widths (minimum 2 columns for rectangle)
                    for width in range(2, max_width + 1):
                        # Check if this rectangle contains only empty cells
                        is_valid_rectangle = True
                        
                        for r in range(start_r, start_r + height):
                            for c in range(start_c, start_c + width):
                                if grid[r, c] != 0:  # If any cell is not empty
                                    is_valid_rectangle = False
                                    break
                            if not is_valid_rectangle:
                                break
                        
                        if is_valid_rectangle:
                            # Create rectangle definition
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
    
    def create_input(self, taskvars, grid_size):
        """Create a completely random grid and check if it has non-overlapping rectangles/squares."""
        cell_color = taskvars['cell_color']
        
        def generate_random_grid():
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Completely random filling
            density = random.uniform(0.4, 0.75)
            random_cell_coloring(grid, cell_color, density=density, background=0, overwrite=False)
            
            # Find ALL possible rectangles and squares
            all_rectangles = self.find_all_rectangles_and_squares(grid)
            
            # Filter out rectangles that are too small or too large
            good_rectangles = []
            for rect in all_rectangles:
                area = rect['height'] * rect['width']
                # We want rectangles with area between 4 and 16, and at least 2 rows/cols
                if (4 <= area <= 16 and 
                    rect['height'] >= 2 and rect['width'] >= 2):
                    good_rectangles.append(rect)
            
            # Select non-overlapping rectangles
            selected_rectangles = self.select_non_overlapping_rectangles(good_rectangles)
            
            return grid, len(selected_rectangles) >= 1
        
        # Generate random grids until we find one with good non-overlapping rectangles/squares
        grid, has_good_shapes = retry(
            generate_random_grid,
            lambda x: x[1],
            max_attempts=300
        )
        
        return grid
    
    def transform_input(self, input_grid):
        """Transform by filling non-overlapping rectangles and squares found in the grid."""
        output_grid = input_grid.copy()
        fill_color = self.taskvars['fill_color']
        
        # Find all rectangles and squares
        all_rectangles = self.find_all_rectangles_and_squares(input_grid)
        
        # Filter rectangles by size (same criteria as in create_input)
        good_rectangles = []
        for rect in all_rectangles:
            area = rect['height'] * rect['width']
            if (4 <= area <= 16 and 
                rect['height'] >= 2 and rect['width'] >= 2):
                good_rectangles.append(rect)
        
        # Select non-overlapping rectangles
        selected_rectangles = self.select_non_overlapping_rectangles(good_rectangles)
        
        # Fill each selected rectangle/square
        for rect in selected_rectangles:
            for r, c in rect['cells']:
                output_grid[r, c] = fill_color
        
        return output_grid
    
    def create_grids(self):
        """Create train and test grids with consistent variables."""
        # Generate random colors ensuring they're different
        cell_color = random.randint(1, 9)
        fill_color = random.choice([c for c in range(1, 10) if c != cell_color])
        
        # Store task variables
        taskvars = {
            'cell_color': cell_color,
            'fill_color': fill_color,
        }
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Make taskvars available to transform_input
        self.taskvars = taskvars

        # Replace placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('cell_color')}", color_fmt('cell_color'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('fill_color')}", color_fmt('fill_color'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate 3-5 training pairs with different sizes
        num_train_pairs = random.randint(3, 5)
        
        # Generate grid sizes - need to be large enough to have rectangles
        min_size = 6
        max_size = 10
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_pairs + 1)]
        
        train_pairs = []
        
        for i in range(num_train_pairs):
            grid_size = all_sizes[i]
            input_grid = self.create_input(taskvars, grid_size)
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_grid_size = all_sizes[-1]
        test_input = self.create_input(taskvars, test_grid_size)
        test_output = self.transform_input(test_input)
        
        # Create the TrainTestData object
        data = TrainTestData(train=train_pairs, test=[GridPair(input=test_input, output=test_output)])
        
        return taskvars, data

