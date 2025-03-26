from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects

class Task56ff96f3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain one or more pairs of same-colored cells, with the remaining cells being empty (0).",
            "Each pair of same-colored cells is positioned so that it marks two opposite corners of a rectangle—either top-left and bottom-right or top-right and bottom-left.",
            "If there is more than one pair of same-colored cells in the grid, each pair must be of a different color and positioned such that the rectangles they define do not overlap with those defined by other pairs.",
            "The colors, sizes, and positions marked by the colored pairs vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are constructed by copying the input grids and identifying all same-colored pairs of cells.",
            "These pairs mark two opposite corners of a rectangle.",
            "To construct the output grids, fill in each rectangle marked by the pairs using the color of the corresponding pair."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Define task variables
        taskvars = {
            'rows': random.randint(5, 15),
            'cols': random.randint(5, 15)
        }
        
        # Create 3-4 training examples
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        # First training example - include only one pair
        input_grid = self.create_input(taskvars, {'num_pairs': 1})
        output_grid = self.transform_input(input_grid, taskvars)
        train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create additional training examples with varying number of pairs
        for i in range(1, num_train_examples):
            max_pairs = min(5, (taskvars['rows'] * taskvars['cols']) // 30)
            num_pairs = random.randint(1, max_pairs)
            input_grid = self.create_input(taskvars, {'num_pairs': num_pairs})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        max_pairs = min(9, (taskvars['rows'] * taskvars['cols']) // 30) 
        num_pairs = random.randint(1, max_pairs)
        input_grid = self.create_input(taskvars, {'num_pairs': num_pairs})
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples = [{'input': input_grid, 'output': output_grid}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        num_pairs = gridvars.get('num_pairs', random.randint(1, min(5, (rows * cols) // 30)))
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Track rectangle areas to ensure separation
        rectangle_areas = []
        
        # Generate each pair with a different color
        colors_used = []
        for pair_idx in range(num_pairs):
            # Choose a color not yet used (colors 1-9)
            available_colors = [i for i in range(1, 10) if i not in colors_used]
            if not available_colors:
                break  # Can't add more pairs if we're out of colors
            
            color = random.choice(available_colors)
            colors_used.append(color)
            
            # Function to create a valid pair of points that form opposite corners
            def generate_valid_corners():
                # Random dimensions for rectangle
                width = random.randint(2, cols - 1)
                height = random.randint(2, rows - 1)
                
                # Random starting point ensuring everything fits
                x1 = random.randint(0, cols - width)
                y1 = random.randint(0, rows - height)
                
                # Determine opposite corner
                x2 = x1 + width - 1
                y2 = y1 + height - 1
                
                # Decide corner configuration (diagonal)
                if random.choice([True, False]):
                    # Top-left and bottom-right
                    corners = [(y1, x1), (y2, x2)]
                else:
                    # Top-right and bottom-left
                    corners = [(y1, x2), (y2, x1)]
                
                # Define the rectangle area plus a 1-cell buffer for separation
                buffer_x1 = max(0, min(corners[0][1], corners[1][1]) - 1)
                buffer_y1 = max(0, min(corners[0][0], corners[1][0]) - 1)
                buffer_x2 = min(cols - 1, max(corners[0][1], corners[1][1]) + 1)
                buffer_y2 = min(rows - 1, max(corners[0][0], corners[1][0]) + 1)
                
                rect_area = (buffer_x1, buffer_y1, buffer_x2, buffer_y2)
                
                # Check if this rectangle overlaps with existing ones
                for existing_rect in rectangle_areas:
                    ex1, ey1, ex2, ey2 = existing_rect
                    # Check for overlap
                    if not (buffer_x2 < ex1 or buffer_x1 > ex2 or buffer_y2 < ey1 or buffer_y1 > ey2):
                        return None, None
                
                return corners, rect_area
            
            # Try to generate valid corners multiple times
            for attempt in range(100):
                corners, rect_area = generate_valid_corners()
                if corners is not None:
                    break
            
            # If we couldn't find a valid placement, skip this pair
            if corners is None:
                continue
                
            # Place the corners on the grid
            for y, x in corners:
                grid[y, x] = color
                
            # Add this rectangle to our tracked areas
            rectangle_areas.append(rect_area)
        
        return grid

    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        rows, cols = grid.shape
        
        # Find all objects (cells with same color)
        objects_by_color = {}
        for r in range(rows):
            for c in range(cols):
                color = grid[r, c]
                if color > 0:  # Not background
                    if color not in objects_by_color:
                        objects_by_color[color] = []
                    objects_by_color[color].append((r, c))
        
        # Process each color group
        for color, points in objects_by_color.items():
            if len(points) == 2:  # We need exactly 2 points to define a rectangle
                # Get the corner points
                (y1, x1), (y2, x2) = points
                
                # Determine the rectangle corners (min/max of coordinates)
                min_x, max_x = min(x1, x2), max(x1, x2)
                min_y, max_y = min(y1, y2), max(y1, y2)
                
                # Fill in the rectangle
                for r in range(min_y, max_y + 1):
                    for c in range(min_x, max_x + 1):
                        output_grid[r, c] = color
        
        return output_grid


