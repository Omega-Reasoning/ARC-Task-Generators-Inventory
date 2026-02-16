from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry
import numpy as np
import random

class Task11e1fe23Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains exactly three differently colored cells, with the remaining cells being empty (0).",
            "The positions of the three differently colored cells are determined by first sketching a rough square and placing the three colored cells on any three of its sides.",
            "The square must have odd dimensions, starting from 5, and must be smaller than min of rows and columns.",
            "The colors and positions of the cells must vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the three colored cells, which form three corners of a square.",
            "First, the center of the square is determined based on the positions of the three colored cells.",
            "This center cell is filled with {color('center_color')}.",
            "Then, three more cells are added diagonally adjacent to the {color('center_color')} cell, based on the positions of the original three colored cells.",
            "Each diagonal cell is filled with the same color as the corresponding original corner cell — for example, if a colored cell is placed at the top-left corner of the square, then a same-colored cell is added to the top-left diagonal of the {color('center_color')} cell."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        height = gridvars['height']
        width = gridvars['width']
        square_size = gridvars['square_size']
        colors = gridvars['colors']
        
        def generate_valid_grid():
            grid = np.zeros((height, width), dtype=int)
            
            # Calculate possible positions for the square within the grid
            max_top = height - square_size
            max_left = width - square_size
            
            # Random position for the square
            square_top = random.randint(0, max_top)
            square_left = random.randint(0, max_left)
            
            # Define the four corners of the square
            corners = [
                (square_top, square_left),  # top-left
                (square_top, square_left + square_size - 1),  # top-right
                (square_top + square_size - 1, square_left),  # bottom-left
                (square_top + square_size - 1, square_left + square_size - 1)  # bottom-right
            ]
            
            # Randomly choose 3 out of 4 corners
            chosen_corners = random.sample(corners, 3)
            
            # Place the three colored cells
            for i, (r, c) in enumerate(chosen_corners):
                grid[r, c] = colors[i]
            
            return grid, square_top + square_size // 2, square_left + square_size // 2, chosen_corners
        
        grid, center_r, center_c, chosen_corners = generate_valid_grid()
        
        # Store additional info for transformation
        gridvars['center_r'] = center_r
        gridvars['center_c'] = center_c
        gridvars['chosen_corners'] = chosen_corners
        
        return grid

    def transform_input(self, grid, taskvars):
        # Find the three colored cells
        objects = find_connected_objects(grid, background=0)
        
        if len(objects) != 3:
            raise ValueError(f"Expected 3 objects, found {len(objects)}")
        
        # Get the positions and colors of the three cells
        colored_cells = []
        for obj in objects:
            if len(obj) == 1:  # Should be single cells
                r, c, color = list(obj.cells)[0]
                colored_cells.append((r, c, color))
        
        if len(colored_cells) != 3:
            raise ValueError("Expected exactly 3 single colored cells")
        
        # Sort by position to ensure consistent ordering
        colored_cells.sort()
        
        # Calculate the center of the implied square
        # Find the bounding box of the three points
        rows = [r for r, c, color in colored_cells]
        cols = [c for r, c, color in colored_cells]
        
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)
        
        # The center should be equidistant from the corners
        center_r = (min_r + max_r) // 2
        center_c = (min_c + max_c) // 2
        
        # Create output grid
        output_grid = grid.copy()
        
        # Fill center with center_color
        output_grid[center_r, center_c] = taskvars['center_color']
        
        # Add diagonal cells around the center
        for r, c, color in colored_cells:
            # Determine which corner this represents relative to center
            dr = 1 if r > center_r else -1
            dc = 1 if c > center_c else -1
            
            # Place diagonal cell
            diag_r = center_r + dr
            diag_c = center_c + dc
            
            if 0 <= diag_r < output_grid.shape[0] and 0 <= diag_c < output_grid.shape[1]:
                output_grid[diag_r, diag_c] = color
        
        return output_grid

    def create_grids(self):
        # Generate task variables
        center_color = random.randint(1, 9)
        
        # Generate training and test examples
        num_train = random.randint(3, 4)
        
        train_examples = []
        test_examples = []
        
        def generate_example():
            # Grid dimensions
            height = random.randint(8, 30)
            width = random.randint(8, 30)
            
            # Square size (odd, starting from 5)
            max_square_size = min(height, width)
            possible_sizes = [s for s in range(5, max_square_size, 2)]
            if not possible_sizes:
                possible_sizes = [5]
            square_size = random.choice(possible_sizes)
            
            # Generate 3 different colors (different from center_color)
            available_colors = [c for c in range(1, 10) if c != center_color]
            colors = random.sample(available_colors, 3)
            
            gridvars = {
                'height': height,
                'width': width,
                'square_size': square_size,
                'colors': colors
            }
            
            input_grid = self.create_input({}, gridvars)
            output_grid = self.transform_input(input_grid, {'center_color': center_color})
            
            return {'input': input_grid, 'output': output_grid}
        
        # Generate training examples
        for _ in range(num_train):
            example = retry(generate_example, lambda x: not np.array_equal(x['input'], x['output']))
            train_examples.append(example)
        
        # Generate test example
        test_example = retry(generate_example, lambda x: not np.array_equal(x['input'], x['output']))
        test_examples.append(test_example)
        
        taskvars = {'center_color': center_color}
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

