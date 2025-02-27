from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry, create_object, Contiguity
from transformation_library import find_connected_objects, GridObject

class ARCTask868de0faGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid with dimension n.",
            "There are a few 4-way objects present in the input grid, each of these are a square whose perimeter cells are filled with color {color('per_color')} and the remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "Identify all the squares in the output grid, if the number of cells inside the square, i.e. not considering the perimeter but the cells inside the perimeter is even the color it {color('color_1')} else color {color('color_2')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        # Determine grid size (between 13 to 30)
        grid_size = random.randint(13, 30)
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Decide how many squares to generate (2-4 squares)
        num_squares = random.randint(2, 4)
        
        # Maximum square size is grid_size//2 - 1
        max_square_size = grid_size // 2 - 1
        
        # Generate squares of different sizes
        square_sizes = []
        
        # Ensure we have at least one even and one odd sized square
        even_size = random.randrange(4, max_square_size + 1, 2)  # Even size
        odd_size = random.randrange(3, max_square_size + 1, 2)   # Odd size
        
        square_sizes.append(even_size)
        square_sizes.append(odd_size)
        
        # Add more square sizes if needed
        while len(square_sizes) < num_squares:
            size = random.randint(4, max_square_size)
            if size not in square_sizes:
                square_sizes.append(size)
        
        # Shuffle the sizes
        random.shuffle(square_sizes)
        
        # Place squares on the grid
        for size in square_sizes:
            self._place_square_on_grid(grid, size, taskvars['per_color'])
        
        return grid
    
    def _place_square_on_grid(self, grid, size, color):
        """Place a square with perimeter of specified color on the grid."""
        grid_size = grid.shape[0]
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Random position for top-left corner
            top = random.randint(0, grid_size - size)
            left = random.randint(0, grid_size - size)
            
            # Check if this area is free
            if np.any(grid[top:top+size, left:left+size] != 0):
                continue  # Area not free, try again
            
            # Place the square perimeter
            # Top and bottom rows
            grid[top, left:left+size] = color
            grid[top+size-1, left:left+size] = color
            
            # Left and right columns (excluding corners which are already set)
            grid[top+1:top+size-1, left] = color
            grid[top+1:top+size-1, left+size-1] = color
            
            return True  # Successfully placed
        
        return False  # Could not place after max attempts
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Find all squares in the grid
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == taskvars['per_color']:
                    # Try to detect if this is part of a square perimeter
                    square_size = self._detect_square(grid, r, c, taskvars['per_color'])
                    if square_size > 0:
                        # Calculate number of cells inside the square
                        inside_cells = (square_size - 2) * (square_size - 2)
                        
                        # Fill inside based on whether number of inside cells is even or odd
                        fill_color = taskvars['color_1'] if inside_cells % 2 == 0 else taskvars['color_2']
                        
                        # Fill the inside of the square
                        output_grid[r+1:r+square_size-1, c+1:c+square_size-1] = fill_color
        
        return output_grid
    
    def _detect_square(self, grid, top_r, left_c, color):
        """
        Detect if there's a square starting at (top_r, left_c)
        Returns the size of the square if found, 0 otherwise
        """
        grid_size = grid.shape[0]
        
        # Look for horizontal line to the right
        size = 0
        for c in range(left_c, grid_size):
            if grid[top_r, c] == color:
                size += 1
            else:
                break
        
        if size < 2:  # Too small to be a square
            return 0
        
        # Check if there's a vertical line of the same length going down
        if top_r + size > grid_size:
            return 0  # Would go out of bounds
        
        # Check right vertical line
        for r in range(top_r, top_r + size):
            if grid[r, left_c + size - 1] != color:
                return 0  # Not a square
        
        # Check bottom horizontal line
        for c in range(left_c, left_c + size):
            if grid[top_r + size - 1, c] != color:
                return 0  # Not a square
        
        # Check left vertical line
        for r in range(top_r, top_r + size):
            if grid[r, left_c] != color:
                return 0  # Not a square
        
        # Check that inside is empty (not part of another square)
        if np.any(grid[top_r+1:top_r+size-1, left_c+1:left_c+size-1] != 0):
            return 0  # Inside is not empty
        
        return size
    
    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'per_color': random.randint(1, 9)
        }
        
        # Ensure color_1 and color_2 are different from per_color and each other
        colors = [i for i in range(1, 10) if i != taskvars['per_color']]
        taskvars['color_1'], taskvars['color_2'] = random.sample(colors, 2)
        
        # Generate 4-5 training examples and 1 test example
        num_train = random.randint(4, 5)
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples.append({'input': test_input, 'output': test_output})
        
        return taskvars, {'train': train_examples, 'test': test_examples}
