from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, retry
import numpy as np
import random

class Taskff28f65a(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square-shaped and come in different sizes.",
            "Each input grid contains a random number of square shapes, ranging from 1 to 5.",
            "Each square shape has fixed dimensions of {vars['m']} × {vars['m']} and is uniformly colored {color('square_color')}.",
            "The square shapes are not edge-connected.",
            "The number of square shapes varies across input grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is initialized as an empty 3×3 grid.",
            "The number of square-shaped objects in the input grid is counted and stored as n.",
            "Based on the value of n, the following cells are filled with the color {color('fill_color')}: If n = 1, cell (0, 0) is filled. If n = 2, cells (0, 0) and (0, 2) are filled. If n = 3, cells (0, 0), (0, 2), and (1, 1) are filled. If n = 4, cells (0, 0), (0, 2), (1, 1), and (2, 0) are filled. If n = 5, cells (0, 0), (0, 2), (1, 1), (2, 0), and (2, 2) are filled."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        grid_size = gridvars['grid_size']
        num_squares = gridvars['num_squares']
        m = taskvars['m']
        square_color = taskvars['square_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Place squares ensuring they don't overlap or connect
        squares_placed = 0
        max_attempts = 100
        
        for attempt in range(max_attempts):
            if squares_placed >= num_squares:
                break
                
            # Random position for top-left corner of square
            max_row = grid_size - m
            max_col = grid_size - m
            
            if max_row < 0 or max_col < 0:
                break  # Grid too small for this square size
                
            row = random.randint(0, max_row)
            col = random.randint(0, max_col)
            
            # Check if this position would overlap with existing squares
            # or create edge-connected shapes
            valid_position = True
            
            # Check the square area plus 1-cell border for conflicts
            check_row_start = max(0, row - 1)
            check_row_end = min(grid_size, row + m + 1)
            check_col_start = max(0, col - 1)
            check_col_end = min(grid_size, col + m + 1)
            
            if np.any(grid[check_row_start:check_row_end, check_col_start:check_col_end] != 0):
                valid_position = False
            
            if valid_position:
                # Place the square
                grid[row:row+m, col:col+m] = square_color
                squares_placed += 1
        
        return grid

    def transform_input(self, grid, taskvars):
        fill_color = taskvars['fill_color']
        
        # Find connected objects (squares)
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # Count the number of squares
        n = len(objects)
        
        # Initialize 3x3 output grid
        output = np.zeros((3, 3), dtype=int)
        
        # Fill cells based on count
        fill_positions = {
            1: [(0, 0)],
            2: [(0, 0), (0, 2)],
            3: [(0, 0), (0, 2), (1, 1)],
            4: [(0, 0), (0, 2), (1, 1), (2, 0)],
            5: [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
        }
        
        if n in fill_positions:
            for row, col in fill_positions[n]:
                output[row, col] = fill_color
        
        return output

    def create_grids(self):
        # Initialize task variables
        taskvars = {
            'm': random.randint(2, 4),  # Square size
            'square_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'fill_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        # Ensure different colors
        while taskvars['fill_color'] == taskvars['square_color']:
            taskvars['fill_color'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # Create training examples ensuring we cover all square counts 1-5
        # Generate 5-6 training examples with at least one of each count
        num_train = random.randint(5, 6)
        train_examples = []
        
        # First, ensure we have one example of each count (1-5)
        required_counts = [1, 2, 3, 4, 5]
        random.shuffle(required_counts)
        
        # If we're generating 6 examples, add one extra random count
        if num_train == 6:
            required_counts.append(random.choice([1, 2, 3, 4, 5]))
        
        for i in range(num_train):
            num_squares = required_counts[i]
            
            # Generate a valid grid
            def generate_valid_grid():
                # Calculate minimum grid size needed
                min_size_for_squares = max(5, int(np.ceil(np.sqrt(num_squares)) * (taskvars['m'] + 1) + taskvars['m']))
                grid_size = random.randint(
                    min_size_for_squares,
                    min(30, min_size_for_squares + 10)  # Don't make it too large
                )
                gridvars = {
                    'grid_size': grid_size,
                    'num_squares': num_squares
                }
                return self.create_input(taskvars, gridvars)
            
            # Retry until we get a valid grid with the exact number of squares
            input_grid = retry(
                generate_valid_grid,
                lambda g: len(find_connected_objects(g, diagonal_connectivity=False, background=0)) == num_squares,
                max_attempts=100
            )
            
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_num_squares = random.choice([1, 2, 3, 4, 5])
        
        def generate_test_grid():
            min_size_for_squares = max(5, int(np.ceil(np.sqrt(test_num_squares)) * (taskvars['m'] + 1) + taskvars['m']))
            grid_size = random.randint(
                min_size_for_squares,
                min(30, min_size_for_squares + 10)
            )
            gridvars = {
                'grid_size': grid_size,
                'num_squares': test_num_squares
            }
            return self.create_input(taskvars, gridvars)
        
        test_input = retry(
            generate_test_grid,
            lambda g: len(find_connected_objects(g, diagonal_connectivity=False, background=0)) == test_num_squares,
            max_attempts=100
        )
        
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

