from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry, create_object, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task3f7978a0Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids can have different sizes.",
            "They contain two vertical lines made of {color('color1')} and {color('color2')} cells, along with multiple {color('color2')} cells, while the remaining cells are empty (0).",
            "The vertical lines are identical, beginning and ending with a single {color('color2')} cell, with at least three {color('color1')} cells in between.",
            "Both lines are exactly aligned, parallel to each other, and separated by at least two columns.",
            "The {color('color2')} cells mostly appear as single cells and sometimes form 4-way connected objects, with some of the {color('color2')} cells positioned between the two vertical lines."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are always smaller than the input grids.",
            "They are constructed by identifying the two vertical lines and copying the subgrid  that only encloses the two lines and the region between them."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Generate task variables
        taskvars = {}
        taskvars['color1'] = random.randint(1, 9)
        # Ensure color2 is different from color1
        taskvars['color2'] = random.choice([c for c in range(1, 10) if c != taskvars['color1']])
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        for _ in range(num_train_examples):
            # Retry until we generate an input that yields a valid (non-empty) output
            attempts = 0
            while attempts < 100:
                input_grid = self.create_input(taskvars, {})
                output_grid = self.transform_input(input_grid, taskvars)
                # transform_input returns a 1x1 zero grid on failure; detect and retry
                if not (output_grid.shape == (1, 1) and output_grid[0, 0] == 0):
                    train_examples.append({'input': input_grid, 'output': output_grid})
                    break
                attempts += 1
            if attempts >= 100:
                raise RuntimeError('Failed to generate valid training example after 100 attempts')
        
        # Generate 1 test example (also retry until valid)
        attempts = 0
        while attempts < 200:
            test_input_grid = self.create_input(taskvars, {})
            test_output_grid = self.transform_input(test_input_grid, taskvars)
            if not (test_output_grid.shape == (1, 1) and test_output_grid[0, 0] == 0):
                test_examples = [{'input': test_input_grid, 'output': test_output_grid}]
                break
            attempts += 1
        else:
            raise RuntimeError('Failed to generate valid test example after 200 attempts')
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        color1 = taskvars['color1']
        color2 = taskvars['color2']
        
        # Randomly determine grid size (between 5 and 30)
        height = random.randint(7, 30)
        width = random.randint(7, 30)
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Determine vertical line length (at least 5 to accommodate color2 at both ends and 3+ color1 cells)
        line_length = random.randint(5, height - 2)
        
        # Randomly place the first vertical line (but ensure it fits within grid)
        start_row = random.randint(0, height - line_length)
        first_col = random.randint(1, width - 4)  # Leave space for second line
        
        # Determine distance between lines (at least 2 columns)
        distance = random.randint(2, min(width - first_col - 2, 10))
        second_col = first_col + distance + 1
        
        # Create vertical lines
        for i in range(line_length):
            row = start_row + i
            # First and last cells are color2, the rest are color1
            if i == 0 or i == line_length - 1:
                grid[row, first_col] = color2
                grid[row, second_col] = color2
            else:
                grid[row, first_col] = color1
                grid[row, second_col] = color1
        
        # Store the endpoint positions of the vertical lines
        endpoint_positions = [
            (start_row, first_col),
            (start_row, second_col),
            (start_row + line_length - 1, first_col),
            (start_row + line_length - 1, second_col)
        ]
        
        # Define reserved positions (adjacent to endpoints - only for color1 in vertical lines)
        reserved_positions = set()
        for r, c in endpoint_positions:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_r, new_c = r + dr, c + dc
                # Skip if out of bounds
                if new_r < 0 or new_r >= height or new_c < 0 or new_c >= width:
                    continue
                reserved_positions.add((new_r, new_c))
        
        # Remove the positions that are part of the vertical lines (these are allowed to be color1)
        for i in range(1, line_length - 1):
            reserved_positions.discard((start_row + i, first_col))
            reserved_positions.discard((start_row + i, second_col))
        
        # Add random color2 cells, including some between the vertical lines
        num_color2_cells = random.randint(3, (height * width) // 10)
        
        # Ensure at least one color2 cell between the lines
        attempts = 0
        placed_between = False
        while not placed_between and attempts < 50:
            between_row = random.randint(start_row + 1, start_row + line_length - 2)
            between_col = random.randint(first_col + 1, second_col - 1)
            if (between_row, between_col) not in reserved_positions:
                grid[between_row, between_col] = color2
                placed_between = True
            attempts += 1
        
        # Add remaining color2 cells randomly, possibly creating some pairs
        remaining_attempts = 0
        placed_cells = 1  # We already placed one cell
        while placed_cells < num_color2_cells and remaining_attempts < 100:
            row = random.randint(0, height - 1)
            col = random.randint(0, width - 1)
            
            # Skip if it's already filled or in reserved positions
            if grid[row, col] != 0 or (row, col) in reserved_positions:
                remaining_attempts += 1
                continue
            
            # Sometimes create pairs
            if (random.random() < 0.3 and col + 1 < width and 
                grid[row, col + 1] == 0 and 
                (row, col + 1) not in reserved_positions):
                grid[row, col] = color2
                grid[row, col + 1] = color2
                placed_cells += 2
            else:
                grid[row, col] = color2
                placed_cells += 1
                
            remaining_attempts = 0  # Reset attempts after successful placement
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        color1 = taskvars['color1']
        color2 = taskvars['color2']
        
        # Find all vertical lines (defined as contiguous color1 cells with color2 at ends)
        vertical_lines = []
        
        # First, find all color1 connected components
        color1_objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        color1_objects = color1_objects.filter(lambda obj: color1 in obj.colors)
        
        # Find all color2 cells
        color2_cells = set()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == color2:
                    color2_cells.add((r, c))
        
        # Check each color1 object to see if it forms part of a vertical line
        for obj in color1_objects:
            # Get bounding box to check if it's a vertical line
            r_slice, c_slice = obj.bounding_box
            
            # Check if it's a vertical line (width of 1)
            if c_slice.stop - c_slice.start == 1:
                col = c_slice.start
                min_row = r_slice.start
                max_row = r_slice.stop - 1
                
                # Check if there are color2 cells at the top and bottom
                if ((min_row-1, col) in color2_cells and 
                    (max_row+1, col) in color2_cells):
                    # This is a vertical line
                    vertical_lines.append((min_row-1, max_row+1, col))
        
        # We need exactly 2 vertical lines
        if len(vertical_lines) != 2:
            # Fallback in case we can't find exactly 2 lines
            return np.zeros((1, 1), dtype=int)
        
        # Sort by column
        vertical_lines.sort(key=lambda x: x[2])
        
        # Extract the boundaries of the two lines
        left_min_row, left_max_row, left_col = vertical_lines[0]
        right_min_row, right_max_row, right_col = vertical_lines[1]
        
        # Determine the subgrid boundaries
        min_row = min(left_min_row, right_min_row)
        max_row = max(left_max_row, right_max_row)
        
        # Extract the subgrid
        output_grid = grid[min_row:max_row+1, left_col:right_col+1].copy()
        
        return output_grid
    
