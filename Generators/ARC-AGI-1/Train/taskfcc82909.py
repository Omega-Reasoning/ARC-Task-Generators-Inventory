from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, List, Any, Tuple

class Taskfcc82909Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}",
            "A random number of 2×2 squares is placed in the grid. The number of such squares is randomly chosen between 2 and {vars['n']} // 2.",
            "Each 2×2 square contains between 1 and 4 different colors, selected randomly.",
            "Squares are positioned such that: All cells in the same columns directly beneath the square are empty (contain 0). The number of empty rows below each square is at least equal to the number of colors used in that square. Squares do not touch one another (including diagonally)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All 2×2 squares in the input grid are identified, along with the number of distinct colors used in each square.",
            "For each square, a number of rows equal to the number of colors used in that square are colored with {color('output_color')} in the same columns as the square, starting directly below it."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'n': random.randint(10, 30),  # Grid size between 10-30 for manageable complexity
            'output_color': random.randint(1, 9)  # Color for the output rows
        }
        
        # Create training examples with varying numbers of squares
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Vary the number of squares and colors used across examples
            available_colors = [c for c in range(1, 10) if c != taskvars['output_color']]
            color_palette = random.sample(available_colors, k=min(len(available_colors), random.randint(4, 8)))
            
            # Number of squares between 2 and n//2
            min_squares = 2
            max_squares = max(2, taskvars['n'] // 2)
            
            gridvars = {
                'num_squares': random.randint(min_squares, max_squares),
                'color_palette': color_palette
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        available_colors = [c for c in range(1, 10) if c != taskvars['output_color']]
        test_color_palette = random.sample(available_colors, k=min(len(available_colors), random.randint(4, 8)))
        
        min_squares = 2
        max_squares = max(2, taskvars['n'] // 2)
        
        test_gridvars = {
            'num_squares': random.randint(min_squares, max_squares),
            'color_palette': test_color_palette
        }
        
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
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        num_squares = gridvars['num_squares']
        color_palette = gridvars['color_palette']
        
        def generate_grid():
            grid = np.zeros((n, n), dtype=int)
            placed_squares = []  # Store (row, col, num_colors) for each successfully placed square
            occupied_columns = set()  # Track columns that are already occupied by squares
            
            for attempt in range(num_squares):
                # Try to place a square
                def try_place_square():
                    # Determine number of colors for this square first
                    max_colors = min(4, len(color_palette))
                    num_colors = random.randint(1, max_colors)
                    
                    # We need at least num_colors empty rows below the square
                    # So the square can be placed at most at row (n - 2 - num_colors)
                    max_start_row = n - 2 - num_colors
                    
                    if max_start_row < 0:
                        return None
                    
                    # Try random positions
                    for _ in range(100):  # Max attempts per square
                        row = random.randint(0, max_start_row)
                        col = random.randint(0, n - 2)  # Room for 2x2 square
                        
                        # Check if the columns for this square are already occupied
                        if col in occupied_columns or (col + 1) in occupied_columns:
                            continue
                        
                        # Check if this position conflicts with existing squares
                        # Squares should not touch (including diagonally)
                        conflict = False
                        for existing_row, existing_col, _ in placed_squares:
                            # The new square occupies rows [row, row+1] and cols [col, col+1]
                            # The existing square occupies rows [existing_row, existing_row+1] and cols [existing_col, existing_col+1]
                            
                            # For squares not to touch, there must be at least 1 empty cell between them
                            # This means the bounding boxes with 1-cell padding should not overlap
                            
                            # Check if they're separated horizontally (at least 1 cell gap)
                            horizontally_separated = (col + 1 < existing_col - 1) or (existing_col + 1 < col - 1)
                            
                            # Check if they're separated vertically (at least 1 cell gap)
                            vertically_separated = (row + 1 < existing_row - 1) or (existing_row + 1 < row - 1)
                            
                            # If neither separated horizontally nor vertically, they touch
                            if not (horizontally_separated or vertically_separated):
                                conflict = True
                                break
                        
                        if not conflict:
                            # Verify the entire area below this square is clear
                            # (ALL rows from row+2 to bottom must be empty in these columns)
                            area_clear = True
                            for check_row in range(row + 2, n):
                                for check_col in range(col, col + 2):
                                    if grid[check_row, check_col] != 0:
                                        area_clear = False
                                        break
                                if not area_clear:
                                    break
                            
                            # Also verify we have at least num_colors empty rows
                            if area_clear and (n - (row + 2)) >= num_colors:
                                return row, col, num_colors
                    
                    return None
                
                placement = try_place_square()
                if placement is None:
                    continue
                
                row, col, num_colors = placement
                
                # Choose colors for this square
                square_colors = random.sample(color_palette, k=num_colors)
                
                # Fill the 2x2 square ensuring we use exactly num_colors distinct colors
                if num_colors == 1:
                    # All cells same color
                    for i in range(2):
                        for j in range(2):
                            grid[row + i, col + j] = square_colors[0]
                else:
                    # Distribute colors to ensure all are used
                    positions = [(0,0), (0,1), (1,0), (1,1)]
                    random.shuffle(positions)
                    
                    # Assign each color to at least one position
                    for i, color in enumerate(square_colors):
                        pos_i, pos_j = positions[i]
                        grid[row + pos_i, col + pos_j] = color
                    
                    # Fill remaining positions if any
                    for i in range(num_colors, 4):
                        pos_i, pos_j = positions[i]
                        grid[row + pos_i, col + pos_j] = random.choice(square_colors)
                
                placed_squares.append((row, col, num_colors))
                # Mark these columns as occupied
                occupied_columns.add(col)
                occupied_columns.add(col + 1)
            
            return grid
        
        return retry(generate_grid, lambda g: np.any(g != 0), max_attempts=50)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        output_color = taskvars['output_color']
        n = grid.shape[0]
        
        # Find all 2x2 squares by checking each possible position
        squares = []
        for row in range(n - 1):
            for col in range(n - 1):
                # Check if this forms a 2x2 square (all cells non-zero)
                square_region = grid[row:row+2, col:col+2]
                if np.all(square_region != 0):
                    # Count distinct colors in this square
                    unique_colors = len(np.unique(square_region))
                    
                    # Check if this is a new square (not part of a larger pattern)
                    # We'll consider it a square if it's the top-left corner of a 2x2 region
                    is_new_square = True
                    
                    # Check if this square overlaps with already found squares
                    for existing_row, existing_col, _ in squares:
                        if (abs(row - existing_row) < 2 and abs(col - existing_col) < 2):
                            is_new_square = False
                            break
                    
                    if is_new_square:
                        squares.append((row, col, unique_colors))
        
        # For each square, color rows below it
        for row, col, num_colors in squares:
            # Color 'num_colors' rows below the square in the same columns
            start_row = row + 2  # Start right below the square
            for i in range(num_colors):
                if start_row + i < n:
                    output_grid[start_row + i, col] = output_color
                    output_grid[start_row + i, col + 1] = output_color
        
        return output_grid