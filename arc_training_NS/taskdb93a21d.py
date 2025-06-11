from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskdb93a21d(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes but are always square-shaped.",
            "Each grid contains a random number of square-shaped regions, all filled with the same color: {color('color_1')}.",
            "The sizes of the squares are always even numbers, and may differ across squares within the same grid.",
            "Each square is surrounded by a border of empty cells (0) that is half the size of the square.",
            "The empty border around each square is unique and non-overlapping with any other square — no two squares share any border cell.",
            "Below each square and its left and right borders, all cells in those columns are empty (0) from the bottom edge of the border down to the bottom of the grid.",
            "The remaining cells are all empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All square-shaped regions are identified.",
            "For each square, a border is first added around it using {color('color_2')}, with a thickness equal to half the side length of the square.",
            "Subsequently, a vertical fill is applied: all cells in the same column of the square (Excluding the borders), from the bottom edge of the border to the bottom of the grid, are filled with {color('color_3')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'color_1': random.randint(1, 9),  # Square color
            'color_2': random.randint(1, 9),  # Border color  
            'color_3': random.randint(1, 9)   # Fill color
        }
        
        # Ensure all colors are different
        while taskvars['color_2'] == taskvars['color_1']:
            taskvars['color_2'] = random.randint(1, 9)
        while taskvars['color_3'] in [taskvars['color_1'], taskvars['color_2']]:
            taskvars['color_3'] = random.randint(1, 9)
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_data = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        def generate_valid_grid():
            grid_size = random.randint(12, 30)  # Ensure room for squares, borders, and fills
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Determine number of squares (2-4 to ensure multiple squares)
            num_squares = random.randint(2, 4)
            
            placed_squares = []
            
            for _ in range(num_squares):
                # Try to place a square
                attempts = 100
                placed = False
                
                for _ in range(attempts):
                    # Generate even square size
                    square_size = random.choice([2, 4, 6])
                    border_size = square_size // 2
                    
                    # Calculate total area needed: square + border + some space for vertical fill
                    total_width = square_size + 2 * border_size
                    total_height = square_size + 2 * border_size
                    
                    # Ensure there's space for the square, its border, and vertical fill below
                    min_space_below = 2  # At least some space for vertical fill
                    
                    max_start_row = grid_size - total_height - min_space_below
                    max_start_col = grid_size - total_width
                    
                    if max_start_row < 0 or max_start_col < 0:
                        continue
                    
                    # Random position for the square (accounting for border)
                    square_start_row = random.randint(border_size, max_start_row + border_size)
                    square_start_col = random.randint(border_size, max_start_col + border_size)
                    
                    # Define the full region (square + border + vertical fill area)
                    border_start_row = square_start_row - border_size
                    border_end_row = square_start_row + square_size + border_size
                    border_start_col = square_start_col - border_size
                    border_end_col = square_start_col + square_size + border_size
                    
                    # Check for overlaps with existing squares' regions
                    overlaps = False
                    for existing in placed_squares:
                        # Check if border regions overlap
                        if not (border_end_col <= existing['border_start_col'] or 
                               border_start_col >= existing['border_end_col'] or
                               border_end_row <= existing['border_start_row'] or
                               border_start_row >= existing['border_end_row']):
                            overlaps = True
                            break
                        
                        # Check if vertical fill columns overlap
                        if not (border_end_col <= existing['border_start_col'] or 
                               border_start_col >= existing['border_end_col']):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        # Place the square
                        grid[square_start_row:square_start_row + square_size, 
                             square_start_col:square_start_col + square_size] = taskvars['color_1']
                        
                        # Record the placement
                        placed_squares.append({
                            'square_start_row': square_start_row,
                            'square_start_col': square_start_col,
                            'square_size': square_size,
                            'border_start_row': border_start_row,
                            'border_end_row': border_end_row,
                            'border_start_col': border_start_col,
                            'border_end_col': border_end_col
                        })
                        placed = True
                        break
                
                if not placed and len(placed_squares) == 0:
                    return None
            
            # Ensure we have at least 2 squares and at least one has space for vertical fill
            if len(placed_squares) < 2:
                return None
            
            # Check that at least one square has space below for filling
            has_fill_space = False
            for square in placed_squares:
                if square['border_end_row'] < grid_size - 1:
                    has_fill_space = True
                    break
            
            if not has_fill_space:
                return None
                
            return grid
        
        # Use retry to ensure we get a valid grid
        return retry(generate_valid_grid, lambda x: x is not None, max_attempts=200)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        
        # Find all square objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        square_objects = objects.with_color(taskvars['color_1'])
        
        for square_obj in square_objects:
            # Get square properties
            bbox = square_obj.bounding_box
            square_start_row = bbox[0].start
            square_end_row = bbox[0].stop
            square_start_col = bbox[1].start
            square_end_col = bbox[1].stop
            square_size = square_end_row - square_start_row
            
            border_thickness = square_size // 2
            
            # Calculate border region
            border_start_row = square_start_row - border_thickness
            border_end_row = square_end_row + border_thickness
            border_start_col = square_start_col - border_thickness
            border_end_col = square_end_col + border_thickness
            
            # Ensure we don't go out of bounds
            border_start_row = max(0, border_start_row)
            border_end_row = min(grid.shape[0], border_end_row)
            border_start_col = max(0, border_start_col)
            border_end_col = min(grid.shape[1], border_end_col)
            
            # Add border (only on empty cells, don't overwrite the square)
            for r in range(border_start_row, border_end_row):
                for c in range(border_start_col, border_end_col):
                    # Only fill if it's empty and not part of the original square
                    if (output_grid[r, c] == 0 and 
                        not (square_start_row <= r < square_end_row and 
                             square_start_col <= c < square_end_col)):
                        output_grid[r, c] = taskvars['color_2']
            
            # Vertical fill ONLY in the columns of the square itself (not the border)
            # From the bottom edge of the border to the bottom of the grid
            for col in range(square_start_col, square_end_col):  # Only square columns, not border columns
                for row in range(border_end_row, grid.shape[0]):
                    if output_grid[row, col] == 0:  # Only fill empty cells
                        output_grid[row, col] = taskvars['color_3']
        
        return output_grid
