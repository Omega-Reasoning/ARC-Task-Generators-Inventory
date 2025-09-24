from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import create_object, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class ARCTask868de0faGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid with dimension n.",
            "There is a minimum of 2 and maximum of 5, 4-way connected objects present in the input grid, each of these are a square whose perimeter is filled with {color('per_color')} and the remaining cells within the perimeter of the square are empty cells(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "Identify all the squares in the output grid, if the number of cells inside the square, i.e. not considering the perimeter but the cells inside the perimeter is even the color it {color('color_1')} else color {color('color_2')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = gridvars['n']
        per_color = taskvars['per_color']
        num_squares = gridvars['num_squares']
        square_sizes = gridvars['square_sizes']
        
        grid = np.zeros((n, n), dtype=int)
        
        def generate_valid_grid():
            test_grid = np.zeros((n, n), dtype=int)
            placed_squares = []
            
            for size in square_sizes:
                # Try to place this square without overlapping others
                max_attempts = 50
                placed = False
                
                for _ in range(max_attempts):
                    # Random position for top-left corner
                    max_row = n - size
                    max_col = n - size
                    if max_row <= 0 or max_col <= 0:
                        continue
                        
                    row = random.randint(0, max_row)
                    col = random.randint(0, max_col)
                    
                    # Check if this square is at least 1 cell away from any existing squares
                    overlaps = False
                    for existing_row, existing_col, existing_size in placed_squares:
                        # Expand both rectangles by 1 cell in all directions for spacing check
                        # Current square bounds (expanded by 1)
                        curr_top = row - 1
                        curr_bottom = row + size
                        curr_left = col - 1
                        curr_right = col + size
                        
                        # Existing square bounds (expanded by 1)
                        exist_top = existing_row - 1
                        exist_bottom = existing_row + existing_size
                        exist_left = existing_col - 1
                        exist_right = existing_col + existing_size
                        
                        # Check if expanded rectangles overlap (meaning squares are too close)
                        if not (curr_right <= exist_left or 
                            exist_right <= curr_left or 
                            curr_bottom <= exist_top or 
                            exist_bottom <= curr_top):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        # Place the square perimeter
                        # Top and bottom edges
                        test_grid[row, col:col+size] = per_color
                        test_grid[row+size-1, col:col+size] = per_color
                        # Left and right edges
                        test_grid[row:row+size, col] = per_color
                        test_grid[row:row+size, col+size-1] = per_color
                        
                        placed_squares.append((row, col, size))
                        placed = True
                        break
                
                if not placed:
                    return None
            
            return test_grid, placed_squares
        
        result = retry(generate_valid_grid, lambda x: x is not None, max_attempts=200)
        return result[0]

    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        per_color = taskvars['per_color']
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        
        # Find all squares by detecting connected components of the perimeter color
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        perimeter_objects = objects.with_color(per_color)
        
        for obj in perimeter_objects:
            # Get bounding box of the object
            bbox = obj.bounding_box
            height = bbox[0].stop - bbox[0].start
            width = bbox[1].stop - bbox[1].start
            
            # Verify this is a square perimeter
            if height == width and height >= 3:
                # Calculate interior area (exclude perimeter)
                interior_cells = (height - 2) * (width - 2)
                
                # Fill interior based on parity
                fill_color = color_1 if interior_cells % 2 == 0 else color_2
                
                # Fill the interior
                start_row = bbox[0].start + 1
                end_row = bbox[0].stop - 1
                start_col = bbox[1].start + 1
                end_col = bbox[1].stop - 1
                
                if start_row < end_row and start_col < end_col:
                    output_grid[start_row:end_row, start_col:end_col] = fill_color
        
        return output_grid

    def _get_square_sizes(self, n: int, num_squares: int) -> List[int]:
        # Generate square sizes ensuring variety
        max_size = n // 2 - 1
        min_size = 3  # Minimum size to have interior
        
        if max_size < min_size:
            max_size = min_size
        
        # Generate sizes ensuring at least one even and one odd interior
        square_sizes = []
        for i in range(num_squares):
            size = random.randint(min_size, max_size)
            square_sizes.append(size)
        
        # Ensure unique sizes
        square_sizes = list(set(square_sizes))
        while len(square_sizes) < min(num_squares, (max_size - min_size + 1)):
            size = random.randint(min_size, max_size)
            if size not in square_sizes:
                square_sizes.append(size)
        
        square_sizes = square_sizes[:num_squares]
        
        # Ensure at least one even and one odd interior area
        has_even = any((size - 2) * (size - 2) % 2 == 0 for size in square_sizes)
        has_odd = any((size - 2) * (size - 2) % 2 == 1 for size in square_sizes)
        
        if not has_even:
            # Replace a size to ensure even interior
            for i, size in enumerate(square_sizes):
                new_size = size + (1 if size % 2 == 1 else -1)
                if new_size >= min_size and new_size <= max_size:
                    if (new_size - 2) * (new_size - 2) % 2 == 0:
                        square_sizes[i] = new_size
                        break
        
        if not has_odd:
            # Replace a size to ensure odd interior
            for i, size in enumerate(square_sizes):
                new_size = size + (1 if size % 2 == 0 else -1)
                if new_size >= min_size and new_size <= max_size:
                    if (new_size - 2) * (new_size - 2) % 2 == 1:
                        square_sizes[i] = new_size
                        break
        
        return square_sizes
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate different colors
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        per_color, color_1, color_2 = all_colors[:3]
        
        
        taskvars = {
            'per_color': per_color,
            'color_1': color_1,
            'color_2': color_2,
        }
        
        # Generate training examples
        num_train = random.randint(4, 5)
        train_examples = []
        for _ in range(num_train):
            n = random.randint(13, 30)
            num_squares = random.randint(2, 5)

            square_sizes = self._get_square_sizes(n, num_squares)
            gridvars = {
                'n': n,
                'num_squares': num_squares,
                'square_sizes': square_sizes
            }

            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        n = random.randint(13, 30)
        num_squares = random.randint(2, 5)
        square_sizes = self._get_square_sizes(n, num_squares)
        gridvars = {
            'n': n,
            'num_squares': num_squares,
            'square_sizes': square_sizes
        }
        # Generate test example
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

