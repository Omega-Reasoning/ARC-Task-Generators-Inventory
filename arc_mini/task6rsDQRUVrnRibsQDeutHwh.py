from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import create_object, retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task6rsDQRUVrnRibsQDeutHwhGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of different sizes.",
            "Each input grid contains exactly one {color('strip')} strip, consisting of 2 or 3 connected cells.",
            "These cells are connected either vertically or horizontally, forming either a vertical strip or a horizontal strip."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grid and adding {color('boundary')} cells around the strip.",
            "If the {color('strip')} strip is vertical: Place two same-sized {color('boundary')} strips to the left and right of the vertical {color('strip')} strip so that all three strips are vertically aligned. Add one {color('boundary')} cell above and one below the vertical {color('strip')} strip.",
            "If the {color('strip')} strip is horizontal: Place two same-sized {color('boundary')} strips above and below the horizontal {color('strip')} strip so that all three strips are horizontally aligned. Add one {color('boundary')} cell to the left and one to the right of the horizontal {color('strip')} strip.",
            "If the {color('strip')} strip is placed so that all {color('boundary')} cells cannot be added just add all those possible."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        strip_color = taskvars['strip']
        grid_size = gridvars.get('grid_size', random.randint(5, 12))
        strip_length = gridvars.get('strip_length', random.choice([2, 3]))
        is_vertical = gridvars.get('is_vertical', random.choice([True, False]))
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Generate strip position that leaves room for boundary cells
        if is_vertical:
            # For vertical strip, need space for left/right strips and top/bottom cells
            min_col = 2  # space for left strip
            max_col = grid_size - 3  # space for right strip
            min_row = 1  # space for top cell
            max_row = grid_size - strip_length - 1  # space for bottom cell
            
            if min_col <= max_col and min_row <= max_row:
                start_row = random.randint(min_row, max_row)
                start_col = random.randint(min_col, max_col)
                
                # Place vertical strip
                for i in range(strip_length):
                    grid[start_row + i, start_col] = strip_color
            else:
                # Fallback: place strip anywhere it fits
                start_row = random.randint(0, grid_size - strip_length)
                start_col = random.randint(0, grid_size - 1)
                for i in range(strip_length):
                    grid[start_row + i, start_col] = strip_color
        else:
            # For horizontal strip, need space for top/bottom strips and left/right cells
            min_row = 2  # space for top strip
            max_row = grid_size - 3  # space for bottom strip
            min_col = 1  # space for left cell
            max_col = grid_size - strip_length - 1  # space for right cell
            
            if min_row <= max_row and min_col <= max_col:
                start_row = random.randint(min_row, max_row)
                start_col = random.randint(min_col, max_col)
                
                # Place horizontal strip
                for i in range(strip_length):
                    grid[start_row, start_col + i] = strip_color
            else:
                # Fallback: place strip anywhere it fits
                start_row = random.randint(0, grid_size - 1)
                start_col = random.randint(0, grid_size - strip_length)
                for i in range(strip_length):
                    grid[start_row, start_col + i] = strip_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        strip_color = taskvars['strip']
        boundary_color = taskvars['boundary']
        
        # Find the strip
        objects = find_connected_objects(output_grid, diagonal_connectivity=False, background=0)
        strip_objects = objects.with_color(strip_color)
        
        if len(strip_objects) == 0:
            return output_grid
            
        strip = strip_objects[0]
        strip_coords = list(strip.coords)
        
        # Determine if strip is vertical or horizontal
        rows = [r for r, c in strip_coords]
        cols = [c for r, c in strip_coords]
        
        is_vertical = len(set(cols)) == 1  # All cells in same column
        is_horizontal = len(set(rows)) == 1  # All cells in same row
        
        if is_vertical:
            # Vertical strip: add strips to left/right, cells above/below
            strip_col = cols[0]
            min_row, max_row = min(rows), max(rows)
            strip_length = max_row - min_row + 1
            
            # Add boundary strips to left and right
            for offset in [-1, 1]:  # left and right
                new_col = strip_col + offset
                if 0 <= new_col < output_grid.shape[1]:
                    for row in range(min_row, max_row + 1):
                        if 0 <= row < output_grid.shape[0]:
                            output_grid[row, new_col] = boundary_color
            
            # Add boundary cells above and below
            for row_offset in [-1, 1]:  # above and below
                new_row = min_row + row_offset if row_offset == -1 else max_row + row_offset
                if 0 <= new_row < output_grid.shape[0]:
                    output_grid[new_row, strip_col] = boundary_color
                    
        elif is_horizontal:
            # Horizontal strip: add strips above/below, cells left/right
            strip_row = rows[0]
            min_col, max_col = min(cols), max(cols)
            strip_length = max_col - min_col + 1
            
            # Add boundary strips above and below
            for offset in [-1, 1]:  # above and below
                new_row = strip_row + offset
                if 0 <= new_row < output_grid.shape[0]:
                    for col in range(min_col, max_col + 1):
                        if 0 <= col < output_grid.shape[1]:
                            output_grid[new_row, col] = boundary_color
            
            # Add boundary cells to left and right
            for col_offset in [-1, 1]:  # left and right
                new_col = min_col + col_offset if col_offset == -1 else max_col + col_offset
                if 0 <= new_col < output_grid.shape[1]:
                    output_grid[strip_row, new_col] = boundary_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        strip_color = random.choice(available_colors)
        available_colors.remove(strip_color)
        boundary_color = random.choice(available_colors)
        
        taskvars = {
            'strip': strip_color,
            'boundary': boundary_color
        }
        
        # Generate training examples with variety
        train_examples = []
        num_train = random.randint(3, 5)
        
        for _ in range(num_train):
            gridvars = {
                'grid_size': random.randint(6, 15),
                'strip_length': random.choice([2, 3]),
                'is_vertical': random.choice([True, False])
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_gridvars = {
            'grid_size': random.randint(8, 20),
            'strip_length': random.choice([2, 3]),
            'is_vertical': random.choice([True, False])
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
