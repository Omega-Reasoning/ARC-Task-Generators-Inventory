from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple
from input_library import create_object, random_cell_coloring, retry
from transformation_library import find_connected_objects, GridObject

class AlternatingPatternFillGenerator(ARCTaskGenerator):
    def __init__(self):
        self.input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain horizontal lines of {color(\"object_color\")} color and empty (0) cells.",
            "The horizontal lines must have alternating cells filled with {color(\"object_color\")} color and unfilled cells with empty (0) cells.",
            "All other cells in the grid are empty (0) cells.",
            "The horizontal lines with alternating pattern can appear in multiple rows of the grid."
        ]
        
        self.transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid but here the horizontal line whose alternating empty cells must be filled with {color(\"fill_color\")} color."
        ]
        
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)
    
    def create_input(self, gridvars=None):
        if gridvars is None:
            gridvars = {}
        
        # Generate random grid size between 5 and 20 (square grid)
        size = gridvars.get('size', random.randint(5, 20))
        
        # Generate random object color between 1 and 9
        object_color = gridvars.get('object_color', random.randint(1, 9))
        
        # Create an empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # Determine number of alternating pattern segments to create
        num_patterns = gridvars.get('num_patterns', random.randint(1, 3 + size // 5))
        
        # Track which rows already have patterns
        used_rows = set()
        
        # Place patterns with alternating rows (at least one empty row between patterns)
        patterns_placed = 0
        max_attempts = 100
        attempts = 0
        
        while patterns_placed < num_patterns and attempts < max_attempts:
            attempts += 1
            
            # Available rows are those with at least one empty row between them
            available_rows = [r for r in range(size) if all(abs(r - used_r) > 1 for used_r in used_rows)]
            
            if not available_rows:
                break  # No more suitable rows available
            
            # Choose a random available row
            row = random.choice(available_rows)
            used_rows.add(row)
            
            # Now decide if we want to place multiple patterns on this row
            num_patterns_for_row = random.randint(1, 2)  # Max 2 patterns per row
            
            for pattern_idx in range(num_patterns_for_row):
                # Minimum pattern length (must have at least 2 alternating cells)
                min_pattern_length = 3  # At least 2 colored cells with 1 empty between
                
                # Maximum pattern length (to ensure patterns don't take up the whole row)
                max_pattern_length = min(size // 2, 7)  # Up to 7 cells or half the row
                
                if max_pattern_length < min_pattern_length:
                    max_pattern_length = min_pattern_length
                
                pattern_length = random.randint(min_pattern_length, max_pattern_length)
                if pattern_length % 2 == 0:  # Ensure odd length
                    pattern_length += 1
                
                # Track columns already used by patterns on this row
                used_cols = set()
                for c in range(size):
                    if grid[row, c] != 0:
                        used_cols.add(c)
                
                # Find valid starting positions with sufficient spacing
                valid_start_positions = []
                min_spacing = 3  # Minimum spacing between patterns on same row
                
                for start_col in range(size - pattern_length + 1):
                    valid = True
                    
                    # Check if any part of the pattern or its buffer would overlap with existing patterns
                    for c in range(start_col - min_spacing, start_col + pattern_length + min_spacing):
                        if 0 <= c < size and c in used_cols:
                            valid = False
                            break
                    
                    if valid:
                        valid_start_positions.append(start_col)
                
                if not valid_start_positions:
                    break  # No valid positions for another pattern on this row
                
                # Choose a random valid starting position
                start_col = random.choice(valid_start_positions)
                
                # Place the alternating pattern
                for offset in range(0, pattern_length, 2):
                    col = start_col + offset
                    if col < size:
                        grid[row, col] = object_color
                        used_cols.add(col)
                
                patterns_placed += 1
                if patterns_placed >= num_patterns:
                    break
        
        # If no patterns were placed (rare but possible), add at least one
        if patterns_placed == 0:
            row = random.randint(0, size - 1)
            start_col = random.randint(0, size - 3)  # Ensure at least 3 cells fit
            
            grid[row, start_col] = object_color
            grid[row, start_col + 2] = object_color
        
        return grid
    
    def transform_input(self, grid, gridvars=None):
        if gridvars is None:
            gridvars = {}
        
        # Get the object color and determine the fill color
        object_color = gridvars.get('object_color')
        fill_color = gridvars.get('fill_color')
        
        # If colors were not provided, detect them from the grid
        if object_color is None:
            non_zero_values = set(grid.flatten()) - {0}
            if non_zero_values:
                object_color = list(non_zero_values)[0]
            else:
                object_color = 1  # Default if not found
        
        # Ensure fill color is different from object color
        if fill_color is None or fill_color == object_color:
            fill_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Process grid row by row
        for r in range(grid.shape[0]):
            # First find all colored cells that are part of an alternating pattern
            pattern_cells = []
            for c in range(grid.shape[1]):
                if grid[r, c] == object_color:
                    pattern_cells.append(c)
            
            # Now find pairs of adjacent pattern cells that need fill color in between
            for i in range(len(pattern_cells) - 1):
                # Check if these form a pair with exactly one empty cell between them
                if pattern_cells[i+1] - pattern_cells[i] == 2:
                    # Fill the empty cell in between
                    output_grid[r, pattern_cells[i] + 1] = fill_color
        
        return output_grid
    
    def create_grids(self):
        # Generate random colors ensuring they're different
        object_color = random.randint(1, 9)
        fill_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        # Store grid variables
        gridvars = {
            'object_color': object_color,
            'fill_color': fill_color
        }
        
        # Generate 3-5 training pairs with varying grid sizes
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train_pairs):
            # Random square grid size between 5 and 20
            size = random.randint(5, 20)
            
            # Set variables for this specific grid
            grid_specific_vars = {
                'size': size,
                'num_patterns': random.randint(1, 3 + size // 5),
                'object_color': object_color,
                'fill_color': fill_color
            }
            
            # Create input and transform it
            input_grid = self.create_input(grid_specific_vars)
            output_grid = self.transform_input(input_grid, grid_specific_vars)
            
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_size = random.randint(5, 20)
        
        test_grid_vars = {
            'size': test_size,
            'num_patterns': random.randint(1, 3 + test_size // 5),
            'object_color': object_color,
            'fill_color': fill_color
        }
        
        test_input = self.create_input(test_grid_vars)
        test_output = self.transform_input(test_input, test_grid_vars)
        
        # Create the TrainTestData object
        data = TrainTestData(train=train_pairs, test=[GridPair(input=test_input, output=test_output)])
        
        return gridvars, data

# Test the generator
if __name__ == "__main__":
    generator = AlternatingPatternFillGenerator()
    gridvars, data = generator.create_grids()
    print("Grid variables:", gridvars)
    print(f"Generated {len(data.train)} training pairs and {len(data.test)} test pairs.")
    generator.visualize_train_test_data(data)