from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry

class Taska699fb00Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain horizontal lines of {color('object_color')} color and empty (0) cells.",
            "The horizontal lines must have alternating cells filled with {color('object_color')} color and unfilled cells with empty (0) cells.",
            "All other cells in the grid are empty (0) cells.",
            "The horizontal lines with alternating pattern can appear in multiple rows of the grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid but here the horizontal line whose alternating empty cells must be filled with {color('fill_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with alternating horizontal patterns."""
        object_color = taskvars['object_color']
        grid_size = gridvars['grid_size']
        
        # Create an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine number of alternating pattern segments to create
        num_patterns = random.randint(1, 3 + grid_size // 5)
        
        # Track which rows already have patterns
        used_rows = set()
        
        # Place patterns with alternating rows (at least one empty row between patterns)
        patterns_placed = 0
        max_attempts = 100
        attempts = 0
        
        while patterns_placed < num_patterns and attempts < max_attempts:
            attempts += 1
            
            # Available rows are those with at least one empty row between them
            available_rows = [r for r in range(grid_size) if all(abs(r - used_r) > 1 for used_r in used_rows)]
            
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
                max_pattern_length = min(grid_size // 2, 7)  # Up to 7 cells or half the row
                
                if max_pattern_length < min_pattern_length:
                    max_pattern_length = min_pattern_length
                
                pattern_length = random.randint(min_pattern_length, max_pattern_length)
                if pattern_length % 2 == 0:  # Ensure odd length
                    pattern_length += 1
                
                # Track columns already used by patterns on this row
                used_cols = set()
                for c in range(grid_size):
                    if grid[row, c] != 0:
                        used_cols.add(c)
                
                # Find valid starting positions with sufficient spacing
                valid_start_positions = []
                min_spacing = 3  # Minimum spacing between patterns on same row
                
                for start_col in range(grid_size - pattern_length + 1):
                    valid = True
                    
                    # Check if any part of the pattern or its buffer would overlap with existing patterns
                    for c in range(start_col - min_spacing, start_col + pattern_length + min_spacing):
                        if 0 <= c < grid_size and c in used_cols:
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
                    if col < grid_size:
                        grid[row, col] = object_color
                        used_cols.add(col)
                
                patterns_placed += 1
                if patterns_placed >= num_patterns:
                    break
        
        # If no patterns were placed (rare but possible), add at least one
        if patterns_placed == 0:
            row = random.randint(0, grid_size - 1)
            start_col = random.randint(0, grid_size - 3)  # Ensure at least 3 cells fit
            
            grid[row, start_col] = object_color
            grid[row, start_col + 2] = object_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by filling alternating gaps with fill color."""
        fill_color = taskvars['fill_color']
        object_color = taskvars['object_color']
        
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
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Generate random colors ensuring they're different
        object_color = random.randint(1, 9)
        fill_color = random.choice([c for c in range(1, 10) if c != object_color])
        
        # Store task variables
        taskvars = {
            'object_color': object_color,
            'fill_color': fill_color,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate unique grid sizes for all grids (training + test)
        min_size = 8  # Minimum size to accommodate patterns
        max_size = 20
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': all_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': all_sizes[-1]}
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