from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject

class Task6cdd2623Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain multiple colored cells randomly distributed throughout the grid using exactly three distinct colors, with all remaining cells being empty (0).",
            "One of these colors, which is called pattern-color, appears exactly four times, forming two same-colored pairs — each pair appears either in a row and a column or in two separate rows.",
            "The pairs are positioned so that one cell appears in the start and one cell at the end of the specific row or column.",
            "If the pairs appear in two rows, the rows must be non-consecutive.",
            "The pairs must not be positioned in any of the grid corners.",
            "The other two colors, different from the pattern-color, appear more frequently—either as isolated single cells or as small groups made of 4-way connected cells using both the colors.",
            "The colors including the pattern-color, must vary across examples"
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the color that appears exactly in four cells, which is called pattern-color.",
            "Remove all other colored cells except the four cells with pattern-color",
            "These four cells form two pairs, with each pair positioned either in a single row or a single column.",
            "Once identified, locate the rows or columns where the two same-colored pairs are placed.",
            "If one pair appears in a row and the other in a column, completely fill both the row and the column with the pattern-color.",
            "If both pairs appear in two different rows, completely fill each of these rows with the pattern-color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Choose consistent grid size for all examples
        rows = random.randint(8, 30)
        cols = random.randint(8, 30)
        
        # We'll create 3-4 train examples
        num_train = random.randint(3, 4)
        
        train_pairs = []
        test_pairs = []
        
        # Create color sets for each example to ensure they vary
        # Need num_train + 2 color sets (for train examples + 2 test cases)
        all_color_sets = []
        
        # Generate unique color sets for each example (training + 2 test examples)
        for _ in range(num_train + 2):
            # Create a new color set that's different from all previous ones
            new_colors = self._generate_distinct_colors(all_color_sets)
            all_color_sets.append(new_colors)
        
        # Add grid dimensions to each color set
        for color_set in all_color_sets:
            color_set['rows'] = rows
            color_set['cols'] = cols
        
        # First example with pairs in a row and column
        train_pairs.append(self._create_row_column_case(all_color_sets[0]))
        
        # Second example with pairs in two non-consecutive rows
        train_pairs.append(self._create_two_rows_case(all_color_sets[1]))
        
        # Additional train examples (random configuration)
        for i in range(2, num_train):
            if random.choice([True, False]):
                train_pairs.append(self._create_row_column_case(all_color_sets[i]))
            else:
                train_pairs.append(self._create_two_rows_case(all_color_sets[i]))
        
        # Test cases (one for each pattern)
        # First test case: row-column pattern
        test_pairs.append(self._create_row_column_case(all_color_sets[num_train]))
        # Second test case: two-rows pattern
        test_pairs.append(self._create_two_rows_case(all_color_sets[num_train + 1]))
        
        # Use grid dimensions for display in the reasoning chain
        taskvars = {
            'rows': rows,
            'cols': cols
        }
        
        return taskvars, {
            'train': train_pairs,
            'test': test_pairs
        }
    
    def _generate_distinct_colors(self, existing_color_sets):
        """Generate a set of colors that differs from all existing sets"""
        while True:
            color1 = random.randint(1, 9)  # Color for the pairs
            color2 = random.randint(1, 9)  # Additional color
            color3 = random.randint(1, 9)  # Additional color
            
            # Make sure all colors in this set are distinct
            if len({color1, color2, color3}) < 3:
                continue
                
            # Check if this color set is unique compared to existing ones
            is_unique = True
            for existing_set in existing_color_sets:
                if (color1 == existing_set['color1'] and 
                    color2 == existing_set['color2'] and 
                    color3 == existing_set['color3']):
                    is_unique = False
                    break
            
            if is_unique:
                return {
                    'color1': color1,
                    'color2': color2,
                    'color3': color3
                }
    
    def _create_row_column_case(self, taskvars):
        """Create a case where pairs are in one row and one column"""
        gridvars = {'pattern': 'row_column'}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        return {'input': input_grid, 'output': output_grid}
    
    def _create_two_rows_case(self, taskvars):
        """Create a case where pairs are in two non-consecutive rows"""
        gridvars = {'pattern': 'two_rows'}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        return {'input': input_grid, 'output': output_grid}
    
    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        color1 = taskvars['color1']  # The color that will appear exactly 4 times
        color2 = taskvars['color2']
        color3 = taskvars['color3']
        
        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        pattern = gridvars.get('pattern', random.choice(['row_column', 'two_rows']))
        
        if pattern == 'row_column':
            # Place pairs in one row and one column
            # First ensure we're not using the first or last row/column to avoid corners
            available_rows = list(range(1, rows-1))
            available_cols = list(range(1, cols-1))
            
            row = random.choice(available_rows)
            col = random.choice(available_cols)
            
            # Place pair in row
            grid[row, 0] = color1
            grid[row, cols-1] = color1
            
            # Place pair in column
            grid[0, col] = color1
            grid[rows-1, col] = color1
            
        elif pattern == 'two_rows':
            # Place pairs in two non-consecutive rows
            # First ensure we're not using the first or last row to avoid corners
            available_rows = list(range(1, rows-1))
            
            # Choose two non-consecutive rows
            if len(available_rows) < 3:
                # Fallback if grid is too small
                row1 = 1
                row2 = rows - 2
            else:
                row1 = random.choice(available_rows)
                available_rows = [r for r in available_rows if abs(r - row1) > 1]
                if not available_rows:
                    # Fallback if no suitable second row
                    row2 = 0 if row1 > rows//2 else rows-1
                else:
                    row2 = random.choice(available_rows)
            
            # Place pair in first row
            grid[row1, 0] = color1
            grid[row1, cols-1] = color1
            
            # Place pair in second row
            grid[row2, 0] = color1
            grid[row2, cols-1] = color1
        
        # Now add the other colors with higher frequency
        # First identify cells that can be filled (not already containing color1)
        fillable_positions = [(r, c) for r in range(rows) for c in range(cols) 
                             if grid[r, c] == 0]
        
        # Determine how many cells to fill with colors 2 and 3
        # We want them to appear more frequently than color1 (which appears 4 times)
        num_color2 = random.randint(8, min(15, len(fillable_positions)//2))
        num_color3 = random.randint(8, min(15, len(fillable_positions) - num_color2))
        
        # Select positions for colors 2 and 3
        random.shuffle(fillable_positions)
        for i, (r, c) in enumerate(fillable_positions):
            if i < num_color2:
                grid[r, c] = color2
            elif i < num_color2 + num_color3:
                grid[r, c] = color3
            else:
                break
        
        # Now create some 2-3 cell groups of connected cells with different colors
        # Find existing color2 and color3 cells that are isolated
        for color in [color2, color3]:
            isolated_cells = []
            for r in range(rows):
                for c in range(cols):
                    if grid[r, c] == color:
                        # Check if isolated (no neighbors of same color)
                        neighbors = [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                                    if 0 <= r+dr < rows and 0 <= c+dc < cols]
                        if not any(grid[nr, nc] == color for nr, nc in neighbors):
                            isolated_cells.append((r, c))
            
            # For some isolated cells, add a connected different color
            for r, c in isolated_cells[:min(3, len(isolated_cells))]:
                # Find empty neighbors
                empty_neighbors = [(r+dr, c+dc) for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]
                                  if 0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr, c+dc] == 0]
                
                if empty_neighbors:
                    nr, nc = random.choice(empty_neighbors)
                    # Add different color (not color1)
                    other_color = color3 if color == color2 else color2
                    grid[nr, nc] = other_color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Initialize the output grid with zeros (clear all colors)
        output = np.zeros_like(grid)
        
        # Count occurrences of each color
        color_counts = {}
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                color = grid[r, c]
                if color > 0:  # Skip background
                    color_counts[color] = color_counts.get(color, 0) + 1
        
        # Find the color that appears exactly 4 times (pattern-color)
        target_color = None
        for color, count in color_counts.items():
            if count == 4:
                target_color = color
                break
        
        if target_color is None:
            return output  # No color with exactly 4 occurrences
        
        # Find the positions of the target color
        target_positions = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == target_color:
                    target_positions.append((r, c))
                    # Keep the pattern-color cells in the output
                    output[r, c] = target_color
        
        # Group by rows and columns
        rows_with_targets = {}
        cols_with_targets = {}
        
        for r, c in target_positions:
            rows_with_targets[r] = rows_with_targets.get(r, 0) + 1
            cols_with_targets[c] = cols_with_targets.get(c, 0) + 1
        
        # Find rows and columns with pairs
        pair_rows = [r for r, count in rows_with_targets.items() if count == 2]
        pair_cols = [c for c, count in cols_with_targets.items() if count == 2]
        
        # Apply transformation based on pattern detected
        if len(pair_rows) == 1 and len(pair_cols) == 1:
            # One row and one column pattern
            row, col = pair_rows[0], pair_cols[0]
            # Fill the row
            output[row, :] = target_color
            # Fill the column
            output[:, col] = target_color
        elif len(pair_rows) == 2:
            # Two rows pattern
            for row in pair_rows:
                output[row, :] = target_color
        
        return output