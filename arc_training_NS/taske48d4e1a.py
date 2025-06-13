from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taske48d4e1a(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "In each grid, the top m cells of the rightmost column are colored {color('bar_color')}, where m is a positive integer.",
            "Each grid contains a cross shape colored of a random color, with its arms extending to touch the grid boundaries.",
            "The cross is positioned so that its center does not share any row or any column with the bar cells.",
            "The center of the cross is placed such that moving m cells down along the anti diagonal from this center remains within the grid boundaries."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The bar in the rightmost column and the cross shape are identified.",
            "The length of the bar, denoted by m, is determined.",
            "The entire cross shape is then shifted m cells down along the anti-diagonal (from top-right to bottom-left) relative to its center position."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        bar_color = taskvars['bar_color']
        cross_color = gridvars['cross_color']  # Now comes from gridvars (changes per grid)
        bar_length = gridvars['bar_length']
        cross_center_row = gridvars['cross_center_row']
        cross_center_col = gridvars['cross_center_col']
        
        # Create empty grid
        grid = np.zeros((n, n), dtype=int)
        
        # Create bar in rightmost column (top m cells)
        for i in range(bar_length):
            grid[i, n-1] = bar_color
        
        # Create cross shape centered at (cross_center_row, cross_center_col)
        # Horizontal arm (extends to boundaries)
        for c in range(n):
            grid[cross_center_row, c] = cross_color
        
        # Vertical arm (extends to boundaries)
        for r in range(n):
            grid[r, cross_center_col] = cross_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        bar_color = taskvars['bar_color']
        
        # Copy input grid
        output_grid = grid.copy()
        
        # Find bar length by counting colored cells in rightmost column
        bar_length = 0
        for i in range(n):
            if grid[i, n-1] == bar_color:
                bar_length += 1
            else:
                break
        
        # Find cross center and color by finding intersection of horizontal and vertical lines
        cross_center_row = None
        cross_center_col = None
        cross_color = None
        
        # Find cross by looking for lines that span the entire grid
        for r in range(n):
            # Check if this row forms a horizontal line (ignoring bar color)
            row_colors = set(grid[r, :])
            row_colors.discard(0)  # Remove background
            row_colors.discard(bar_color)  # Remove bar color
            if len(row_colors) == 1:  # Only one color remains
                potential_cross_color = list(row_colors)[0]
                # Check if this color spans most of the row (allowing for bar interference)
                cross_cells = np.sum(grid[r, :] == potential_cross_color)
                if cross_cells >= n - 1:  # Allow for one cell to be bar color
                    cross_center_row = r
                    cross_color = potential_cross_color
                    break
        
        # Find cross center column
        if cross_color is not None:
            for c in range(n):
                cross_cells = np.sum(grid[:, c] == cross_color)
                if cross_cells >= n - 1:  # Allow for interference
                    cross_center_col = c
                    break
        
        if cross_center_row is None or cross_center_col is None or cross_color is None:
            return output_grid  # Should not happen with valid input
        
        # Clear the old cross completely
        for c in range(n):
            if output_grid[cross_center_row, c] == cross_color:
                output_grid[cross_center_row, c] = 0
        for r in range(n):
            if output_grid[r, cross_center_col] == cross_color:
                output_grid[r, cross_center_col] = 0
        
        # Calculate new cross center position (shift by bar_length along anti-diagonal)
        # Anti-diagonal direction: down and left, so (+bar_length, -bar_length)
        new_center_row = cross_center_row + bar_length
        new_center_col = cross_center_col - bar_length
        
        # Draw new cross at shifted position - but only if both center coordinates are in bounds
        if 0 <= new_center_row < n and 0 <= new_center_col < n:
            # Horizontal arm (full width)
            for c in range(n):
                output_grid[new_center_row, c] = cross_color
            
            # Vertical arm (full height)
            for r in range(n):
                output_grid[r, new_center_col] = cross_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        n = random.randint(8, 15)  # Grid size
        
        # Choose bar color (must be non-zero)
        bar_color = random.randint(1, 9)
        
        taskvars = {
            'n': n,
            'bar_color': bar_color
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            def generate_valid_grid():
                # Choose cross color (must be different from bar color and non-zero)
                available_colors = list(range(1, 10))
                available_colors.remove(bar_color)
                cross_color = random.choice(available_colors)
                
                # Bar length between 1 and n//3 to leave room for cross and shift
                bar_length = random.randint(1, max(1, n//3))
                
                # Cross center must not share row/column with bar cells (rows 0 to bar_length-1, column n-1)
                # Also, shifting by bar_length along anti-diagonal must stay in bounds
                
                valid_rows = []
                for r in range(n):
                    if r < bar_length:  # Shares row with bar
                        continue
                    if r + bar_length >= n:  # Shift would go out of bounds (row)
                        continue
                    valid_rows.append(r)
                
                valid_cols = []
                for c in range(n):
                    if c == n-1:  # Shares column with bar
                        continue
                    if c - bar_length < 0:  # Shift would go out of bounds (column)
                        continue
                    valid_cols.append(c)
                
                if not valid_rows or not valid_cols:
                    return None
                
                cross_center_row = random.choice(valid_rows)
                cross_center_col = random.choice(valid_cols)
                
                return {
                    'bar_length': bar_length,
                    'cross_center_row': cross_center_row,
                    'cross_center_col': cross_center_col,
                    'cross_color': cross_color
                }
            
            gridvars = retry(
                generate_valid_grid,
                lambda x: x is not None,
                max_attempts=100
            )
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        def generate_valid_test_grid():
            # Choose cross color (must be different from bar color and non-zero)
            available_colors = list(range(1, 10))
            available_colors.remove(bar_color)
            cross_color = random.choice(available_colors)
            
            bar_length = random.randint(1, max(1, n//3))
            
            valid_rows = []
            for r in range(n):
                if r < bar_length:
                    continue
                if r + bar_length >= n:
                    continue
                valid_rows.append(r)
            
            valid_cols = []
            for c in range(n):
                if c == n-1:
                    continue
                if c - bar_length < 0:
                    continue
                valid_cols.append(c)
            
            if not valid_rows or not valid_cols:
                return None
            
            cross_center_row = random.choice(valid_rows)
            cross_center_col = random.choice(valid_cols)
            
            return {
                'bar_length': bar_length,
                'cross_center_row': cross_center_row,
                'cross_center_col': cross_center_col,
                'cross_color': cross_color
            }
        
        test_gridvars = retry(
            generate_valid_test_grid,
            lambda x: x is not None,
            max_attempts=100
        )
        
        test_input = self.create_input(taskvars, test_gridvars)
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
