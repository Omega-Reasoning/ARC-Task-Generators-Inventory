from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Optional

class Task15696249Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "Each input grid is completely filled with cells using exactly {vars['grid_size']} different colors.",
            "In each example, exactly one entire row or one entire column is filled with a single color.",
            "The position of this fully colored row or column varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {vars['grid_size'] * vars['grid_size']} x {vars['grid_size'] * vars['grid_size']}.",
            "To construct the output, first identify the row or column in the input grid that is completely filled with a single color.",
            "The output grid is formed by tiling the input grid {vars['grid_size']} times along the same axis — depending on whether a row or a column is fully colored.",
            "If it is a row, the input grid is copied horizontally {vars['grid_size']} times, starting at the corresponding row position in the output grid.",
            "If it is a column, the input grid is copied vertically {vars['grid_size']} times, starting at the corresponding column position in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create input grid with exactly one fully colored row or column."""
        size = taskvars['grid_size']
        
        # Generate enough distinct colors
        available_colors = list(range(1, 10))  # Colors 1-9, avoiding 0 (background)
        random.shuffle(available_colors)
        colors = available_colors[:size]
        
        def is_valid_grid(grid):
            """Check if grid has exactly one fully colored row OR column, and all others are mixed."""
            # Count fully colored rows
            fully_colored_rows = 0
            for i in range(size):
                if len(np.unique(grid[i, :])) == 1:
                    fully_colored_rows += 1
            
            # Count fully colored columns  
            fully_colored_cols = 0
            for j in range(size):
                if len(np.unique(grid[:, j])) == 1:
                    fully_colored_cols += 1
            
            # Should have exactly one fully colored row OR one fully colored column, but not both
            return (fully_colored_rows == 1 and fully_colored_cols == 0) or \
                   (fully_colored_rows == 0 and fully_colored_cols == 1)
        
        def generate_valid_grid():
            # Decide whether to fill a row or column completely
            fill_row = random.choice([True, False])
            
            if fill_row:
                target_row = random.randint(0, size - 1)
                return self._create_grid_with_colored_row(size, colors, target_row)
            else:
                target_col = random.randint(0, size - 1)
                return self._create_grid_with_colored_col(size, colors, target_col)
        
        # Generate grid ensuring exactly one row or column is fully colored
        grid = retry(
            generate_valid_grid,
            is_valid_grid,
            max_attempts=200
        )
        
        return grid
    
    def _create_grid_with_colored_row(self, size: int, colors: List[int], target_row: int) -> np.ndarray:
        """Create grid with one fully colored row and mixed colors elsewhere."""
        grid = np.zeros((size, size), dtype=int)
        
        # Fill the target row with a single color
        row_color = random.choice(colors)
        grid[target_row, :] = row_color
        
        # Fill other rows ensuring they have mixed colors
        for i in range(size):
            if i == target_row:
                continue  # Skip the already filled target row
                
            # Ensure this row has at least 2 different colors
            row_colors = random.sample(colors, min(len(colors), random.randint(2, len(colors))))
            
            # Fill row with mixed colors
            for j in range(size):
                grid[i, j] = random.choice(row_colors)
        
        # Now ensure columns are mixed (except we can't change the target row)
        for j in range(size):
            col_values = grid[:, j]
            if len(np.unique(col_values)) == 1:
                # This column is monochromatic, we need to change some cells
                # but we can't change the target_row
                changeable_positions = [pos for pos in range(size) if pos != target_row]
                if changeable_positions:
                    # Change at least one cell in this column to make it mixed
                    pos_to_change = random.choice(changeable_positions)
                    current_color = grid[pos_to_change, j]
                    new_colors = [c for c in colors if c != current_color]
                    if new_colors:
                        grid[pos_to_change, j] = random.choice(new_colors)
        
        return grid
    
    def _create_grid_with_colored_col(self, size: int, colors: List[int], target_col: int) -> np.ndarray:
        """Create grid with one fully colored column and mixed colors elsewhere."""
        grid = np.zeros((size, size), dtype=int)
        
        # Fill the target column with a single color
        col_color = random.choice(colors)
        grid[:, target_col] = col_color
        
        # Fill other columns ensuring they have mixed colors
        for j in range(size):
            if j == target_col:
                continue  # Skip the already filled target column
                
            # Ensure this column has at least 2 different colors
            col_colors = random.sample(colors, min(len(colors), random.randint(2, len(colors))))
            
            # Fill column with mixed colors
            for i in range(size):
                grid[i, j] = random.choice(col_colors)
        
        # Now ensure rows are mixed (except we can't change the target column)
        for i in range(size):
            row_values = grid[i, :]
            if len(np.unique(row_values)) == 1:
                # This row is monochromatic, we need to change some cells
                # but we can't change the target_col
                changeable_positions = [pos for pos in range(size) if pos != target_col]
                if changeable_positions:
                    # Change at least one cell in this row to make it mixed
                    pos_to_change = random.choice(changeable_positions)
                    current_color = grid[i, pos_to_change]
                    new_colors = [c for c in colors if c != current_color]
                    if new_colors:
                        grid[i, pos_to_change] = random.choice(new_colors)
        
        return grid
    
    def _find_fully_colored_line(self, grid: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """Find the fully colored row or column in the input grid.
        
        Returns:
            Tuple of (fully_colored_row, fully_colored_col) where one is None and the other contains the index
        """
        size = grid.shape[0]
        fully_colored_row = None
        fully_colored_col = None
        
        # Check for fully colored rows
        for i in range(size):
            row = grid[i, :]
            if len(np.unique(row)) == 1:  # All cells in row have same color
                fully_colored_row = i
                break
        
        # Check for fully colored columns
        for j in range(size):
            col = grid[:, j]
            if len(np.unique(col)) == 1:  # All cells in column have same color
                fully_colored_col = j
                break
        
        return fully_colored_row, fully_colored_col
    
    def _create_tiled_output(self, grid: np.ndarray, fully_colored_row: Optional[int], 
                            fully_colored_col: Optional[int], size: int) -> np.ndarray:
        """Create output grid by tiling the input grid based on the fully colored line.
        
        Args:
            grid: Input grid to tile
            fully_colored_row: Index of fully colored row (None if no such row)
            fully_colored_col: Index of fully colored column (None if no such column)
            size: Size of the input grid
            
        Returns:
            Output grid with tiled pattern
        """
        output_size = size * size
        output = np.zeros((output_size, output_size), dtype=int)
        
        if fully_colored_row is not None:
            # Tile horizontally - copy input grid 'size' times horizontally
            # starting at the corresponding row position
            start_row = fully_colored_row * size
            for i in range(size):
                start_col = i * size
                output[start_row:start_row + size, start_col:start_col + size] = grid
        
        elif fully_colored_col is not None:
            # Tile vertically - copy input grid 'size' times vertically
            # starting at the corresponding column position
            start_col = fully_colored_col * size
            for i in range(size):
                start_row = i * size
                output[start_row:start_row + size, start_col:start_col + size] = grid
        
        return output
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by tiling based on fully colored row/column."""
        size = taskvars['grid_size']
        
        # Function 1: Find the fully colored row or column
        fully_colored_row, fully_colored_col = self._find_fully_colored_line(grid)
        
        # Function 2: Create output grid by copying input grid and pasting to respective position in output
        output = self._create_tiled_output(grid, fully_colored_row, fully_colored_col, size)
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and generate train/test grids."""
        
        # Initialize task variables
        taskvars = {
            'grid_size': random.choice([3, 4, 5]),
            # Adding the color variables mentioned in constraints even though they're not used
            'object_color': random.choice(range(1, 10)),
            'object_color2': random.choice(range(1, 10)), 
            'strip_color1': random.choice(range(1, 10)),
            'strip_color2': random.choice(range(1, 10)),
            'background_color': random.choice(range(1, 10))
        }
        
        # Ensure all constraint colors are different
        colors_used = set()
        for key in ['object_color', 'object_color2', 'strip_color1', 'strip_color2', 'background_color']:
            while taskvars[key] in colors_used:
                taskvars[key] = random.choice(range(1, 10))
            colors_used.add(taskvars[key])
        
        # Generate train examples
        train_examples = []
        for _ in range(3):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
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

