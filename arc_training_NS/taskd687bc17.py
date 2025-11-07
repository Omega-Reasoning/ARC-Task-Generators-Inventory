from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from typing import Dict, Any, Tuple, List
import numpy as np
import random

class TaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of rectangular shape and can have different sizes.",
            "Each grid contains four colored bars: a vertical bar of a random color (called first_column_color) in the first column, a vertical bar of another color (called last_column_color) in the last column, a horizontal bar of another color (called first_row_color) in the first row, and a horizontal bar of another color (called last_row_color) in the last row.",
            "One of the 4 bars are of color {color('color')}.",
            "All four corner cells are empty (0).",
            "Additionally, each input grid includes a random number of single-colored cells placed within the internal area (excluding the two outermost rows and two outermost columns on each side) in the colors first_column_color, last_column_color, first_row_color, and last_row_color.",
            "The number of single-colored cells for each of these four colors varies.",
            "Each grid may contain up to five single-colored cells of random colors other than the four specified.",
            "Cells of color first_column_color and last_column_color never share the same row with another cell of the same color.",
            "Cells of color first_row_color and last_row_color never share the same column with another cell of the same color.",
            "All remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The colors of the four bars in the grid are identified: the vertical bar in the first column is first_column_color, the vertical bar in the last column is last_column_color, the horizontal bar in the first row is first_row_color, and the horizontal bar in the last row is last_row_color.",
            "All single-colored cells placed within the internal area are identified, and any of them with a color other than the four identified colors are replaced with empty cells (0).",
            "The remaining single-colored cells are moved closer to their corresponding bars: cells of color first_column_color are moved to the second column in their current row; cells of color last_column_color are moved to the second-to-last column in their current row; cells of color first_row_color are moved to the second row in their current column; and cells of color last_row_color are moved to the second-to-last row in their current column.",
            "All four corners and the internal area—excluding the bar positions and the relocated single-colored cells—are empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        height = gridvars['height']
        width = gridvars['width']
        first_column_color = gridvars['first_column_color']
        last_column_color = gridvars['last_column_color']
        first_row_color = gridvars['first_row_color']
        last_row_color = gridvars['last_row_color']
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Add four bars (excluding corners)
        grid[1:height-1, 0] = first_column_color  # First column (vertical bar)
        grid[1:height-1, width-1] = last_column_color  # Last column (vertical bar)
        grid[0, 1:width-1] = first_row_color  # First row (horizontal bar)
        grid[height-1, 1:width-1] = last_row_color  # Last row (horizontal bar)
        
        # Define internal area (excluding the two outermost rows and columns on each side)
        internal_row_start = 2
        internal_row_end = height - 2
        internal_col_start = 2
        internal_col_end = width - 2
        
        # Skip if grid is too small for internal area
        if internal_row_end <= internal_row_start or internal_col_end <= internal_col_start:
            return grid
        
        bar_colors = [first_column_color, last_column_color, first_row_color, last_row_color]
        
        # Track which rows/columns are used for each color to enforce constraints
        used_rows_first_col = set()
        used_rows_last_col = set()
        used_cols_first_row = set()
        used_cols_last_row = set()
        
        # Track occupied cells
        occupied = set()
        
        # Add cells for first_column_color and last_column_color
        for color, used_rows in [(first_column_color, used_rows_first_col), 
                                  (last_column_color, used_rows_last_col)]:
            num_cells = random.randint(0, min(5, internal_row_end - internal_row_start))
            for _ in range(num_cells):
                # Find valid row (not already used for this color)
                valid_rows = [r for r in range(internal_row_start, internal_row_end) 
                             if r not in used_rows]
                if not valid_rows:
                    break
                row = random.choice(valid_rows)
                used_rows.add(row)
                
                # Find valid column
                valid_cols = [c for c in range(internal_col_start, internal_col_end) 
                             if (row, c) not in occupied]
                if valid_cols:
                    col = random.choice(valid_cols)
                    grid[row, col] = color
                    occupied.add((row, col))
        
        # Add cells for first_row_color and last_row_color
        for color, used_cols in [(first_row_color, used_cols_first_row), 
                                  (last_row_color, used_cols_last_row)]:
            num_cells = random.randint(0, min(5, internal_col_end - internal_col_start))
            for _ in range(num_cells):
                # Find valid column (not already used for this color)
                valid_cols = [c for c in range(internal_col_start, internal_col_end) 
                             if c not in used_cols]
                if not valid_cols:
                    break
                col = random.choice(valid_cols)
                used_cols.add(col)
                
                # Find valid row
                valid_rows = [r for r in range(internal_row_start, internal_row_end) 
                             if (r, col) not in occupied]
                if valid_rows:
                    row = random.choice(valid_rows)
                    grid[row, col] = color
                    occupied.add((row, col))
        
        # Add up to 5 cells of random colors (other than the four bar colors)
        other_colors = [c for c in range(1, 10) if c not in bar_colors]
        num_other_cells = random.randint(0, min(5, len(other_colors)))
        for _ in range(num_other_cells):
            # Find valid position in internal area
            valid_positions = [(r, c) for r in range(internal_row_start, internal_row_end) 
                              for c in range(internal_col_start, internal_col_end)
                              if (r, c) not in occupied]
            if valid_positions and other_colors:
                row, col = random.choice(valid_positions)
                color = random.choice(other_colors)
                grid[row, col] = color
                occupied.add((row, col))
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        height, width = grid.shape
        
        # Identify the four bar colors
        first_column_color = grid[1, 0] if height > 1 else 0
        last_column_color = grid[1, width - 1] if height > 1 and width > 1 else 0
        first_row_color = grid[0, 1] if width > 1 else 0
        last_row_color = grid[height - 1, 1] if height > 1 and width > 1 else 0
        
        bar_colors = {first_column_color, last_column_color, first_row_color, last_row_color}
        
        # Clear internal area (will rebuild with moved cells)
        for r in range(2, height - 2):
            for c in range(2, width - 2):
                output[r, c] = 0
        
        # Process cells in the internal area
        for r in range(2, height - 2):
            for c in range(2, width - 2):
                cell_color = grid[r, c]
                if cell_color == 0:
                    continue
                
                # Remove cells with colors not in the four bar colors
                if cell_color not in bar_colors:
                    continue
                
                # Move cells closer to their corresponding bars
                if cell_color == first_column_color:
                    output[r, 1] = cell_color
                elif cell_color == last_column_color:
                    output[r, width - 2] = cell_color
                elif cell_color == first_row_color:
                    output[1, c] = cell_color
                elif cell_color == last_row_color:
                    output[height - 2, c] = cell_color
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Choose a fixed color that will appear in one of the bars across all examples
        fixed_color = random.randint(1, 9)
        taskvars = {'color': fixed_color}
        
        # Generate 3-6 train examples and 1 test example
        num_train = random.randint(3, 6)
        
        train_pairs = []
        for _ in range(num_train):
            # Generate random grid dimensions (at least 7 to have meaningful internal area)
            height = random.randint(7, 30)
            width = random.randint(7, 30)
            
            # Generate four different colors for the bars
            # Ensure one of them is the fixed color
            all_colors = [c for c in range(1, 10) if c != fixed_color]
            random.shuffle(all_colors)
            other_three_colors = all_colors[:3]
            
            # Randomly assign the fixed color to one of the four bar positions
            bar_colors = other_three_colors + [fixed_color]
            random.shuffle(bar_colors)
            
            gridvars = {
                'height': height,
                'width': width,
                'first_column_color': bar_colors[0],
                'last_column_color': bar_colors[1],
                'first_row_color': bar_colors[2],
                'last_row_color': bar_colors[3]
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        height = random.randint(7, 30)
        width = random.randint(7, 30)
        all_colors = [c for c in range(1, 10) if c != fixed_color]
        random.shuffle(all_colors)
        other_three_colors = all_colors[:3]
        
        # Randomly assign the fixed color to one of the four bar positions
        bar_colors = other_three_colors + [fixed_color]
        random.shuffle(bar_colors)
        
        gridvars = {
            'height': height,
            'width': width,
            'first_column_color': bar_colors[0],
            'last_column_color': bar_colors[1],
            'first_row_color': bar_colors[2],
            'last_row_color': bar_colors[3]
        }
        
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_pairs, 'test': test_pairs}
