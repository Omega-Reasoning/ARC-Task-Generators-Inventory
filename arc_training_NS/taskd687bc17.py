from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taskd687bc17(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of rectangular shape and can have different sizes.",
            "Each grid contains four colored bars: a vertical bar of color {color('first_column')} in the first column, a vertical bar of color {color('last_column')} in the last column, a horizontal bar of color {color('first_row')} in the first row, and a horizontal bar of color {color('last_row')} in the last row.",
            "All four corner cells are empty (0).",
            "Additionally, each input grid includes a random number of single-colored cells placed within the internal area (excluding the bars and corners) in the colors {color('first_column')}, {color('last_column')}, {color('first_row')}, and {color('last_row')}.",
            "The number of single-colored cells for each of these four colors varies.",
            "Each grid may contain up to five single-colored cells of random colors other than the four specified.",
            "Cells of color {color('first_column')} and {color('last_column')} never share the same row with another cell of the same color.",
            "Cells of color {color('first_row')} and {color('last_row')} never share the same column with another cell of the same color.",
            "The row indices of cells with colors {color('first_column')} or {color('last_column')} are never equal to the column indices of cells with colors {color('first_row')} or {color('last_row')}.",
            "All remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The colors of the four bars in the grid are identified: the vertical bar in the first column is {color('first_column')}, the vertical bar in the last column is {color('last_column')}, the horizontal bar in the first row is {color('first_row')}, and the horizontal bar in the last row is {color('last_row')}.",
            "All single-colored cells placed within the internal area are identified, and any of them with a color other than the four identified colors are replaced with empty cells (0).",
            "The remaining single-colored cells are moved close to their corresponding bars: cells of color {color('first_column')} are moved to the second column in their current row; cells of color {color('last_column')} are moved to the second-to-last column in their current row; cells of color {color('first_row')} are moved to the second row in their current column; and cells of color {color('last_row')} are moved to the second-to-last row in their current column.",
            "All four corners and the internal area—excluding the bar positions and the relocated single-colored cells—are empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random colors for the four bars
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        
        taskvars = {
            'first_column': all_colors[0],
            'last_column': all_colors[1], 
            'first_row': all_colors[2],
            'last_row': all_colors[3]
        }
        
        # Generate 3-6 training examples and 1 test example
        num_train = random.randint(3, 6)
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Random grid size between 5x5 and 30x30
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        
        grid = np.zeros((height, width), dtype=int)
        
        # Extract bar colors
        first_col_color = taskvars['first_column']
        last_col_color = taskvars['last_column']
        first_row_color = taskvars['first_row']
        last_row_color = taskvars['last_row']
        
        # Create the four bars, leaving corners empty
        # First column (vertical bar)
        grid[1:height-1, 0] = first_col_color
        
        # Last column (vertical bar)
        grid[1:height-1, width-1] = last_col_color
        
        # First row (horizontal bar)
        grid[0, 1:width-1] = first_row_color
        
        # Last row (horizontal bar)
        grid[height-1, 1:width-1] = last_row_color
        
        # Add internal colored cells for each of the four bar colors
        bar_colors = [first_col_color, last_col_color, first_row_color, last_row_color]
        
        # Get available internal positions (excluding bars and corners)
        internal_positions = []
        for r in range(1, height-1):
            for c in range(1, width-1):
                internal_positions.append((r, c))
        
        # Track used positions and constraints
        used_positions = set()
        used_rows_for_col_colors = {first_col_color: set(), last_col_color: set()}
        used_cols_for_row_colors = {first_row_color: set(), last_row_color: set()}
        # Track which row/column indices are forbidden for the cross-constraint
        forbidden_rows_for_col_colors = set()  # rows where col colors can't be placed
        forbidden_cols_for_row_colors = set()  # cols where row colors can't be placed
        
        # Place cells for each bar color
        for color in bar_colors:
            num_cells = random.randint(0, 3)  # 0-3 cells per color
            
            for _ in range(num_cells):
                # Find valid positions for this color
                valid_positions = []
                
                for r, c in internal_positions:
                    if (r, c) in used_positions:
                        continue
                    
                    # Check constraints based on color type
                    if color in [first_col_color, last_col_color]:
                        # Vertical bar colors: can't share row with same color
                        if r in used_rows_for_col_colors[color]:
                            continue
                        # Row index can't be same as any column index used by horizontal colors
                        if r in forbidden_rows_for_col_colors:
                            continue
                    
                    if color in [first_row_color, last_row_color]:
                        # Horizontal bar colors: can't share column with same color
                        if c in used_cols_for_row_colors[color]:
                            continue
                        # Column index can't be same as any row index used by vertical colors
                        if c in forbidden_cols_for_row_colors:
                            continue
                    
                    valid_positions.append((r, c))
                
                if valid_positions:
                    r, c = random.choice(valid_positions)
                    grid[r, c] = color
                    used_positions.add((r, c))
                    
                    # Update constraint tracking
                    if color in [first_col_color, last_col_color]:
                        used_rows_for_col_colors[color].add(r)
                        forbidden_cols_for_row_colors.add(r)  # No row colors can use column index = r
                    if color in [first_row_color, last_row_color]:
                        used_cols_for_row_colors[color].add(c)
                        forbidden_rows_for_col_colors.add(c)  # No col colors can use row index = c
        
        # Add up to 5 random colored cells with other colors
        other_colors = [c for c in range(1, 10) if c not in bar_colors]
        num_random_cells = random.randint(0, 5)
        
        for _ in range(num_random_cells):
            available_positions = [(r, c) for r, c in internal_positions 
                                 if (r, c) not in used_positions]
            if available_positions and other_colors:
                r, c = random.choice(available_positions)
                color = random.choice(other_colors)
                grid[r, c] = color
                used_positions.add((r, c))
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        height, width = grid.shape
        
        # Extract bar colors
        first_col_color = taskvars['first_column']
        last_col_color = taskvars['last_column']
        first_row_color = taskvars['first_row']
        last_row_color = taskvars['last_row']
        
        bar_colors = {first_col_color, last_col_color, first_row_color, last_row_color}
        
        # Clear internal area first (keeping bars)
        for r in range(1, height-1):
            for c in range(1, width-1):
                output_grid[r, c] = 0
        
        # Find and process internal colored cells
        for r in range(1, height-1):
            for c in range(1, width-1):
                cell_color = grid[r, c]
                
                if cell_color == 0:
                    continue
                
                # Remove cells with colors not in bar colors
                if cell_color not in bar_colors:
                    continue
                
                # Move cells closer to their corresponding bars
                if cell_color == first_col_color:
                    # Move to second column
                    output_grid[r, 1] = cell_color
                elif cell_color == last_col_color:
                    # Move to second-to-last column
                    output_grid[r, width-2] = cell_color
                elif cell_color == first_row_color:
                    # Move to second row
                    output_grid[1, c] = cell_color
                elif cell_color == last_row_color:
                    # Move to second-to-last row
                    output_grid[height-2, c] = cell_color
        
        return output_grid
