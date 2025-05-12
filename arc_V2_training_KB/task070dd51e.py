from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Set

class Task070dd51eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each grid contains several pairs of same-colored cells and empty (0) cells.",
            "To construct the input grids, first create vertical and horizontal lines, which are atleast three cells long.",
            "Sometimes, vertical and horizontal lines intersect perpendicularly.",
            "No two vertical or horizontal lines are placed consecutively without a gap.",
            "Each grid contains exactly one {color('line_color1')} line, one {color('line_color2')} line and a few more lines colored differently.",
            "After drawing the lines, the middle cells of each line are removed, keeping only the first and last cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all pairs of same-colored cells.",
            "Once identified, connect each pair of same-colored cells by forming either a vertical or horizontal line, depending on their relative positions.",
            "If two lines intersect perpendicularly, the vertical line takes priority, and the overlapping cell adopts the color of the vertical line."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'line_color1': random.randint(1, 9),
            'line_color2': random.randint(1, 9)
        }
        
        # Ensure line_color1 and line_color2 are different
        while taskvars['line_color1'] == taskvars['line_color2']:
            taskvars['line_color2'] = random.randint(1, 9)
        
        # Create 3-6 training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Create gridvars for this specific example
            gridvars = self._create_gridvars(taskvars)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_gridvars = self._create_gridvars(taskvars)
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def _create_gridvars(self, taskvars: Dict[str, Any]) -> Dict[str, Any]:
        """Create grid-specific variables"""
        # Choose grid size between 10 and 20
        rows = random.randint(10, 20)  
        cols = random.randint(10, 20)  
        
        # Choose number of color pairs (excluding line_color1 and line_color2)
        num_pairs = random.randint(2, 4)
        
        # Choose colors for additional pairs
        colors = list(range(1, 10))
        colors.remove(taskvars['line_color1'])
        colors.remove(taskvars['line_color2'])
        pair_colors = random.sample(colors, num_pairs)
        
        return {
            'rows': rows,
            'cols': cols,
            'pair_colors': pair_colors
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        max_attempts = 50
        for attempt in range(max_attempts):
            # Try to create a valid grid
            grid = self._try_create_input(taskvars, gridvars)
            if grid is not None:
                return grid
        
        # If we couldn't create a valid grid after max attempts, 
        # try with simplified constraints
        return self._fallback_create_input(taskvars, gridvars)
    
    def _try_create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = gridvars['rows']
        cols = gridvars['cols']
        pair_colors = gridvars['pair_colors']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Track which rows and columns have been assigned a color
        row_color = {}  # row_idx -> color value
        col_color = {}  # col_idx -> color value
        
        # Track all planned cell placements
        placements = []  # (row, col, color)
        
        # Create pairs
        all_colors = [taskvars['line_color1'], taskvars['line_color2']] + pair_colors
        placed_colors = set()
        has_intersection = False
        
        # First, place a vertical line using line_color1
        vertical_col = random.randint(0, cols - 1)
        available_rows = [r for r in range(rows) if r not in row_color]
        if len(available_rows) < 2:
            return None  # Not enough rows
        
        # Ensure at least 2 rows between the vertical endpoints
        if len(available_rows) < 4:  # Need at least 4 rows for proper spacing
            return None
            
        # Get two rows with at least 2 rows between them
        available_rows.sort()
        row_pairs = [(r1, r2) for i, r1 in enumerate(available_rows) 
                    for r2 in available_rows[i+3:]]
        if not row_pairs:
            return None
            
        vertical_row1, vertical_row2 = random.choice(row_pairs)
        
        # Mark these rows and this column as used by line_color1
        row_color[vertical_row1] = taskvars['line_color1']
        row_color[vertical_row2] = taskvars['line_color1']
        col_color[vertical_col] = taskvars['line_color1']
        
        # Add the placements
        placements.append((vertical_row1, vertical_col, taskvars['line_color1']))
        placements.append((vertical_row2, vertical_col, taskvars['line_color1']))
        placed_colors.add(taskvars['line_color1'])
        
        # Now place a horizontal line using line_color2
        # Choose a row between the vertical endpoints for intersection
        intersect_row = random.randint(vertical_row1 + 1, vertical_row2 - 1)
        
        # Make sure this row isn't already used
        if intersect_row in row_color:
            return None
            
        # Find available columns for the horizontal line
        available_cols = [c for c in range(cols) if c != vertical_col and c not in col_color]
        if len(available_cols) < 2:
            return None
            
        # Get two columns with the vertical_col between them (for intersection)
        left_cols = [c for c in available_cols if c < vertical_col]
        right_cols = [c for c in available_cols if c > vertical_col]
        
        if not left_cols or not right_cols:
            return None
            
        horizontal_col1 = random.choice(left_cols)
        horizontal_col2 = random.choice(right_cols)
        
        # Mark this row and these columns as used by line_color2
        row_color[intersect_row] = taskvars['line_color2']
        col_color[horizontal_col1] = taskvars['line_color2']
        col_color[horizontal_col2] = taskvars['line_color2']
        
        # Add the placements
        placements.append((intersect_row, horizontal_col1, taskvars['line_color2']))
        placements.append((intersect_row, horizontal_col2, taskvars['line_color2']))
        placed_colors.add(taskvars['line_color2'])
        
        # We've created an intersection
        has_intersection = True
        
        # Now add additional colored pairs
        for color in pair_colors:
            # Decide randomly whether to make a vertical or horizontal pair
            if random.choice([True, False]):
                # Try to make a vertical pair
                available_cols = [c for c in range(cols) if c not in col_color]
                if not available_cols:
                    continue
                    
                col = random.choice(available_cols)
                
                # Find available rows
                available_rows = [r for r in range(rows) if r not in row_color]
                if len(available_rows) < 2:
                    continue
                    
                # Find row pairs with at least one row between them
                available_rows.sort()
                row_pairs = [(r1, r2) for i, r1 in enumerate(available_rows) 
                            for r2 in available_rows[i+2:]]
                if not row_pairs:
                    continue
                    
                row1, row2 = random.choice(row_pairs)
                
                # Mark these rows and this column as used by this color
                row_color[row1] = color
                row_color[row2] = color
                col_color[col] = color
                
                # Add the placements
                placements.append((row1, col, color))
                placements.append((row2, col, color))
                placed_colors.add(color)
                
            else:
                # Try to make a horizontal pair
                available_rows = [r for r in range(rows) if r not in row_color]
                if not available_rows:
                    continue
                    
                row = random.choice(available_rows)
                
                # Find available columns
                available_cols = [c for c in range(cols) if c not in col_color]
                if len(available_cols) < 2:
                    continue
                    
                # Find column pairs with at least one column between them
                available_cols.sort()
                col_pairs = [(c1, c2) for i, c1 in enumerate(available_cols) 
                            for c2 in available_cols[i+2:]]
                if not col_pairs:
                    continue
                    
                col1, col2 = random.choice(col_pairs)
                
                # Mark this row and these columns as used by this color
                row_color[row] = color
                col_color[col1] = color
                col_color[col2] = color
                
                # Add the placements
                placements.append((row, col1, color))
                placements.append((row, col2, color))
                placed_colors.add(color)
        
        # Verify we have enough colored pairs and an intersection
        if len(placed_colors) < 3 or not has_intersection:  # At least 3 colors including line_color1 and line_color2
            return None
        
        # Place all cells in the grid
        for r, c, color in placements:
            grid[r, c] = color
        
        return grid
    
    def _fallback_create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create a simpler grid if the full algorithm fails"""
        rows = gridvars['rows']
        cols = gridvars['cols']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Create a simple vertical line for line_color1
        col = cols // 3
        row1 = rows // 4
        row2 = 3 * rows // 4
        grid[row1, col] = taskvars['line_color1']
        grid[row2, col] = taskvars['line_color1']
        
        # Create a simple horizontal line for line_color2
        row = (row1 + row2) // 2  # Midpoint for intersection
        col1 = cols // 5
        col2 = 4 * cols // 5
        grid[row, col1] = taskvars['line_color2']
        grid[row, col2] = taskvars['line_color2']
        
        # Add one more color for a third pair (minimum requirement)
        third_color = gridvars['pair_colors'][0]
        row3 = rows // 6
        col3 = 2 * cols // 3
        col4 = 4 * cols // 5
        grid[row3, col3] = third_color
        grid[row3, col4] = third_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Copy the input grid
        output_grid = grid.copy()
        
        # Find all pairs of same-colored cells
        rows, cols = grid.shape
        color_positions: Dict[int, List[Tuple[int, int]]] = {}
        
        # Group positions by color
        for r in range(rows):
            for c in range(cols):
                color = grid[r, c]
                if color != 0:
                    if color not in color_positions:
                        color_positions[color] = []
                    color_positions[color].append((r, c))
        
        # First process horizontal lines
        for color, positions in color_positions.items():
            # Process pairs of positions
            if len(positions) == 2:  # Each color should have exactly 2 positions
                r1, c1 = positions[0]
                r2, c2 = positions[1]
                
                if r1 == r2:  # Horizontal line
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        output_grid[r1, c] = color
        
        # Then process vertical lines (they take priority at intersections)
        for color, positions in color_positions.items():
            if len(positions) == 2:  # Each color should have exactly 2 positions
                r1, c1 = positions[0]
                r2, c2 = positions[1]
                
                if c1 == c2:  # Vertical line
                    for r in range(min(r1, r2), max(r1, r2) + 1):
                        # Vertical lines always take priority
                        output_grid[r, c1] = color
        
        return output_grid

