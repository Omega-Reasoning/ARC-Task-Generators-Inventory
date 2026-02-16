from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects

class Task40853293Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids can have different sizes.",
            "They contain five pairs of colored cells, where each colored cell is completely surrounded by empty (0) cells.",
            "Cells in the same pair share the same color and appear in the same row or column, with at least three empty (0) cells between them.",
            "Each pair has a unique color that varies across examples, except for {color('cell_color1')} and {color('cell_color2')}, which remain consistent.",
            "Each input grid contains two pairs arranged so that one appears in a row and the other in a column, ensuring they intersect when connected."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all five pairs.",
            "Once the pairs are identified, they are connected using colored cells that match the respective pair.",
            "This process forms vertical and horizontal lines.",
            "If two lines overlap, vertical lines take priority over horizontal lines."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize cell_color1 randomly from 1-9
        cell_color1 = random.randint(1, 9)
        
        # Initialize cell_color2 to be different from cell_color1
        cell_color2 = random.choice([i for i in range(1, 10) if i != cell_color1])
        
        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2
        }
        
        # Number of train pairs (3 or 4)
        num_train = random.randint(3, 4)
        
        # Create train grids
        train_grids = []
        
        # Special case 1: Two plus shapes and one separate vertical line
        gridvars1 = {
            "pattern": "two_plus_one_vertical",
            "height": random.randint(15, 25),
            "width": random.randint(15, 25)
        }
        input_grid1 = self.create_input(taskvars, gridvars1)
        output_grid1 = self.transform_input(input_grid1, taskvars)
        train_grids.append({"input": input_grid1, "output": output_grid1})
        
        # Special case 2: One plus and one horizontal line cut by two vertical lines
        gridvars2 = {
            "pattern": "one_plus_cut_horizontals",
            "height": random.randint(15, 25),
            "width": random.randint(15, 25)
        }
        input_grid2 = self.create_input(taskvars, gridvars2)
        output_grid2 = self.transform_input(input_grid2, taskvars)
        train_grids.append({"input": input_grid2, "output": output_grid2})
        
        # Additional random case(s) if needed - ensure at least one plus
        for i in range(num_train - 2):
            gridvars = {
                "pattern": "random_with_plus",
                "height": random.randint(15, 25),
                "width": random.randint(15, 25)
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_grids.append({"input": input_grid, "output": output_grid})
        
        # Create test grid - ensure at least one plus
        test_gridvars = {
            "pattern": "random_with_plus",
            "height": random.randint(15, 25),
            "width": random.randint(15, 25)
        }
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            "train": train_grids,
            "test": [{"input": test_input, "output": test_output}]
        }
    
    def create_input(self, taskvars, gridvars):
        height = gridvars.get("height", random.randint(15, 25))
        width = gridvars.get("width", random.randint(15, 25))
        pattern = gridvars.get("pattern", "random_with_plus")
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Choose unique colors for each pair
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]
        used_colors = [cell_color1, cell_color2]
        remaining_colors = [i for i in range(1, 10) if i not in used_colors]
        random.shuffle(remaining_colors)
        
        # We need 5 colors total
        pair_colors = used_colors + remaining_colors[:3]
        random.shuffle(pair_colors)  # Shuffle all colors
        
        # Determine which pairs will be horizontal vs vertical
        if pattern == "two_plus_one_vertical":
            # 2 horizontal pairs, 3 vertical pairs with careful positioning for two intersections
            horizontal_indices = [0, 1]
            vertical_indices = [2, 3, 4]
            min_intersections = 2
        elif pattern == "one_plus_cut_horizontals":
            # 3 horizontal pairs, 2 vertical pairs with careful positioning
            horizontal_indices = [0, 1, 2]
            vertical_indices = [3, 4]
            min_intersections = 1
        elif pattern == "random_with_plus":
            # Random pattern: 2-3 horizontal, 2-3 vertical, but ensure at least one intersection
            num_horizontal = random.randint(2, 3)
            horizontal_indices = random.sample(range(5), num_horizontal)
            vertical_indices = [i for i in range(5) if i not in horizontal_indices]
            min_intersections = 1
        else:
            # Completely random pattern, but ensure at least one intersection
            num_horizontal = random.randint(2, 3)
            horizontal_indices = random.sample(range(5), num_horizontal)
            vertical_indices = [i for i in range(5) if i not in horizontal_indices]
            min_intersections = 1
        
        # Track positions of cells to ensure limited cells per row/column
        # Format: {row: [col1, col2, ...], ...}
        cells_in_row = {}
        # Format: {col: [row1, row2, ...], ...}
        cells_in_col = {}
        
        # Create pairs
        pairs = []  # Store all pairs for validation
        
        def get_possible_intersections():
            """Calculate potential intersections from existing horizontal and vertical pairs"""
            potential_intersections = []
            for h_pair in [p for p in pairs if p[0]]:  # horizontal pairs
                _, h_row, h_col1, h_col2, h_color = h_pair
                for v_pair in [p for p in pairs if not p[0]]:  # vertical pairs
                    _, v_col, v_row1, v_row2, v_color = v_pair
                    # Check if lines would intersect
                    if h_col1 <= v_col <= h_col2 and v_row1 <= h_row <= v_row2:
                        potential_intersections.append((h_row, v_col, h_color, v_color))
            return potential_intersections
        
        def create_pair(is_horizontal, color_idx, force_intersect=False):
            """Create a pair of cells in the grid, with option to force intersection"""
            color = pair_colors[color_idx]
            
            if is_horizontal:
                # Find available rows that have fewer than 2 cells
                available_rows = [r for r in range(2, height-2) 
                                if r not in cells_in_row or len(cells_in_row[r]) < 2]
                
                if not available_rows:
                    return False
                
                # If we need to force an intersection, prioritize rows that would create one
                if force_intersect and len([p for p in pairs if not p[0]]) > 0:
                    # We have vertical pairs, try to intersect with one
                    priority_rows = []
                    for v_pair in [p for p in pairs if not p[0]]:
                        _, v_col, v_row1, v_row2, _ = v_pair
                        # Consider rows between v_row1 and v_row2 that are available
                        for r in range(v_row1, v_row2 + 1):
                            if r in available_rows:
                                priority_rows.append((r, v_col))
                    
                    if priority_rows:
                        # Choose a random priority row and its corresponding column
                        row, intersect_col = random.choice(priority_rows)
                    else:
                        # No intersection possible, choose random row
                        row = random.choice(available_rows)
                        intersect_col = None
                else:
                    # No need to force intersection, choose random row
                    row = random.choice(available_rows)
                    intersect_col = None
                
                # Find columns that don't already have 2 cells
                available_cols = [c for c in range(2, width-2) 
                                if c not in cells_in_col or len(cells_in_col[c]) < 2]
                
                if len(available_cols) < 2:
                    return False
                
                # If we have an intersection column, ensure it's included
                if intersect_col is not None and intersect_col in available_cols:
                    # Choose another column ensuring minimum gap
                    min_gap = 4  # Minimum 3 spaces between cells
                    other_avail_cols = [c for c in available_cols if c != intersect_col]
                    
                    # Find columns that have sufficient gap from intersect_col
                    possible_cols = [c for c in other_avail_cols 
                                    if abs(c - intersect_col) >= min_gap]
                    
                    if not possible_cols:
                        return False
                    
                    other_col = random.choice(possible_cols)
                    col1, col2 = sorted([intersect_col, other_col])
                else:
                    # Choose two columns ensuring minimum gap
                    available_cols.sort()
                    possible_col_pairs = []
                    for i, c1 in enumerate(available_cols):
                        for c2 in available_cols[i+1:]:
                            if c2 - c1 >= 4:  # Minimum 3 spaces between cells
                                possible_col_pairs.append((c1, c2))
                    
                    if not possible_col_pairs:
                        return False
                    
                    col1, col2 = random.choice(possible_col_pairs)
                
                # Ensure surroundings are empty
                surroundings_empty = True
                for dr in [-1, 0, 1]:
                    for pos in [col1, col2]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip the cell itself
                            nr, nc = row + dr, pos + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                if grid[nr, nc] != 0:
                                    surroundings_empty = False
                                    break
                
                if not surroundings_empty:
                    return False
                
                # Update cell tracking
                if row not in cells_in_row:
                    cells_in_row[row] = []
                cells_in_row[row].extend([col1, col2])
                
                for col in [col1, col2]:
                    if col not in cells_in_col:
                        cells_in_col[col] = []
                    cells_in_col[col].append(row)
                
                # Place cells
                grid[row, col1] = color
                grid[row, col2] = color
                pairs.append((True, row, col1, col2, color))
                return True
            else:  # vertical
                # Find available columns that have fewer than 2 cells
                available_cols = [c for c in range(2, width-2) 
                                if c not in cells_in_col or len(cells_in_col[c]) < 2]
                
                if not available_cols:
                    return False
                
                # If we need to force an intersection, prioritize columns that would create one
                if force_intersect and len([p for p in pairs if p[0]]) > 0:
                    # We have horizontal pairs, try to intersect with one
                    priority_cols = []
                    for h_pair in [p for p in pairs if p[0]]:
                        _, h_row, h_col1, h_col2, _ = h_pair
                        # Consider columns between h_col1 and h_col2 that are available
                        for c in range(h_col1, h_col2 + 1):
                            if c in available_cols:
                                priority_cols.append((c, h_row))
                    
                    if priority_cols:
                        # Choose a random priority column and its corresponding row
                        col, intersect_row = random.choice(priority_cols)
                    else:
                        # No intersection possible, choose random column
                        col = random.choice(available_cols)
                        intersect_row = None
                else:
                    # No need to force intersection, choose random column
                    col = random.choice(available_cols)
                    intersect_row = None
                
                # Find rows that don't already have 2 cells
                available_rows = [r for r in range(2, height-2) 
                                if r not in cells_in_row or len(cells_in_row[r]) < 2]
                
                if len(available_rows) < 2:
                    return False
                
                # If we have an intersection row, ensure it's included
                if intersect_row is not None and intersect_row in available_rows:
                    # Choose another row ensuring minimum gap
                    min_gap = 4  # Minimum 3 spaces between cells
                    other_avail_rows = [r for r in available_rows if r != intersect_row]
                    
                    # Find rows that have sufficient gap from intersect_row
                    possible_rows = [r for r in other_avail_rows 
                                    if abs(r - intersect_row) >= min_gap]
                    
                    if not possible_rows:
                        return False
                    
                    other_row = random.choice(possible_rows)
                    row1, row2 = sorted([intersect_row, other_row])
                else:
                    # Choose two rows ensuring minimum gap
                    available_rows.sort()
                    possible_row_pairs = []
                    for i, r1 in enumerate(available_rows):
                        for r2 in available_rows[i+1:]:
                            if r2 - r1 >= 4:  # Minimum 3 spaces between cells
                                possible_row_pairs.append((r1, r2))
                    
                    if not possible_row_pairs:
                        return False
                    
                    row1, row2 = random.choice(possible_row_pairs)
                
                # Ensure surroundings are empty
                surroundings_empty = True
                for dc in [-1, 0, 1]:
                    for pos in [row1, row2]:
                        for dr in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip the cell itself
                            nr, nc = pos + dr, col + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                if grid[nr, nc] != 0:
                                    surroundings_empty = False
                                    break
                
                if not surroundings_empty:
                    return False
                
                # Update cell tracking
                if col not in cells_in_col:
                    cells_in_col[col] = []
                cells_in_col[col].extend([row1, row2])
                
                for row in [row1, row2]:
                    if row not in cells_in_row:
                        cells_in_row[row] = []
                    cells_in_row[row].append(col)
                
                # Place cells
                grid[row1, col] = color
                grid[row2, col] = color
                pairs.append((False, col, row1, row2, color))
                return True
        
        # First attempt to create all pairs
        success = True
        
        # Place horizontal pairs first
        for idx in horizontal_indices:
            # For the last horizontal pair, try to force an intersection if we don't have enough
            force_intersect = (idx == horizontal_indices[-1] and 
                              len(get_possible_intersections()) < min_intersections)
            
            if not create_pair(True, idx, force_intersect):
                success = False
                break
        
        # Then place vertical pairs
        if success:
            for idx in vertical_indices:
                # For the last vertical pair, try to force an intersection if we don't have enough
                force_intersect = (idx == vertical_indices[-1] and 
                                  len(get_possible_intersections()) < min_intersections)
                
                if not create_pair(False, idx, force_intersect):
                    success = False
                    break
        
        # If failed to create all pairs, try again with different settings
        if not success or len(pairs) < 5:
            return self.create_input(taskvars, {
                "height": height,
                "width": width,
                "pattern": pattern
            })
        
        # Verify no row or column has more than 2 cells
        row_counts = {}
        col_counts = {}
        for r in range(height):
            for c in range(width):
                if grid[r, c] != 0:
                    row_counts[r] = row_counts.get(r, 0) + 1
                    col_counts[c] = col_counts.get(c, 0) + 1
        
        if max(row_counts.values(), default=0) > 2 or max(col_counts.values(), default=0) > 2:
            # Too many cells in a row or column, try again
            return self.create_input(taskvars, gridvars)
        
        # Calculate actual intersections
        intersections = get_possible_intersections()
        
        # Verify we have the required number of intersections
        if len(intersections) < min_intersections:
            return self.create_input(taskvars, gridvars)
        
        # For specific patterns, verify additional criteria
        if pattern == "one_plus_cut_horizontals":
            # In addition to one intersection, verify we have at least 2 cuts 
            # (vertical lines that cross horizontal lines without intersection)
            cuts = 0
            for h_pair in [p for p in pairs if p[0]]:  # horizontal pairs
                _, h_row, h_col1, h_col2, _ = h_pair
                for v_pair in [p for p in pairs if not p[0]]:  # vertical pairs
                    _, v_col, v_row1, v_row2, _ = v_pair
                    if h_col1 <= v_col <= h_col2 and (h_row < v_row1 or h_row > v_row2):
                        cuts += 1
            
            if cuts < 2:
                return self.create_input(taskvars, gridvars)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid
        output_grid = grid.copy()
        height, width = grid.shape
        
        # Find all colored cells and group by color
        color_to_positions = {}
        for r in range(height):
            for c in range(width):
                color = grid[r, c]
                if color != 0:  # Skip empty cells
                    if color not in color_to_positions:
                        color_to_positions[color] = []
                    color_to_positions[color].append((r, c))
        
        # Connect each pair
        for color, positions in color_to_positions.items():
            if len(positions) == 2:
                (r1, c1), (r2, c2) = positions
                
                # Determine if horizontal or vertical pair
                if r1 == r2:  # Horizontal line
                    for c in range(min(c1, c2), max(c1, c2) + 1):
                        output_grid[r1, c] = color
                else:  # Vertical line
                    for r in range(min(r1, r2), max(r1, r2) + 1):
                        output_grid[r, c1] = color
        
        # Enforce vertical line priority at intersections
        for r in range(height):
            for c in range(width):
                if grid[r, c] == 0:  # Look at cells that were originally empty
                    # Find if this is an intersection of a vertical and horizontal line
                    is_part_of_vertical = False
                    vertical_color = None
                    
                    # Check if this cell is part of any vertical line
                    for color, positions in color_to_positions.items():
                        if len(positions) == 2:
                            (r1, c1), (r2, c2) = positions
                            if c1 == c2 == c and min(r1, r2) <= r <= max(r1, r2):
                                is_part_of_vertical = True
                                vertical_color = color
                                break
                    
                    if is_part_of_vertical and vertical_color is not None:
                        # If it's part of a vertical line, ensure it has the vertical line's color
                        output_grid[r, c] = vertical_color
        
        return output_grid

