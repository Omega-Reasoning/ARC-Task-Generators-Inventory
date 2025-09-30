from collections import defaultdict
import numpy as np
import random

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from typing import Dict, List, Any, Tuple

from transformation_library import GridObject, get_objects_from_raster
from utilities import visualize_grid

class ARCTask06df4c85Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The grid is a raster with {color('sep_color')} rows and columns as separators.",
            "The first row and column is {color('sep_color')}, i.e. a separator row/column.",
            "Subgrids have a dimension of {vars['subgrid_rows']}x{vars['subgrid_cols']} i.e. every {vars['subgrid_rows']+1} rows there is a separator row and every {vars['subgrid_cols']+1} columns there is a separator column.",
            "Some of the subgrids are colored (i.e. all cells in them have the same color).",
            "The colors which are used vary between input matrices."
        ]
        transformation_reasoning_chain = [
            "The output grid is obtained by copying the input grid and colouring additional subgrids.",
            "If two subgrids share the same raster row or column and have the same color, then all subgrids between them are coloured with the same color.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids_old(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Creates the task variables and the training/test pairs of grids.
        """
        # 1) Pick random subgrid scaling factors
        subgrid_rows = random.randint(1, 4)
        subgrid_cols = random.randint(1, 4)

        # 2) Pick a separator color (1..9)
        sep_color = random.randint(1, 9)
        
        taskvars = {
            "subgrid_rows": subgrid_rows,
            "subgrid_cols": subgrid_cols,
            "sep_color": sep_color,
        }
        
        nr_train = random.randint(3, 5)
        nr_test = random.randint(1, 2)

        return taskvars, self.create_grids_default(nr_train, nr_test, taskvars)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # 1) Pick random subgrid scaling factors
        subgrid_rows = random.randint(1, 4)
        subgrid_cols = random.randint(1, 4)

        # 2) Pick a separator color (1..9)
        sep_color = random.randint(1, 9)
        
        taskvars = {
            "subgrid_rows": subgrid_rows,
            "subgrid_cols": subgrid_cols,
            "sep_color": sep_color,
        }
        
        nr_train = random.randint(3, 5)
        nr_test = random.randint(1, 2)

        # there is a small risk that input and output are equal - we want to have at most one such case in the train data
        # and if there is a test case which is equal we need to have a second inequal test case to probe true understanding
        def generate_examples(n, is_test=False):
            examples = []
            attempts = 0
            max_attempts = 100 
            
            while len(examples) < n and attempts < max_attempts:
                input_grid = self.create_input(taskvars, {})
                output_grid = self.transform_input(input_grid, taskvars)
                
                # For test examples, always add the first one and ensure second is different
                if is_test and len(examples) == 0:
                    examples.append({'input': input_grid, 'output': output_grid})
                    continue
                    
                # Check if input equals output
                if not np.array_equal(input_grid, output_grid):
                    examples.append({'input': input_grid, 'output': output_grid})
                else:
                    # For training set, allow at most one equal case
                    equal_cases = sum(1 for ex in examples if np.array_equal(ex['input'], ex['output']))
                    if equal_cases == 0:
                        examples.append({'input': input_grid, 'output': output_grid})
                        
                attempts += 1
                
            if attempts >= max_attempts:
                raise RuntimeError("Could not generate enough diverse examples")
                
            return examples

        return taskvars, {
            'train': generate_examples(nr_train),
            'test': generate_examples(nr_test, is_test=True)
        }

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        1) Choose dimensions for the base grid, ensuring that when scaled up
           it won't exceed 30 in either dimension.
        2) Color some cells with the chosen object colors (others remain 0).
        3) Scale up the base grid and insert the jail bars (sep_color).
        4) Return the final (scaled) input grid.
        """
        subR = taskvars["subgrid_rows"]
        subC = taskvars["subgrid_cols"]
        sep_color = taskvars["sep_color"]
        
        # choose 1-4 object colors different from sep_color
        all_colors = list(range(1, 10))
        all_colors.remove(sep_color)
        random.shuffle(all_colors)
        obj_colors = all_colors[: random.randint(1, 4)]
        
        # --- 1) Determine base grid shape (H_small, W_small) ---
        # Each dimension n must satisfy: n * (sub + 1) - 1 <= 30
        # And we want at least 3 cells in each dimension
        minH = 3
        maxH = 31 // (subR + 1)  # integer division
        H_small = random.randint(minH, maxH)

        minW = 3
        maxW = 31 // (subC + 1)  # integer division
        W_small = random.randint(minW, maxW)

        # print("building new base grid")
        # Build the base grid
        base_grid = np.zeros((H_small, W_small), dtype=int)

        # --- 2) Color some cells in base_grid ---
        nr_cells = H_small * W_small
        # Number of potential pairs between 3 and sqrt(area)
        n_potential_pairs = random.randint(2, min(int(np.sqrt(nr_cells)), nr_cells // 2))

        # Track eligible cells (initially all cells are eligible)
        eligible = np.ones((H_small, W_small), dtype=bool)
        coords = [(r, c) for r in range(H_small) for c in range(W_small)]
        random.shuffle(coords)

        pairs_created = 0
        i = 0
        while pairs_created < n_potential_pairs and i < len(coords):
            r, c = coords[i]
            i += 1
            
            # print(f"Step {i}:")
            # print(visualize_grid(base_grid))
            # print(visualize_grid(eligible))

            if not eligible[r, c]:
                continue

            # Try colors until we find one that works
            valid_colors = obj_colors.copy()
            random.shuffle(valid_colors)
            color_is_valid = False

            for color in valid_colors:
                color_is_valid = True
                
                # Check row for same-colored cells and cells between
                for c2 in range(W_small):
                    if c2 != c and base_grid[r, c2] == color:
                        min_c, max_c = min(c, c2), max(c, c2)
                        # If ANY cell between (including endpoints) is ineligible, color is invalid
                        if any(not eligible[r, x] for x in range(min_c, max_c + 1)):
                            color_is_valid = False
                            # print(f"Color {color} is invalid for cell ({r}, {c}) - row check")
                            break
                
                if not color_is_valid:
                    continue
                    
                # Check column for same-colored cells and cells between
                for r2 in range(H_small):
                    if r2 != r and base_grid[r2, c] == color:
                        min_r, max_r = min(r, r2), max(r, r2)
                        # If ANY cell between (including endpoints) is ineligible, color is invalid
                        if any(not eligible[x, c] for x in range(min_r, max_r + 1)):
                            color_is_valid = False
                            # print(f"Color {color} is invalid for cell ({r}, {c}) - column check")
                            break
                
                if not color_is_valid:
                    continue
                    
                # print(f"found valid color: {color}")
                break  # Found a valid color

            if not color_is_valid:
                continue

            # Color the cell and mark affected cells as ineligible
            base_grid[r, c] = color
            eligible[r, c] = False

            # Mark cells between this and same-colored cells as ineligible
            # Check row
            for c2 in range(W_small):
                if c2 != c and base_grid[r, c2] == color:
                    min_c, max_c = min(c, c2), max(c, c2)
                    for x in range(min_c, max_c + 1):
                        eligible[r, x] = False

            # Check column
            for r2 in range(H_small):
                if r2 != r and base_grid[r2, c] == color:
                    min_r, max_r = min(r, r2), max(r, r2)
                    for x in range(min_r, max_r + 1):
                        eligible[x, c] = False
            
            # 80% chance to create a pair
            if (random.random() < 0.8) or (pairs_created == 0):
                # 50-50 chance for horizontal vs vertical pair
                is_horizontal = random.random() < 0.5
                
                if is_horizontal:
                    # Find eligible cells in the same row that don't create conflicts
                    possible_pairs = []
                    for c2 in range(c + 1, W_small):
                        if not eligible[r, c2]:
                            continue
                        
                        # Check if all cells between are eligible
                        if not all(eligible[r, x] for x in range(c + 1, c2)):
                            continue
                            
                        # Check if placing color at (r, c2) would create conflicts
                        pair_valid = True
                        
                        # Check row conflicts for second cell
                        for c3 in range(W_small):
                            if c3 != c2 and base_grid[r, c3] == color and c3 != c:
                                min_c, max_c = min(c2, c3), max(c2, c3)
                                if not all(eligible[r, x] for x in range(min_c, max_c + 1)):
                                    pair_valid = False
                                    break
                        
                        if not pair_valid:
                            continue
                            
                        # Check column conflicts for second cell
                        for r2 in range(H_small):
                            if r2 != r and base_grid[r2, c2] == color:
                                min_r, max_r = min(r, r2), max(r, r2)
                                if not all(eligible[x, c2] for x in range(min_r, max_r + 1)):
                                    pair_valid = False
                                    break
                        
                        if pair_valid:
                            possible_pairs.append((r, c2))
                else:
                    # Find eligible cells in the same column that don't create conflicts
                    possible_pairs = []
                    for r2 in range(r + 1, H_small):
                        if not eligible[r2, c]:
                            continue
                        
                        # Check if all cells between are eligible
                        if not all(eligible[x, c] for x in range(r + 1, r2)):
                            continue
                            
                        # Check if placing color at (r2, c) would create conflicts
                        pair_valid = True
                        
                        # Check column conflicts for second cell
                        for r3 in range(H_small):
                            if r3 != r2 and base_grid[r3, c] == color and r3 != r:
                                min_r, max_r = min(r2, r3), max(r2, r3)
                                if not all(eligible[x, c] for x in range(min_r, max_r + 1)):
                                    pair_valid = False
                                    break
                        
                        if not pair_valid:
                            continue
                            
                        # Check row conflicts for second cell
                        for c2 in range(W_small):
                            if c2 != c and base_grid[r2, c2] == color:
                                min_c, max_c = min(c, c2), max(c, c2)
                                if not all(eligible[r2, x] for x in range(min_c, max_c + 1)):
                                    pair_valid = False
                                    break
                        
                        if pair_valid:
                            possible_pairs.append((r2, c))
                
                if possible_pairs:
                    # Choose a random eligible pair
                    pair_r, pair_c = random.choice(possible_pairs)
                    base_grid[pair_r, pair_c] = color
                    
                    # Mark cells between pair (including endpoints) as ineligible
                    if is_horizontal:
                        min_c, max_c = min(c, pair_c), max(c, pair_c)
                        for x in range(min_c, max_c + 1):
                            eligible[r, x] = False
                            
                        # Mark cells for potential vertical connections to second cell
                        for r2 in range(H_small):
                            if r2 != r and base_grid[r2, pair_c] == color:
                                min_r, max_r = min(r, r2), max(r2, r)
                                for x in range(min_r, max_r + 1):
                                    eligible[x, pair_c] = False
                    else:
                        min_r, max_r = min(r, pair_r), max(r, pair_r)
                        for x in range(min_r, max_r + 1):
                            eligible[x, c] = False
                            
                        # Mark cells for potential horizontal connections to second cell
                        for c2 in range(W_small):
                            if c2 != c and base_grid[pair_r, c2] == color:
                                min_c, max_c = min(c, c2), max(c2, c)
                                for x in range(min_c, max_c + 1):
                                    eligible[pair_r, x] = False
                                    
                    pairs_created += 1

        # --- 3) Scale up + insert bars to form the final input grid ---
        final_H = H_small * subR + (H_small - 1)
        final_W = W_small * subC + (W_small - 1)
        grid = np.zeros((final_H, final_W), dtype=int)

        # Put horizontal bars (skip first and last)
        for rb in range(1, H_small):
            r_bar = rb * (subR + 1) - 1  # Corrected formula
            grid[r_bar, :] = sep_color

        # Put vertical bars (skip first and last)
        for cb in range(1, W_small):
            c_bar = cb * (subC + 1) - 1  # Corrected formula
            grid[:, c_bar] = sep_color

        # Fill subR x subC blocks
        for rb in range(H_small):
            for cb in range(W_small):
                color_val = base_grid[rb, cb]
                if color_val != 0:
                    # Corrected formulas for top and left positions
                    top = rb * (subR + 1)
                    left = cb * (subC + 1)
                    # fill the sub-block
                    grid[top : top+subR, left : left+subC] = color_val

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input grid by filling between same-colored objects in rows and columns."""
        subR = taskvars["subgrid_rows"]
        subC = taskvars["subgrid_cols"]
        
        # Get 2D array of objects from input grid
        objects = get_objects_from_raster(grid, subR, subC, True, subR, subC)
        output = grid.copy()
        
        # Create a map to track which objects should be colored with what color
        to_color = defaultdict(set)  # (row_idx, col_idx) -> set of colors
        
        # Mark vertically
        for col_idx in range(len(objects[0])):
            for color in range(1, 10):
                # Find positions where this color appears
                color_rows = [row_idx for row_idx, row in enumerate(objects) 
                            if color in row[col_idx].colors]
                
                if len(color_rows) >= 2:
                    top_idx = min(color_rows)
                    bottom_idx = max(color_rows)
                    
                    # Mark all objects between top and bottom
                    for row_idx in range(top_idx + 1, bottom_idx):
                        to_color[(row_idx, col_idx)].add(color)
        
        # Mark horizontally
        for row_idx, row in enumerate(objects):
            for color in range(1, 10):
                # Find positions where this color appears
                color_cols = [col_idx for col_idx, obj in enumerate(row) 
                            if color in obj.colors]
                
                if len(color_cols) >= 2:
                    left_idx = min(color_cols)
                    right_idx = max(color_cols)
                    
                    # Mark all objects between left and right
                    for col_idx in range(left_idx + 1, right_idx):
                        to_color[(row_idx, col_idx)].add(color)
        
        # Apply all color changes at once
        for (row_idx, col_idx), colors in to_color.items():
            for color in colors:
                objects[row_idx][col_idx].color_all(color)
                # Update output grid with new cells
                for r, c, col in objects[row_idx][col_idx].cells:
                    output[r, c] = col
        
        return output
