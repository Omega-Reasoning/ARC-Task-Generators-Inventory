from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import find_connected_objects, GridObject
import numpy as np
import random

class Task1a07d186Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grids have size {vars['rows']} × {vars['cols']}.",
            "Each input grid contains at most 5 fully colored rows or at most 5 fully colored columns (colors 1–9).",
            "All cells within a filled row or a filled column share the same color.",
            "The colors of the filled rows or columns are all different from each other.",
            "A few additional cells are randomly colored (1–9), while the remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All cells that share a color with any filled row or column are translated so that they directly touch their corresponding row or column.",
            "Cells whose colors do not match any filled row or column are removed in the final output."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)


    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)

        # Allow forcing the fill type via gridvars (used by create_grids to ensure
        # at least one training example has rows and one has columns).
        fill_type = gridvars.get('force_fill_type') if gridvars and 'force_fill_type' in gridvars else random.choice(['row', 'col'])

        # Determine how many filled rows/cols we can place given spacing constraints.
        max_dim = rows if fill_type == 'row' else cols
        min_spacing = 2  # Minimum spacing between filled rows/columns
        available_positions = max_dim - 2  # indices 1..max_dim-2 available
        max_possible_fills = (available_positions + min_spacing) // (min_spacing + 1) if available_positions > 0 else 0

        # Allow forcing the number to fill via gridvars; otherwise pick 1..5 (clamped)
        # The task requirement: number of completely filled rows/cols may be between 1 and 5.
        max_allowed_fills = min(5, max_possible_fills) if max_possible_fills >= 1 else 1
        if gridvars and 'force_num_to_fill' in gridvars and gridvars['force_num_to_fill'] is not None:
            # Clamp forced value to the allowed range [1, max_allowed_fills]
            num_to_fill = max(1, min(int(gridvars['force_num_to_fill']), max_allowed_fills))
        else:
            num_to_fill = random.randint(1, max_allowed_fills)
        # Generate spaced indices, excluding borders
        available_indices = list(range(1, max_dim - 1))  # Exclude 0 and max_dim-1
        filled_indices = []
        for _ in range(num_to_fill):
            if not available_indices:
                break
            index = random.choice(available_indices)
            filled_indices.append(index)
            # Remove nearby indices to maintain spacing
            for offset in range(-min_spacing, min_spacing + 1):
                if index + offset in available_indices:
                    available_indices.remove(index + offset)

        # Sort indices to help with placement constraints
        filled_indices.sort()
        colors = random.sample(range(1, 10), len(filled_indices))

        # Fill the rows/columns
        for index, color in zip(filled_indices, colors):
            if fill_type == 'row':
                grid[index, :] = color
            else:
                grid[:, index] = color

        # Initialize color_counts with all possible colors (1-9)
        color_counts = {i: 0 for i in range(1, 10)}

        # Ensure at least one cell of each color is present in a different row/column
        for i, (index, color) in enumerate(zip(filled_indices, colors)):
            while True:
                if fill_type == 'row':
                    # avoid placing a cell in the filled row or immediately adjacent
                    # to the filled row (to prevent connection)
                    invalid_rows = {index}
                    invalid_rows.update({index - 1, index + 1})
                    valid_rows = [rr for rr in range(rows) if rr not in invalid_rows]
                    if not valid_rows:
                        r = random.choice([i for i in range(rows) if i != index])
                    else:
                        r = random.choice(valid_rows)
                    c = random.randint(0, cols - 1)
                else:
                    # avoid placing a cell in the filled column or immediately adjacent
                    # to the filled column (to prevent connection)
                    invalid_cols = {index}
                    invalid_cols.update({index - 1, index + 1})
                    valid_cols = [cc for cc in range(cols) if cc not in invalid_cols]
                    if not valid_cols:
                        c = random.choice([i for i in range(cols) if i != index])
                    else:
                        c = random.choice(valid_cols)
                    r = random.randint(0, rows - 1)

                if grid[r, c] == 0:  # Only place if cell is empty
                    grid[r, c] = color
                    color_counts[color] += 1
                    break

        # Add additional random cells with colors, maintaining the constraints
        random_fill_density = random.uniform(0.02, 0.1)
        num_random_cells = int(random_fill_density * rows * cols)
        
        # Add some different colors not used in rows/columns
        additional_colors = [c for c in range(1, 10) if c not in colors]
        all_possible_colors = colors + additional_colors

        for _ in range(num_random_cells):
            attempts = 0
            while attempts < 100:  # Limit attempts to prevent infinite loops
                r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
                color = random.choice(all_possible_colors)
                
                # Check if position is valid
                if grid[r, c] != 0:
                    attempts += 1
                    continue
                
                # For row-colored grids: avoid adjacency/connection to any filled row
                if fill_type == 'row':
                    row_idx = next((idx for idx in filled_indices if color == grid[idx, 0]), None)
                    if row_idx is not None:
                        # If chosen cell would be in the filled row or immediately adjacent, skip
                        if r == row_idx or r == row_idx - 1 or r == row_idx + 1:
                            attempts += 1
                            continue
                        # Also avoid placing in same column if there's a chain of same-color cells
                        # directly connecting to the filled row (simple adjacency check)
                        if (r > row_idx and any(grid[i, c] == color for i in range(row_idx + 1, r))) or \
                           (r < row_idx and any(grid[i, c] == color for i in range(r + 1, row_idx))):
                            attempts += 1
                            continue
                
                # For column-colored grids: avoid adjacency/connection to any filled column
                else:
                    col_idx = next((idx for idx in filled_indices if color == grid[0, idx]), None)
                    if col_idx is not None:
                        # If chosen cell would be in the filled column or immediately adjacent, skip
                        if c == col_idx or c == col_idx - 1 or c == col_idx + 1:
                            attempts += 1
                            continue
                        # Also avoid placing in same row if there's a chain of same-color cells
                        # directly connecting to the filled column (simple adjacency check)
                        if (c > col_idx and any(grid[r, j] == color for j in range(col_idx + 1, c))) or \
                           (c < col_idx and any(grid[r, j] == color for j in range(c + 1, col_idx))):
                            attempts += 1
                            continue
                
                # Temporarily place the candidate and ensure it does not create
                # a same-color connected component that touches the grid edge.
                grid[r, c] = color

                # BFS/DFS to check connectivity to any grid edge for this color
                from collections import deque
                dq = deque([(r, c)])
                seen = {(r, c)}
                touches_edge = False
                while dq and not touches_edge:
                    i, j = dq.popleft()
                    if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                        touches_edge = True
                        break
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and (ni, nj) not in seen:
                            if grid[ni, nj] == color:
                                seen.add((ni, nj))
                                dq.append((ni, nj))

                if touches_edge:
                    # Reject this placement
                    grid[r, c] = 0
                    attempts += 1
                    continue

                # Accept placement
                color_counts[color] += 1
                break

        return grid

    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        output_grid = grid.copy()

        # Find colored rows and columns
        colored_rows = {r: grid[r, 0] for r in range(rows) if len(set(grid[r, :]) - {0}) == 1}
        colored_cols = {c: grid[0, c] for c in range(cols) if len(set(grid[:, c]) - {0}) == 1}

        # Process each cell individually
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    self._translate_item([(r, c, grid[r, c])], {grid[r, c]}, colored_rows, colored_cols, output_grid)

        # After translations, recompute completely filled rows/columns on the
        # output grid (translations may have created or completed rows/cols)
        # and remove any cells whose color is not one of those colors.
        new_colored_rows = {r: output_grid[r, 0] for r in range(rows) if len(set(output_grid[r, :]) - {0}) == 1}
        new_colored_cols = {c: output_grid[0, c] for c in range(cols) if len(set(output_grid[:, c]) - {0}) == 1}
        allowed_colors = set(new_colored_rows.values()) | set(new_colored_cols.values())

        for r in range(rows):
            for c in range(cols):
                if output_grid[r, c] != 0 and output_grid[r, c] not in allowed_colors:
                    output_grid[r, c] = 0

        return output_grid

    def _translate_item(self, cells, colors, colored_rows, colored_cols, output_grid):
        # We only need the first cell since we're dealing with individual cells
        r, c, color = cells[0]

        # Check if color matches any colored row
        if color in colored_rows.values():
            row_idx = next(r_idx for r_idx, col in colored_rows.items() if col == color)
            
            # Skip if cell is part of the colored row itself
            if r == row_idx:
                return
            
            # Clear original position
            output_grid[r, c] = 0
            
            # Move cell up or down to touch the row
            new_r = row_idx - 1 if r < row_idx else row_idx + 1
            if 0 <= new_r < output_grid.shape[0]:
                output_grid[new_r, c] = color

        # Check if color matches any colored column
        elif color in colored_cols.values():
            col_idx = next(c_idx for c_idx, col in colored_cols.items() if col == color)
            
            # Skip if cell is part of the colored column itself
            if c == col_idx:
                return
            
            # Clear original position
            output_grid[r, c] = 0
            
            # Move cell left or right to touch the column
            new_c = col_idx - 1 if c < col_idx else col_idx + 1
            if 0 <= new_c < output_grid.shape[1]:
                output_grid[r, new_c] = color

    def create_grids(self):
        taskvars = {
            'rows': random.randint(15,30),
            'cols': random.randint(15,30)
        }

        num_train = random.randint(3, 4)
        num_test = 1
        # spacing used by create_input when placing filled rows/cols
        min_spacing = 2

        # Build training examples ensuring at least one with filled rows and one
        # with filled columns so the model can train on both cases.
        train = []

        # Force one example with filled rows
        inp = self.create_input(taskvars, {'force_fill_type': 'row'})
        train.append({'input': inp, 'output': self.transform_input(inp, taskvars)})

        # Force one example with filled columns
        inp = self.create_input(taskvars, {'force_fill_type': 'col'})
        train.append({'input': inp, 'output': self.transform_input(inp, taskvars)})

        # Remaining train examples (if any)
        for _ in range(max(0, num_train - len(train))):
            inp = self.create_input(taskvars, {})
            train.append({'input': inp, 'output': self.transform_input(inp, taskvars)})

        # Build test examples such that the number of completely filled rows/cols
        # is different from training examples. Prefer a test count > max(train_counts).
        train_counts = []
        for ex in train:
            g = ex['input']
            rows_filled = sum(1 for r in range(g.shape[0]) if len(set(g[r, :]) - {0}) == 1)
            cols_filled = sum(1 for c in range(g.shape[1]) if len(set(g[:, c]) - {0}) == 1)
            train_counts.append(max(rows_filled, cols_filled))

        train_counts_set = set(train_counts)
        max_train_count = max(train_counts) if train_counts else 0

        test = []
        for _ in range(num_test):
            # Try to find a fill_type and count where count > max_train_count and is possible
            chosen = None
            for trial_fill_type in ['row', 'col']:
                max_dim = taskvars['rows'] if trial_fill_type == 'row' else taskvars['cols']
                available_positions = max_dim - 2
                max_possible_fills = (available_positions + min_spacing) // (min_spacing + 1) if available_positions > 0 else 0
                # Prefer a test count greater than any training count, but clamp to 5
                desired_count = min(max_train_count + 1, 5)
                if max_possible_fills >= desired_count:
                    chosen = (trial_fill_type, desired_count)
                    break

            if chosen is None:
                # Fall back to choose any count not in train_counts (within possible range)
                for trial_fill_type in ['row', 'col']:
                    max_dim = taskvars['rows'] if trial_fill_type == 'row' else taskvars['cols']
                    available_positions = max_dim - 2
                    max_possible_fills = (available_positions + min_spacing) // (min_spacing + 1) if available_positions > 0 else 0
                    # Search candidates in the allowed range but no greater than 5
                    for candidate in range(1, min(max_possible_fills, 5) + 1):
                        if candidate not in train_counts_set:
                            chosen = (trial_fill_type, candidate)
                            break
                    if chosen:
                        break

            if chosen is None:
                # As last resort, generate a random test example (should be rare)
                inp = self.create_input(taskvars, {})
            else:
                fill_t, cnt = chosen
                inp = self.create_input(taskvars, {'force_fill_type': fill_t, 'force_num_to_fill': cnt})

            test.append({'input': inp, 'output': self.transform_input(inp, taskvars)})

        train_test_data = {'train': train, 'test': test}

        return taskvars, train_test_data

