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

        fill_type = (
            gridvars.get('force_fill_type')
            if gridvars and 'force_fill_type' in gridvars
            else random.choice(['row', 'col'])
        )

        max_dim = rows if fill_type == 'row' else cols
        min_spacing = 2  # Minimum spacing between filled rows/columns
        available_positions = max_dim - 2  # indices 1..max_dim-2 available
        max_possible_fills = (available_positions + min_spacing) // (min_spacing + 1) if available_positions > 0 else 0

        max_allowed_fills = min(5, max_possible_fills) if max_possible_fills >= 1 else 1
        if gridvars and 'force_num_to_fill' in gridvars and gridvars['force_num_to_fill'] is not None:
            num_to_fill = max(1, min(int(gridvars['force_num_to_fill']), max_allowed_fills))
        else:
            num_to_fill = random.randint(1, max_allowed_fills)

        available_indices = list(range(1, max_dim - 1))  # Exclude borders
        filled_indices = []
        for _ in range(num_to_fill):
            if not available_indices:
                break
            index = random.choice(available_indices)
            filled_indices.append(index)
            for offset in range(-min_spacing, min_spacing + 1):
                if index + offset in available_indices:
                    available_indices.remove(index + offset)

        filled_indices.sort()
        colors = random.sample(range(1, 10), len(filled_indices))

        for index, color in zip(filled_indices, colors):
            if fill_type == 'row':
                grid[index, :] = color
            else:
                grid[:, index] = color

        line_colors = set(colors)

        # Uniqueness bookkeeping for line colors:
        # vertical lines => at most one extra cell of that color per ROW
        # horizontal lines => at most one extra cell of that color per COLUMN
        used_rows_by_color = {c: set() for c in line_colors}
        used_cols_by_color = {c: set() for c in line_colors}

        color_counts = {i: 0 for i in range(1, 10)}

        # Ensure at least one extra cell of EACH line color exists
        for index, color in zip(filled_indices, colors):
            while True:
                if fill_type == 'row':
                    invalid_rows = {index, index - 1, index + 1}
                    valid_rows = [rr for rr in range(rows) if rr not in invalid_rows]
                    r = random.choice(valid_rows) if valid_rows else random.choice([i for i in range(rows) if i != index])
                    c = random.randint(0, cols - 1)

                    # horizontal lines -> at most one extra cell per COLUMN for this color
                    if c in used_cols_by_color[color]:
                        continue
                else:
                    invalid_cols = {index, index - 1, index + 1}
                    valid_cols = [cc for cc in range(cols) if cc not in invalid_cols]
                    c = random.choice(valid_cols) if valid_cols else random.choice([i for i in range(cols) if i != index])
                    r = random.randint(0, rows - 1)

                    # vertical lines -> at most one extra cell per ROW for this color
                    if r in used_rows_by_color[color]:
                        continue

                if grid[r, c] == 0:
                    grid[r, c] = color
                    color_counts[color] += 1

                    if fill_type == 'row':
                        used_cols_by_color[color].add(c)
                    else:
                        used_rows_by_color[color].add(r)
                    break

        # Add additional random cells
        random_fill_density = random.uniform(0.02, 0.1)
        num_random_cells = int(random_fill_density * rows * cols)

        additional_colors = [c for c in range(1, 10) if c not in colors]
        all_possible_colors = colors + additional_colors

        from collections import deque

        for _ in range(num_random_cells):
            attempts = 0
            while attempts < 100:
                r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
                color = random.choice(all_possible_colors)

                if grid[r, c] != 0:
                    attempts += 1
                    continue

                # Enforce uniqueness for EACH line color
                if color in line_colors:
                    if fill_type == 'col':
                        if r in used_rows_by_color[color]:
                            attempts += 1
                            continue
                    else:
                        if c in used_cols_by_color[color]:
                            attempts += 1
                            continue

                # Avoid adjacency/connection to the corresponding filled line
                if fill_type == 'row':
                    row_idx = next((idx for idx in filled_indices if color == grid[idx, 0]), None)
                    if row_idx is not None:
                        if r == row_idx or r == row_idx - 1 or r == row_idx + 1:
                            attempts += 1
                            continue
                        if (r > row_idx and any(grid[i, c] == color for i in range(row_idx + 1, r))) or \
                           (r < row_idx and any(grid[i, c] == color for i in range(r + 1, row_idx))):
                            attempts += 1
                            continue
                else:
                    col_idx = next((idx for idx in filled_indices if color == grid[0, idx]), None)
                    if col_idx is not None:
                        if c == col_idx or c == col_idx - 1 or c == col_idx + 1:
                            attempts += 1
                            continue
                        if (c > col_idx and any(grid[r, j] == color for j in range(col_idx + 1, c))) or \
                           (c < col_idx and any(grid[r, j] == color for j in range(c + 1, col_idx))):
                            attempts += 1
                            continue

                # Ensure candidate same-color component doesn't touch the grid edge
                grid[r, c] = color
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
                    grid[r, c] = 0
                    attempts += 1
                    continue

                color_counts[color] += 1

                if color in line_colors:
                    if fill_type == 'col':
                        used_rows_by_color[color].add(r)
                    else:
                        used_cols_by_color[color].add(c)

                break

        return grid

    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        output_grid = grid.copy()

        def is_filled_row(g, r):
            return np.all(g[r, :] != 0) and len(set(g[r, :])) == 1

        def is_filled_col(g, c):
            return np.all(g[:, c] != 0) and len(set(g[:, c])) == 1

        colored_rows = {r: grid[r, 0] for r in range(rows) if is_filled_row(grid, r)}
        colored_cols = {c: grid[0, c] for c in range(cols) if is_filled_col(grid, c)}

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    self._translate_item([(r, c, grid[r, c])], {grid[r, c]}, colored_rows, colored_cols, output_grid)

        new_colored_rows = {r: output_grid[r, 0] for r in range(rows) if is_filled_row(output_grid, r)}
        new_colored_cols = {c: output_grid[0, c] for c in range(cols) if is_filled_col(output_grid, c)}
        allowed_colors = set(new_colored_rows.values()) | set(new_colored_cols.values())

        for r in range(rows):
            for c in range(cols):
                if output_grid[r, c] != 0 and output_grid[r, c] not in allowed_colors:
                    output_grid[r, c] = 0

        return output_grid

    def _translate_item(self, cells, colors, colored_rows, colored_cols, output_grid):
        r, c, color = cells[0]

        if color in colored_rows.values():
            row_idx = next(r_idx for r_idx, col in colored_rows.items() if col == color)
            if r == row_idx:
                return
            output_grid[r, c] = 0
            new_r = row_idx - 1 if r < row_idx else row_idx + 1
            if 0 <= new_r < output_grid.shape[0]:
                output_grid[new_r, c] = color

        elif color in colored_cols.values():
            col_idx = next(c_idx for c_idx, col in colored_cols.items() if col == color)
            if c == col_idx:
                return
            output_grid[r, c] = 0
            new_c = col_idx - 1 if c < col_idx else col_idx + 1
            if 0 <= new_c < output_grid.shape[1]:
                output_grid[r, new_c] = color

    def create_grids(self):
        taskvars = {
            'rows': random.randint(15, 30),
            'cols': random.randint(15, 30)
        }

        num_train = random.randint(3, 4)
        num_test = 1
        min_spacing = 2

        train = []

        # Ensure one row-filled and one col-filled example, but randomize the order
        first_type = random.choice(['row', 'col'])
        second_type = 'col' if first_type == 'row' else 'row'

        for fill_t in [first_type, second_type]:
            inp = self.create_input(taskvars, {'force_fill_type': fill_t})
            train.append({'input': inp, 'output': self.transform_input(inp, taskvars)})

        for _ in range(max(0, num_train - len(train))):
            inp = self.create_input(taskvars, {})
            train.append({'input': inp, 'output': self.transform_input(inp, taskvars)})

        def is_filled_row(g, r):
            return np.all(g[r, :] != 0) and len(set(g[r, :])) == 1

        def is_filled_col(g, c):
            return np.all(g[:, c] != 0) and len(set(g[:, c])) == 1

        train_counts = []
        for ex in train:
            g = ex['input']
            rows_filled = sum(1 for r in range(g.shape[0]) if is_filled_row(g, r))
            cols_filled = sum(1 for c in range(g.shape[1]) if is_filled_col(g, c))
            train_counts.append(max(rows_filled, cols_filled))

        train_counts_set = set(train_counts)
        max_train_count = max(train_counts) if train_counts else 0

        test = []
        for _ in range(num_test):
            chosen = None
            for trial_fill_type in ['row', 'col']:
                max_dim = taskvars['rows'] if trial_fill_type == 'row' else taskvars['cols']
                available_positions = max_dim - 2
                max_possible_fills = (available_positions + min_spacing) // (min_spacing + 1) if available_positions > 0 else 0
                desired_count = min(max_train_count + 1, 5)
                if max_possible_fills >= desired_count:
                    chosen = (trial_fill_type, desired_count)
                    break

            if chosen is None:
                for trial_fill_type in ['row', 'col']:
                    max_dim = taskvars['rows'] if trial_fill_type == 'row' else taskvars['cols']
                    available_positions = max_dim - 2
                    max_possible_fills = (available_positions + min_spacing) // (min_spacing + 1) if available_positions > 0 else 0
                    for candidate in range(1, min(max_possible_fills, 5) + 1):
                        if candidate not in train_counts_set:
                            chosen = (trial_fill_type, candidate)
                            break
                    if chosen:
                        break

            if chosen is None:
                inp = self.create_input(taskvars, {})
            else:
                fill_t, cnt = chosen
                inp = self.create_input(taskvars, {'force_fill_type': fill_t, 'force_num_to_fill': cnt})

            test.append({'input': inp, 'output': self.transform_input(inp, taskvars)})

        train_test_data = {'train': train, 'test': test}
        return taskvars, train_test_data
