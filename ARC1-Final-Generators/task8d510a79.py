from arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task8d510a79Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains a completely filled row with {color('middle_row')} color , which can be any row within two rows above or below the middle row.",
            "Several {color('cell_color1')} and {color('cell_color2')} cells are placed above and below the {color('middle_row')} line.",
            "There can be at most one colored cell in each column above and at most one colored cell in each column below the {color('middle_row')} line."
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the {color('middle_row')} as well as the {color('cell_color1')} and {color('cell_color2')} cells.",
            "The goal is to extend the colored cells based on their type.",
            "The {color('cell_color1')} cells are extended toward the {color('middle_row')}. If a {color('cell_color1')} lies above the line, it extends downward; if it lies below, it extends upward.",
            "The {color('cell_color2')} cells are extended toward the nearest boundary row. If a {color('cell_color2')} lies above the {color('middle_row')}, it extends upward toward the first row; if it lies below, it extends downward toward the last row.",
            "The {color('middle_row')} remains unchanged."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # ---------------------------
    # Helpers (bounded placement)
    # ---------------------------
    @staticmethod
    def _choose_middle_row_pos(grid_size: int, center: int) -> int:
        # Keep a 1-row buffer from the outer border and a 1-row buffer around the middle row
        lo = max(2, center - 2)
        hi = min(grid_size - 3, center + 2)
        return random.randint(lo, hi)

    @staticmethod
    def _plan_counts(cap: int, must_be_at_least: int, default_low: int, default_high: int) -> int:
        """Pick a count between [default_low..default_high], clipped by 'cap', but >= must_be_at_least if possible."""
        if cap <= 0:
            return 0
        low = min(default_low, cap)
        high = min(default_high, cap)
        if low > high:
            low = high
        n = random.randint(low, high)
        if n < must_be_at_least and cap >= must_be_at_least:
            n = must_be_at_least
        return n

    @staticmethod
    def _assign_colors(num_cells: int, color1: int, color2: int, min_each: int = 2) -> List[int]:
        """Return a shuffled list of length num_cells with at least 'min_each' of each color when possible."""
        if num_cells == 0:
            return []
        # If not enough capacity for 2+2, relax gracefully
        need_each = min(min_each, max(0, num_cells // 2))
        base = [color1] * need_each + [color2] * need_each
        remaining = max(0, num_cells - len(base))
        base += [random.choice([color1, color2]) for _ in range(remaining)]
        random.shuffle(base)
        return base

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        middle_row_color = taskvars['middle_row']
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']

        grid = np.zeros((grid_size, grid_size), dtype=int)

        center = grid_size // 2
        middle_row_pos = gridvars.get('middle_row_pos', self._choose_middle_row_pos(grid_size, center))

        # Fill the middle row fully
        grid[middle_row_pos, :] = middle_row_color

        # Valid rows strictly above/below, leaving one-row gap next to the middle row (as in your original)
        above_rows = list(range(1, middle_row_pos - 1)) if middle_row_pos > 2 else []
        below_rows = list(range(middle_row_pos + 2, grid_size - 1)) if middle_row_pos < grid_size - 3 else []

        all_cols = list(range(grid_size))

        # Decide how many cells to place on each side (bounded; no unbounded while-loops)
        # We try to ensure at least 1 on each side when possible
        max_above_cols = grid_size  # at most one per column per side
        max_below_cols = grid_size

        n_above = self._plan_counts(
            cap=min(len(above_rows), max_above_cols),
            must_be_at_least=1 if len(above_rows) > 0 else 0,
            default_low=1,
            default_high=min(4, grid_size // 2)
        )
        n_below = self._plan_counts(
            cap=min(len(below_rows), max_below_cols),
            must_be_at_least=1 if len(below_rows) > 0 else 0,
            default_low=1,
            default_high=min(4, grid_size // 2)
        )

        # Guarantee at least 4 total cells if capacity allows (to help with min 2 per color)
        total_cap = (len(above_rows) > 0) * grid_size + (len(below_rows) > 0) * grid_size
        target_total = 4 if total_cap >= 4 else min(total_cap, 3)
        if n_above + n_below < target_total:
            # Add the remainder to the side(s) with capacity
            need = target_total - (n_above + n_below)
            for _ in range(need):
                if len(above_rows) > 0 and n_above < grid_size:
                    n_above += 1
                elif len(below_rows) > 0 and n_below < grid_size:
                    n_below += 1

        # Pick distinct columns for each side to ensure ≤1 per column per side
        random.shuffle(all_cols)
        above_cols = all_cols[:n_above] if n_above > 0 else []
        remaining_cols = [c for c in all_cols if c not in above_cols]
        random.shuffle(remaining_cols)
        below_cols = remaining_cols[:n_below] if n_below > 0 else []

        # Assign colors with at least two of each when possible
        colors_above = self._assign_colors(len(above_cols), cell_color1, cell_color2, min_each=2)
        colors_below = self._assign_colors(len(below_cols), cell_color1, cell_color2, min_each=2)

        # If after combining we still don't have ≥2 of each and capacity is small, relax gracefully
        def enforce_minimums(col_list_a, col_list_b, cols_a, cols_b):
            total = len(cols_a) + len(cols_b)
            if total == 0:
                return col_list_a, col_list_b
            want = 2
            # Count current
            c1 = col_list_a.count(cell_color1) + col_list_b.count(cell_color1)
            c2 = total - c1
            # Try to flip some colors to meet minimums (bounded small loops)
            if c1 < want:
                need = min(want - c1, total)
                for palette in (col_list_a, col_list_b):
                    for i in range(len(palette)):
                        if need == 0:
                            break
                        if palette[i] == cell_color2:
                            palette[i] = cell_color1
                            need -= 1
                    if need == 0:
                        break
            elif c2 < want:
                need = min(want - c2, total)
                for palette in (col_list_a, col_list_b):
                    for i in range(len(palette)):
                        if need == 0:
                            break
                        if palette[i] == cell_color1:
                            palette[i] = cell_color2
                            need -= 1
                    if need == 0:
                        break
            return col_list_a, col_list_b

        colors_above, colors_below = enforce_minimums(colors_above, colors_below, above_cols, below_cols)

        # Place cells: pick a random eligible row per (side, column)
        for col, col_color in zip(above_cols, colors_above):
            if not above_rows:
                break
            row = random.choice(above_rows)
            grid[row, col] = col_color

        for col, col_color in zip(below_cols, colors_below):
            if not below_rows:
                break
            row = random.choice(below_rows)
            grid[row, col] = col_color

        # Optional: sprinkle a few extra cells (still bounded and respecting per-side uniqueness)
        # We'll add at most one extra per side in a new column, with small probability.
        def sprinkle(side_rows: List[int], used_cols: set, color_choices: List[int]):
            if not side_rows:
                return
            if random.random() < 0.3:  # small chance to add one more
                free_cols = [c for c in all_cols if c not in used_cols]
                if free_cols:
                    c = random.choice(free_cols)
                    r = random.choice(side_rows)
                    grid[r, c] = random.choice(color_choices)
                    used_cols.add(c)

        used_above = set(above_cols)
        used_below = set(below_cols)
        sprinkle(above_rows, used_above, [cell_color1, cell_color2])
        sprinkle(below_rows, used_below, [cell_color1, cell_color2])

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        middle_row_color = taskvars['middle_row']
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']

        output_grid = grid.copy()

        # Identify the filled middle row
        middle_row_pos = None
        for row in range(grid_size):
            if np.all(grid[row, :] == middle_row_color):
                middle_row_pos = row
                break

        if middle_row_pos is None:
            return output_grid  # Safety: unchanged

        # Process each column exactly once
        for col in range(grid_size):
            # Scan above -> nearest colored cell
            for row in range(0, middle_row_pos):
                val = grid[row, col]
                if val == cell_color1:
                    # extend downward up to just before the middle row
                    output_grid[row + 1:middle_row_pos, col] = cell_color1
                elif val == cell_color2:
                    # extend upward to top boundary
                    if row > 0:
                        output_grid[0:row, col] = cell_color2
                # Only extend from the first colored cell encountered on this side
                if val in (cell_color1, cell_color2):
                    break

            # Scan below -> nearest colored cell
            for row in range(grid_size - 1, middle_row_pos, -1):
                val = grid[row, col]
                if val == cell_color1:
                    # extend upward just after the middle row
                    if middle_row_pos + 1 < row:
                        output_grid[middle_row_pos + 1:row, col] = cell_color1
                elif val == cell_color2:
                    # extend downward to bottom boundary
                    if row + 1 < grid_size:
                        output_grid[row + 1:grid_size, col] = cell_color2
                if val in (cell_color1, cell_color2):
                    break

        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        grid_size = random.choice([7, 9, 11, 13, 15, 17, 19])  # odd sizes
        colors = list(range(1, 10))
        chosen_colors = random.sample(colors, 3)

        taskvars = {
            'grid_size': grid_size,
            'middle_row': chosen_colors[0],
            'cell_color1': chosen_colors[1],
            'cell_color2': chosen_colors[2]
        }

        # Training examples
        num_train = random.randint(3, 5)
        train_examples = []
        center = grid_size // 2
        for _ in range(num_train):
            middle_row_pos = self._choose_middle_row_pos(grid_size, center)
            gridvars = {'middle_row_pos': middle_row_pos}
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)
            train_examples.append({'input': inp, 'output': out})

        # Test example
        test_middle_row_pos = self._choose_middle_row_pos(grid_size, center)
        test_gridvars = {'middle_row_pos': test_middle_row_pos}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)

        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
        return taskvars, train_test_data
