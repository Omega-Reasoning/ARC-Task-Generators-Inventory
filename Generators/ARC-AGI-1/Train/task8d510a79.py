from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List


class Task8d510a79Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains a completely filled row of {color('middle_row')}, positioned within two rows of the grid center, with at least three rows both above and below this row.",
            "Several {color('cell_color1')} and {color('cell_color2')} cells are placed above and below the {color('middle_row')} line.",
            "There can be at most one colored cell in each column above and at most one colored cell in each column below the {color('middle_row')} line.",
            "Ensure that there is at least one {color('cell_color1')} cell and one {color('cell_color2')} cell both above and below the {color('middle_row')} line, and that no two {color('cell_color1')} or {color('cell_color2')} cells appear in consecutive columns."
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
    # Helpers
    # ---------------------------
    @staticmethod
    @staticmethod
    def _choose_middle_row_pos(grid_size: int, center: int) -> int:
        # Ensure at least 3 rows above and 3 rows below the middle row.
        # (0-indexed: rows 0..middle_row_pos-1 are above; rows middle_row_pos+1..grid_size-1 are below)
        lo = 3
        hi = grid_size - 4

        # If you still want it "near the center" within ±2, intersect with that window:
        lo = max(lo, center - 2)
        hi = min(hi, center + 2)

        return random.randint(lo, hi)


    @staticmethod
    def _plan_counts(cap: int, must_be_at_least: int, default_low: int, default_high: int) -> int:
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
    def _max_nonconsecutive_slots(n_cols: int) -> int:
        # maximum size of a set with no adjacent indices in [0..n_cols-1]
        return (n_cols + 1) // 2

    @staticmethod
    def _sample_nonconsecutive_columns(n_cols: int, k: int) -> List[int]:
        """
        Sample k columns from range(n_cols) such that no two are consecutive.
        Bounded constructive approach: shuffle candidates, greedily accept valid columns,
        and if not enough, retry a few times (still bounded).
        """
        if k <= 0:
            return []
        k = min(k, (n_cols + 1) // 2)

        cols = list(range(n_cols))
        for _ in range(20):  # bounded retries
            random.shuffle(cols)
            chosen = []
            chosen_set = set()
            for c in cols:
                if (c - 1) in chosen_set or (c + 1) in chosen_set:
                    continue
                chosen.append(c)
                chosen_set.add(c)
                if len(chosen) == k:
                    return chosen
        # Fallback: deterministic pattern then shuffle subset
        base = list(range(0, n_cols, 2))
        if len(base) < k:
            base = list(range(1, n_cols, 2))
        random.shuffle(base)
        return base[:k]

    @staticmethod
    def _assign_colors_side(num_cells: int, color1: int, color2: int, require_both: bool = True) -> List[int]:
        """
        Return list of length num_cells.
        If require_both and num_cells>=2 -> guarantee at least one of each color.
        """
        if num_cells <= 0:
            return []
        if require_both and num_cells >= 2:
            palette = [color1, color2] + [random.choice([color1, color2]) for _ in range(num_cells - 2)]
        else:
            palette = [random.choice([color1, color2]) for _ in range(num_cells)]
        random.shuffle(palette)
        return palette

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

        # Valid rows strictly above/below with a 1-row gap from the middle row
        above_rows = list(range(1, middle_row_pos - 1)) if middle_row_pos > 2 else []
        below_rows = list(range(middle_row_pos + 2, grid_size - 1)) if middle_row_pos < grid_size - 3 else []

        # If one side has no available rows (should be rare with your bounds), just return the base grid.
        if not above_rows or not below_rows:
            return grid

        # ---- Enforce: at least one c1 and one c2 above AND below -> need >=2 per side
        # Also enforce: no consecutive columns globally -> total columns <= ceil(grid_size/2)
        cap_total = self._max_nonconsecutive_slots(grid_size)

        # Pick base counts (bounded), but force at least 2 per side (so each side can have both colors)
        # We'll still clip if cap_total is tight.
        n_above = self._plan_counts(
            cap=grid_size,
            must_be_at_least=2,
            default_low=2,
            default_high=min(5, grid_size // 2 + 1)
        )
        n_below = self._plan_counts(
            cap=grid_size,
            must_be_at_least=2,
            default_low=2,
            default_high=min(5, grid_size // 2 + 1)
        )

        # If total exceeds nonconsecutive capacity, reduce while keeping >=2 per side if possible.
        total = n_above + n_below
        if total > cap_total:
            # reduce the larger side first, but never below 2 unless absolutely necessary
            while total > cap_total and (n_above > 2 or n_below > 2):
                if n_above >= n_below and n_above > 2:
                    n_above -= 1
                elif n_below > 2:
                    n_below -= 1
                total = n_above + n_below

            # still too many (extreme small grids) -> relax below 2 (but your grid sizes make this unlikely)
            while total > cap_total and (n_above > 1 or n_below > 1):
                if n_above >= n_below and n_above > 1:
                    n_above -= 1
                elif n_below > 1:
                    n_below -= 1
                total = n_above + n_below

        # Now sample TOTAL columns with no adjacency, then split into above/below
        chosen_cols = self._sample_nonconsecutive_columns(grid_size, n_above + n_below)
        random.shuffle(chosen_cols)
        above_cols = chosen_cols[:n_above]
        below_cols = chosen_cols[n_above:n_above + n_below]

        # Assign colors ensuring both colors exist on each side (since n_* >= 2 normally)
        colors_above = self._assign_colors_side(len(above_cols), cell_color1, cell_color2, require_both=True)
        colors_below = self._assign_colors_side(len(below_cols), cell_color1, cell_color2, require_both=True)

        # Place cells: one per chosen column per side, random eligible row
        for col, col_color in zip(above_cols, colors_above):
            r = random.choice(above_rows)
            grid[r, col] = col_color

        for col, col_color in zip(below_cols, colors_below):
            r = random.choice(below_rows)
            grid[r, col] = col_color

        # Optional sprinkle (still respecting "no consecutive columns"):
        # At most one extra globally, and only if it does not touch an existing column.
        if random.random() < 0.25:
            used = set(above_cols) | set(below_cols)
            free = [c for c in range(grid_size) if c not in used and (c - 1) not in used and (c + 1) not in used]
            if free:
                c = random.choice(free)
                # choose side randomly
                if random.random() < 0.5:
                    r = random.choice(above_rows)
                else:
                    r = random.choice(below_rows)
                grid[r, c] = random.choice([cell_color1, cell_color2])

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
            return output_grid

        for col in range(grid_size):
            # Scan above (closest to middle is not required; your original uses first encountered from top)
            for row in range(0, middle_row_pos):
                val = grid[row, col]
                if val == cell_color1:
                    output_grid[row + 1:middle_row_pos, col] = cell_color1
                elif val == cell_color2:
                    if row > 0:
                        output_grid[0:row, col] = cell_color2
                if val in (cell_color1, cell_color2):
                    break

            # Scan below
            for row in range(grid_size - 1, middle_row_pos, -1):
                val = grid[row, col]
                if val == cell_color1:
                    if middle_row_pos + 1 < row:
                        output_grid[middle_row_pos + 1:row, col] = cell_color1
                elif val == cell_color2:
                    if row + 1 < grid_size:
                        output_grid[row + 1:grid_size, col] = cell_color2
                if val in (cell_color1, cell_color2):
                    break

        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        grid_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])  # odd sizes between 7 and 29
        colors = list(range(1, 10))
        chosen_colors = random.sample(colors, 3)

        taskvars = {
            'grid_size': grid_size,
            'middle_row': chosen_colors[0],
            'cell_color1': chosen_colors[1],
            'cell_color2': chosen_colors[2]
        }

        num_train = random.randint(3, 5)
        train_examples = []
        center = grid_size // 2

        for _ in range(num_train):
            middle_row_pos = self._choose_middle_row_pos(grid_size, center)
            inp = self.create_input(taskvars, {'middle_row_pos': middle_row_pos})
            out = self.transform_input(inp, taskvars)
            train_examples.append({'input': inp, 'output': out})

        test_middle_row_pos = self._choose_middle_row_pos(grid_size, center)
        test_input = self.create_input(taskvars, {'middle_row_pos': test_middle_row_pos})
        test_output = self.transform_input(test_input, taskvars)

        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
        return taskvars, train_test_data
