from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

from transformation_library import find_connected_objects


class Task0ca9ddb6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids have different sizes.",
            "Each input grid contains several colored cells of {color('cell_color1')}, {color('cell_color2')}, and some other colors, while the remaining cells are empty.",
            "The {color('cell_color1')} and {color('cell_color2')} cells appear such that each of them has a 3×3 subgrid of empty cells surrounding it.",
            "No two colored cells are connected to each other."
        ]

        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and then identifying all {color('cell_color1')} and {color('cell_color2')} cells.",
            "Around each {color('cell_color1')} cell, four {color('cell_color3')} cells are added at the top-left, top-right, bottom-left, and bottom-right diagonal positions.",
            "Around each {color('cell_color2')} cell, four {color('cell_color4')} cells are added at the top, bottom, left, and right positions.",
            "The other colored cells do not receive any additional surrounding pattern."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # -----------------------------
        # Task variables / palette setup
        # -----------------------------
        taskvars: Dict[str, Any] = {
            'cell_color1': random.randint(1, 9),
            'cell_color2': None,
            'cell_color3': None,
            'cell_color4': None,
        }

        # choose distinct non-zero colors
        while taskvars['cell_color2'] is None or taskvars['cell_color2'] == taskvars['cell_color1']:
            taskvars['cell_color2'] = random.randint(1, 9)

        # pick colors for added patterns distinct from c1/c2
        pool = [c for c in range(1, 10) if c not in (taskvars['cell_color1'], taskvars['cell_color2'])]
        taskvars['cell_color3'] = random.choice(pool)
        pool.remove(taskvars['cell_color3'])
        taskvars['cell_color4'] = random.choice(pool)
        pool.remove(taskvars['cell_color4'])

        # Task-level required "other" colors (NOT stored in taskvars):
        # These colors must appear at least once across the whole task (train+test),
        # but NOT necessarily all together in every single grid.
        num_required_other = random.randint(3, 4)
        required_other_colors = random.sample(pool, num_required_other)

        # Create 3-5 train examples and 1 test example
        num_train = random.randint(3, 5)
        train_test_data = self.create_grids_default(num_train, 1, taskvars)

        # ------------------------------------------------------------
        # Enforce per-example safety:
        #  - c3/c4 cannot appear in input
        #  - recompute outputs if any input changes
        # ------------------------------------------------------------
        def _sanitize_example(ex: Dict[str, Any]) -> None:
            grid = ex['input']
            c3, c4 = taskvars['cell_color3'], taskvars['cell_color4']
            bad = (grid == c3) | (grid == c4)
            if bad.any():
                grid[bad] = 0
            ex['output'] = self.transform_input(grid, taskvars)

        for ex in train_test_data['train']:
            _sanitize_example(ex)
        for ex in train_test_data['test']:
            _sanitize_example(ex)

        # ------------------------------------------------------------
        # Enforce GLOBAL "other color" coverage across train+test:
        # required_other_colors must each appear in at least one grid.
        # ------------------------------------------------------------
        def _get_other_colors_present(grid: np.ndarray) -> set:
            c1, c2 = taskvars['cell_color1'], taskvars['cell_color2']
            c3, c4 = taskvars['cell_color3'], taskvars['cell_color4']
            present = set(int(x) for x in np.unique(grid) if x != 0)
            present.discard(c1)
            present.discard(c2)
            present.discard(c3)
            present.discard(c4)
            return present

        def _build_forbidden_from_centers(grid: np.ndarray) -> np.ndarray:
            """Forbidden zones around c1/c2 centers so we don't violate their reserved neighborhoods."""
            h, w = grid.shape
            forbidden = np.zeros_like(grid, dtype=bool)
            c1, c2 = taskvars['cell_color1'], taskvars['cell_color2']
            centers = list(zip(*np.where((grid == c1) | (grid == c2))))

            for (r, c) in centers:
                # reserve 3x3 around center
                for rr in range(max(0, r - 1), min(h, r + 2)):
                    for cc in range(max(0, c - 1), min(w, c + 2)):
                        forbidden[rr, cc] = True
                # reserve cardinal distance 1 and 2
                for dr in (-1, -2, 1, 2):
                    rr = r + dr
                    if 0 <= rr < h:
                        forbidden[rr, c] = True
                for dc in (-1, -2, 1, 2):
                    cc = c + dc
                    if 0 <= cc < w:
                        forbidden[r, cc] = True

            return forbidden

        def _place_lonely_cell(grid: np.ndarray, forbidden: np.ndarray, col: int) -> bool:
            """Place `col` into an empty cell that is not forbidden and not 4-adjacent to any non-zero."""
            h, w = grid.shape
            candidates = []
            for r in range(2, h - 2):
                for c in range(2, w - 2):
                    if grid[r, c] != 0 or forbidden[r, c]:
                        continue
                    ok = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != 0:
                            ok = False
                            break
                    if ok:
                        candidates.append((r, c))
            if not candidates:
                return False
            r, c = random.choice(candidates)
            grid[r, c] = col
            return True

        all_examples = train_test_data['train'] + train_test_data['test']

        present_global = set()
        for ex in all_examples:
            present_global |= _get_other_colors_present(ex['input'])

        missing_global = [c for c in required_other_colors if c not in present_global]

        if missing_global:
            random.shuffle(all_examples)
            for col in missing_global:
                placed = False
                for ex in all_examples:
                    grid = ex['input']
                    forbidden = _build_forbidden_from_centers(grid)
                    if _place_lonely_cell(grid, forbidden, col):
                        ex['output'] = self.transform_input(grid, taskvars)
                        placed = True
                        break
                # If not placed, we skip silently (rare edge case on tiny crowded grids)

        # ------------------------------------------------------------
        # Ensure at least one train grid has cell_color1 and one has cell_color2
        # ------------------------------------------------------------
        c1 = taskvars['cell_color1']
        c2 = taskvars['cell_color2']

        if not any((ex['input'] == c1).any() for ex in train_test_data['train']):
            train_test_data['train'][0]['input'] = self.create_input(taskvars, {})
            train_test_data['train'][0]['output'] = self.transform_input(train_test_data['train'][0]['input'], taskvars)
            _sanitize_example(train_test_data['train'][0])

        if not any((ex['input'] == c2).any() for ex in train_test_data['train']):
            train_test_data['train'][-1]['input'] = self.create_input(taskvars, {})
            train_test_data['train'][-1]['output'] = self.transform_input(train_test_data['train'][-1]['input'], taskvars)
            _sanitize_example(train_test_data['train'][-1])

        return taskvars, train_test_data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create a single input grid satisfying the input reasoning chain.

        Enforced constraints:
        - random grid size between 10 and 30 (chosen per example)
        - 1..K cells of cell_color1 and 1..K cells of cell_color2 (size-aware)
        - every c1/c2 cell has its 8-neighbours empty (3x3 empty surrounding)
        - no two colored (non-zero) cells are 4-connected
        - cell_color3 and cell_color4 NEVER appear in input
        - input uses 1-4 distinct "other" colors (in addition to c1 and c2);
          NOT all other colors need to appear in every grid
        """
        # Choose grid size per-example to allow varying sizes between 10x10 and 30x30
        h = w = random.randint(10, 30)
        grid = np.zeros((h, w), dtype=int)

        # forbidden mask around already placed "centers" (c1/c2)
        forbidden = np.zeros_like(grid, dtype=bool)

        c1 = taskvars['cell_color1']
        c2 = taskvars['cell_color2']
        c3 = taskvars['cell_color3']
        c4 = taskvars['cell_color4']

        # Per-grid "other" colors: choose a random subset excluding c1/c2/c3/c4
        available_other = [c for c in range(1, 10) if c not in (c1, c2, c3, c4)]
        k_other = random.randint(1, min(4, len(available_other)))
        other_colors = random.sample(available_other, k_other)

        # helper: require 3x3 empty neighborhood AND two empty cells in cardinal directions (distance 1 and 2)
        def can_place_center(r, c):
            if not (2 <= r < h - 2 and 2 <= c < w - 2):
                return False

            # 3x3 must be empty and not reserved
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if grid[rr, cc] != 0 or forbidden[rr, cc]:
                        return False

            # cardinal distances 1 and 2 must be empty
            for dr in (-1, -2, 1, 2):
                rr = r + dr
                if 0 <= rr < h and grid[rr, c] != 0:
                    return False
            for dc in (-1, -2, 1, 2):
                cc = c + dc
                if 0 <= cc < w and grid[r, cc] != 0:
                    return False

            # and not reserved
            for dr in (-1, -2, 1, 2):
                rr = r + dr
                if 0 <= rr < h and forbidden[rr, c]:
                    return False
            for dc in (-1, -2, 1, 2):
                cc = c + dc
                if 0 <= cc < w and forbidden[r, cc]:
                    return False

            return True

        def reserve_center_region(r, c):
            # reserve 3x3
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    forbidden[rr, cc] = True
            # reserve cardinal distance 1 and 2
            for dr in (-1, -2, 1, 2):
                rr = r + dr
                if 0 <= rr < h:
                    forbidden[rr, c] = True
            for dc in (-1, -2, 1, 2):
                cc = c + dc
                if 0 <= cc < w:
                    forbidden[r, cc] = True

        # -----------------------------
        # Place 1..K of each special color (size-aware)
        # -----------------------------
        placed_c1 = 0
        placed_c2 = 0

        area = h * w
        max_k = min(3, max(1, area // 80 + 1))
        targets_c1 = random.randint(1, max_k)
        targets_c2 = random.randint(1, max_k)

        attempts = 0
        max_attempts = 1200
        while (placed_c1 < targets_c1 or placed_c2 < targets_c2) and attempts < max_attempts:
            attempts += 1
            r = random.randint(2, h - 3)
            c = random.randint(2, w - 3)

            if placed_c1 < targets_c1 and placed_c2 < targets_c2:
                want = random.choice([c1, c2])
            elif placed_c1 < targets_c1:
                want = c1
            else:
                want = c2

            if can_place_center(r, c):
                grid[r, c] = want
                reserve_center_region(r, c)
                if want == c1:
                    placed_c1 += 1
                else:
                    placed_c2 += 1

        # Force placements if we somehow failed to place at least one of either
        if placed_c1 == 0:
            for r in range(2, h - 2):
                for c in range(2, w - 2):
                    if can_place_center(r, c):
                        grid[r, c] = c1
                        reserve_center_region(r, c)
                        placed_c1 = 1
                        break
                if placed_c1:
                    break

        if placed_c2 == 0:
            for r in range(h - 3, 1, -1):
                for c in range(w - 3, 1, -1):
                    if can_place_center(r, c):
                        grid[r, c] = c2
                        reserve_center_region(r, c)
                        placed_c2 = 1
                        break
                if placed_c2:
                    break

        # -----------------------------
        # Determine total number of colored cells
        # Must fit: all special centers + at least one for each chosen other color
        # and must respect cap = rows//2
        # -----------------------------
        min_total = placed_c1 + placed_c2 + len(other_colors)
        cap = max(2, h // 2)
        max_total = min(9, cap)
        if max_total < min_total:
            max_total = min_total

        total_colored = random.randint(min_total, max_total)

        # candidate positions that can host a "lonely" colored cell
        empty_positions = [
            (r, c)
            for r in range(2, h - 2)
            for c in range(2, w - 2)
            if grid[r, c] == 0 and (not forbidden[r, c]) and can_place_center(r, c)
        ]
        random.shuffle(empty_positions)

        current_count = int((grid != 0).sum())

        # -----------------------------
        # First: place one cell for each chosen other color (guarantees distinctness per grid)
        # -----------------------------
        idx = 0
        for col in other_colors:
            while idx < len(empty_positions):
                r, c = empty_positions[idx]
                idx += 1
                if can_place_center(r, c) and grid[r, c] == 0 and not forbidden[r, c]:
                    grid[r, c] = col
                    current_count += 1
                    break

        # -----------------------------
        # Then: place remaining extra cells (still only from other_colors)
        # -----------------------------
        while current_count < total_colored and idx < len(empty_positions):
            r, c = empty_positions[idx]
            idx += 1
            if can_place_center(r, c) and grid[r, c] == 0 and not forbidden[r, c]:
                grid[r, c] = random.choice(other_colors)
                current_count += 1

        # -----------------------------
        # Final safety:
        # - ensure no accidental 4-connected colored objects exist
        # - ensure c3/c4 not present
        # -----------------------------
        objs = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        for obj in objs:
            if len(obj) > 1:
                cells = list(obj.coords)
                for (rr, cc) in cells[1:]:
                    if not forbidden[rr, cc]:
                        grid[rr, cc] = 0

        grid[(grid == c3) | (grid == c4)] = 0

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Apply the transformation chain to produce the output grid.

        For every cell of color cell_color1 add diagonally adjacent cells of color cell_color3.
        For every cell of color cell_color2 add orthogonally adjacent cells of color cell_color4.
        """
        out = grid.copy()
        h, w = grid.shape
        c1 = taskvars['cell_color1']
        c2 = taskvars['cell_color2']
        c3 = taskvars['cell_color3']
        c4 = taskvars['cell_color4']

        centers_c1 = list(zip(*np.where(grid == c1)))
        centers_c2 = list(zip(*np.where(grid == c2)))

        # For c1: diagonals
        for (r, c) in centers_c1:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = c3

        # For c2: orthogonals
        for (r, c) in centers_c2:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = c4

        return out
