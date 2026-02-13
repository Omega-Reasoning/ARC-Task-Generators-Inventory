from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Optional

class Task14b8e18cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}×{vars['grid_size']}.",
            "Each grid has a completely filled background of {color('background')} color, along with several same-colored objects. These objects can be filled or unfilled squares and rectangles, and sometimes include patterns like [[{color('background')}, c], [c, {color('background')}]], where c is the object color.",
            "Each grid contains at least one square-shaped object.",
            "All objects must be completely separated from each other.",
            "In some cases, a smaller object may appear inside an unfilled rectangle or square, but it must not touch the surrounding shape."
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying all filled and unfilled squares.",
            "Once identified, 8 {color('background')} cells around each of these squares are recolored to {color('fill')}.",
            "The recolored {color('fill')} cells are the exterior cells surrounding the square.",
            "Specifically, consider the 4 corner cells of each square, and recolor the adjacent 4-way connected exterior cells—those directly above, below, to the left, or to the right of the square—but only if they lie outside the square."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # -------------------- Public API --------------------

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'grid_size': random.randint(14, 30),  # higher min to avoid tight packing
            'fill': random.randint(1, 9),
            'background': random.randint(1, 9),
        }
        while taskvars['fill'] == taskvars['background']:
            taskvars['fill'] = random.randint(1, 9)

        train_data, test_data = [], []
        has_unfilled_square = False
        special_grid_index = random.randint(0, 2)

        for i in range(3):
            # We'll force an unfilled square FIRST on the designated grid if not present yet.
            force_unfilled_first = (not has_unfilled_square) and (i == 2)

            input_grid = self.create_input(
                taskvars,
                {
                    'force_unfilled_first': force_unfilled_first,
                    'add_diagonal_object': (i == special_grid_index),
                },
            )
            output_grid = self.transform_input(input_grid, taskvars)

            if self._has_unfilled_square(input_grid, taskvars):
                has_unfilled_square = True

            train_data.append({'input': input_grid, 'output': output_grid})

        # test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data.append({'input': test_input, 'output': test_output})

        return taskvars, {'train': train_data, 'test': test_data}
    
    def _place_square_first(
        self,
        grid: np.ndarray,
        color: int,
        background: int,
        placed_objects: List[Dict[str, int]],
        buffer: int,
        max_side: int,
        hollow: bool,
    ) -> bool:
        H, W = grid.shape
        # Prefer modest sizes so they fit often
        side_candidates = list(range(min(4, max_side), max(2, max_side + 1))) or [2, 3]
        random.shuffle(side_candidates)

        for side in side_candidates:
            h = w = side
            if not self._can_fit_anywhere(grid.shape, placed_objects, h, w, buffer):
                continue

            rmin, rmax = buffer, H - h - buffer
            cmin, cmax = buffer, W - w - buffer
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if self._has_sufficient_separation(r, c, h, w, placed_objects, buffer):
                        self._place_object(grid, r, c, h, w, color, filled=not hollow, background=background)
                        placed_objects.append({'row': r, 'col': c, 'height': h, 'width': w})
                        return True
        return False


    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        background = taskvars['background']
        buffer = 2 if grid_size >= 16 else 1

        grid = np.full((grid_size, grid_size), background, dtype=int)

        # choose object color different from background & fill
        object_colors = [c for c in range(1, 10) if c != background and c != taskvars['fill']]
        object_color = random.choice(object_colors)

        # object counts adapt to grid size
        if grid_size < 16:
            num_objects = random.randint(2, 3)
            max_side = max(2, grid_size // 5)  # smaller on tiny grids
        else:
            num_objects = random.randint(2, 4)
            max_side = max(3, grid_size // 4)

        placed_objects: List[Dict[str, int]] = []

        # --- NEW: ALWAYS place at least one square first on every grid ---
        # 50% chance hollow; if 'force_unfilled_first' is requested, make it hollow.
        must_be_hollow = bool(gridvars.get('force_unfilled_first', False))
        prefer_hollow = must_be_hollow or (random.random() < 0.5)

        placed_square = self._place_square_first(
            grid=grid,
            color=object_color,
            background=background,
            placed_objects=placed_objects,
            buffer=buffer,
            max_side=max_side,
            hollow=prefer_hollow
        )
        if not placed_square:
            # Relax just this placement
            placed_square = self._place_square_first(
                grid=grid,
                color=object_color,
                background=background,
                placed_objects=placed_objects,
                buffer=1,
                max_side=max(2, max_side - 1),
                hollow=prefer_hollow
            )
        # If still not placed, we carry on; later logic may still create squares,
        # but in practice the relaxed pass virtually always succeeds.

        # Now place remaining objects (mix of rectangles/squares, some filled)
        objects_created = 1 if placed_square else 0
        max_attempts = 50
        attempts = 0

        while objects_created < num_objects and attempts < max_attempts:
            attempts += 1
            # Randomly decide type and size
            want_square = (random.random() < 0.5)
            if want_square:
                side = random.randint(2, max_side)
                h, w = side, side
            else:
                h = random.randint(2, max_side)
                w = random.randint(2, max_side)
                if h == w:
                    if w < max_side: w += 1
                    else: h = max(2, h - 1)

            if not self._can_fit_anywhere(grid.shape, placed_objects, h, w, buffer):
                continue

            filled = random.random() < 0.6

            placed_ok = False
            for _ in range(12):
                rmin, rmax = buffer, grid_size - h - buffer
                cmin, cmax = buffer, grid_size - w - buffer
                if rmax < rmin or cmax < cmin:
                    break

                row = random.randint(rmin, rmax)
                col = random.randint(cmin, cmax)

                if self._has_sufficient_separation(row, col, h, w, placed_objects, buffer):
                    self._place_object(grid, row, col, h, w, object_color, filled, background)
                    placed_objects.append({'row': row, 'col': col, 'height': h, 'width': w})
                    objects_created += 1
                    placed_ok = True
                    break

            if not placed_ok:
                continue

        # Optional: add the tiny 2-cell diagonal object
        if gridvars.get('add_diagonal_object', False):
            self._add_diagonal_object(grid, object_color, background, placed_objects, buffer)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        background = taskvars['background']
        fill = taskvars['fill']
        output = grid.copy()

        # Keep default 4-connectivity to avoid diagonal pairs being misclassified
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=background)

        for obj in objects:
            if self._looks_like_square(obj, grid, background):
                self._outline_square(output, obj, fill, background)

        return output

    # -------------------- Helpers: detection & outlining --------------------

    def _has_unfilled_square(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> bool:
        """True if any object is an unfilled square (hollow, 1px border)."""
        background = taskvars['background']
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=background)
        for obj in objects:
            kind = self._square_kind(obj, grid, background)
            if kind == "hollow":
                return True
        return False

    def _looks_like_square(self, obj: GridObject, grid: np.ndarray, background: int) -> bool:
        """Accept solid or hollow squares; reject diagonal pairs in a 2x2 box."""
        return self._square_kind(obj, grid, background) in ("solid", "hollow")

    def _square_kind(self, obj: GridObject, grid: np.ndarray, background: int) -> Optional[str]:
        """
        Returns "solid" | "hollow" if the object looks like a square,
        otherwise None. Robust to diagonal pairs etc.
        """
        bbox = obj.bounding_box  # (rows_slice, cols_slice)
        r0, r1 = bbox[0].start, bbox[0].stop
        c0, c1 = bbox[1].start, bbox[1].stop
        h, w = r1 - r0, c1 - c0
        if h != w or h < 2:
            return None

        region = grid[r0:r1, c0:c1]
        # Determine predominant (non-background) edge color if present
        # Quickly get a candidate color from any non-background cell in region
        vals, counts = np.unique(region[region != background], return_counts=True)
        if len(vals) == 0:
            return None
        color = int(vals[np.argmax(counts)])

        # Solid check: entire region is that color (non-background)
        if np.all(region == color):
            return "solid"

        # Hollow check: border all color, interior all background, border thickness = 1
        if h >= 2:
            top = region[0, :]
            bottom = region[-1, :]
            left = region[:, 0]
            right = region[:, -1]
            if (
                np.all(top == color)
                and np.all(bottom == color)
                and np.all(left == color)
                and np.all(right == color)
            ):
                if h == 2:
                    # A 2x2 "hollow" square degenerates to full border (no interior).
                    # Ensure it's exactly the four border lines (i.e., all 4 cells color).
                    if np.all(region == color):
                        return "solid"  # that's effectively a solid 2x2
                    else:
                        return None
                else:
                    interior = region[1:-1, 1:-1]
                    if np.all(interior == background):
                        return "hollow"

        return None

    def _outline_square(self, grid: np.ndarray, square: GridObject, fill: int, background: int):
        """Add outline around a square by coloring 8 exterior cells around corners."""
        bbox = square.bounding_box
        top_row = bbox[0].start
        bottom_row = bbox[0].stop - 1
        left_col = bbox[1].start
        right_col = bbox[1].stop - 1

        H, W = grid.shape
        outline_positions = [
            (top_row - 1, left_col),     # above TL
            (top_row, left_col - 1),     # left TL
            (top_row - 1, right_col),    # above TR
            (top_row, right_col + 1),    # right TR
            (bottom_row, left_col - 1),  # left BL
            (bottom_row + 1, left_col),  # below BL
            (bottom_row, right_col + 1), # right BR
            (bottom_row + 1, right_col), # below BR
        ]
        for r, c in outline_positions:
            if 0 <= r < H and 0 <= c < W and grid[r, c] == background:
                grid[r, c] = fill

    # -------------------- Helpers: placement --------------------

    def _place_unfilled_square_first(
        self,
        grid: np.ndarray,
        color: int,
        background: int,
        placed_objects: List[Dict[str, int]],
        buffer: int,
        max_side: int,
    ) -> bool:
        """Try to place a single unfilled square before anything else, to guarantee the requirement."""
        H, W = grid.shape
        side_candidates = list(range(min(4, max_side), max(2, max_side + 1)))  # prefer modest sizes
        if not side_candidates:
            side_candidates = [2, 3]
        random.shuffle(side_candidates)

        for side in side_candidates:
            h = w = side
            if not self._can_fit_anywhere(grid.shape, placed_objects, h, w, buffer):
                continue
            # scan positions deterministically to avoid long rejection storms
            rmin, rmax = buffer, H - h - buffer
            cmin, cmax = buffer, W - w - buffer
            for r in range(rmin, rmax + 1):
                for c in range(cmin, cmax + 1):
                    if self._has_sufficient_separation(r, c, h, w, placed_objects, buffer):
                        # place unfilled (hollow) border
                        self._place_object(grid, r, c, h, w, color, filled=False, background=background)
                        placed_objects.append({'row': r, 'col': c, 'height': h, 'width': w})
                        return True
        return False

    def _can_fit_anywhere(
        self,
        grid_shape: Tuple[int, int],
        placed_objects: List[Dict[str, int]],
        h: int,
        w: int,
        buffer: int,
    ) -> bool:
        """Cheap feasibility check: is there *any* (r,c) where this box could go with separation?"""
        H, W = grid_shape
        rmin, rmax = buffer, H - h - buffer
        cmin, cmax = buffer, W - w - buffer
        if rmax < rmin or cmax < cmin:
            return False

        # Quick bounding-box test against all placed objects to avoid O(N^3) scan:
        # We don't test every cell; instead, we check if there's ANY room left by
        # comparing available 'corridors' between dilated boxes.
        # For simplicity (still fast), probe a small set of candidate anchors.
        probes = []
        step_r = max(1, (rmax - rmin) // 4)
        step_c = max(1, (cmax - cmin) // 4)
        for rr in range(rmin, rmax + 1, step_r):
            for cc in range(cmin, cmax + 1, step_c):
                probes.append((rr, cc))
        probes.extend([(rmin, cmin), (rmin, cmax), (rmax, cmin), (rmax, cmax)])

        for r, c in probes:
            if self._has_sufficient_separation(r, c, h, w, placed_objects, buffer):
                return True
        return False

    def _has_sufficient_separation(
        self,
        row: int,
        col: int,
        height: int,
        width: int,
        placed_objects: List[Dict[str, int]],
        buffer: int,
    ) -> bool:
        """Bounding-box overlap test with buffer 'halo' around every object."""
        new_min_row = row - buffer
        new_max_row = row + height + buffer - 1
        new_min_col = col - buffer
        new_max_col = col + width + buffer - 1

        for obj in placed_objects:
            existing_min_row = obj['row'] - buffer
            existing_max_row = obj['row'] + obj['height'] + buffer - 1
            existing_min_col = obj['col'] - buffer
            existing_max_col = obj['col'] + obj['width'] + buffer - 1

            if not (
                new_max_row < existing_min_row
                or new_min_row > existing_max_row
                or new_max_col < existing_min_col
                or new_min_col > existing_max_col
            ):
                return False
        return True

    def _place_object(
        self,
        grid: np.ndarray,
        row: int,
        col: int,
        height: int,
        width: int,
        color: int,
        filled: bool,
        background: int,
    ):
        if filled:
            grid[row : row + height, col : col + width] = color
        else:
            grid[row : row + height, col : col + width] = background
            grid[row, col : col + width] = color
            grid[row + height - 1, col : col + width] = color
            grid[row : row + height, col] = color
            grid[row : row + height, col + width - 1] = color

    def _add_diagonal_object(
        self,
        grid: np.ndarray,
        object_color: int,
        background: int,
        placed_objects: List[Dict[str, int]],
        buffer: int,
    ):
        H, W = grid.shape
        diagonal_patterns = [
            [(0, 0), (1, 1)],  # TL->BR
            [(0, 1), (1, 0)],  # TR->BL
        ]
        pattern = random.choice(diagonal_patterns)

        max_attempts = 40
        for _ in range(max_attempts):
            row_min, row_max = buffer, H - 2 - buffer
            col_min, col_max = buffer, W - 2 - buffer
            if row_max < row_min or col_max < col_min:
                return
            row = random.randint(row_min, row_max)
            col = random.randint(col_min, col_max)

            if not self._has_sufficient_separation(row, col, 2, 2, placed_objects, buffer):
                continue

            r1, c1 = row + pattern[0][0], col + pattern[0][1]
            r2, c2 = row + pattern[1][0], col + pattern[1][1]

            if grid[r1, c1] == background and grid[r2, c2] == background:
                grid[r1, c1] = object_color
                grid[r2, c2] = object_color
                placed_objects.append({'row': row, 'col': col, 'height': 2, 'width': 2})
                return

    # -------------------- Misc helpers --------------------

    def _count_squares_by_scan(self, grid: np.ndarray, background: int) -> int:
        """Lightweight scan via connected components to count squares (solid or hollow)."""
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=background)
        cnt = 0
        for obj in objects:
            if self._looks_like_square(obj, grid, background):
                cnt += 1
        return cnt
