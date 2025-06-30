import random
from typing import Dict, Any, Tuple, List

import numpy as np

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects


class Task142ca369Generator(ARCTaskGenerator):
    """ARC-AGI task generator for the diagonal-chain pattern described in the
    reasoning chains shipped with the benchmark specification."""
    def __init__(self):
        self.input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each grid contains several colored (1–9) objects made of 4-way connected cells, with the remaining cells being empty (0).",
            "The colored objects can have 4 possible shapes; an arrowhead defined by [[c, c], [c, 0]], a vertical strip made of three consecutive cells, a horizontal strip made of three consecutive cells and a 1×1 colored block.",
            "The arrow-head can point in any diagonal direction. In matrix form;  [[c,c],[c,0]] → tip points north-west (↖), [[c,0],[c,c]] → tip points south-west (↙), [[0,c],[c,c]] → tip points south-east (↘), [[c,c],[0,c]] → tip points north-east (↗). The empty corner in the 2×2 block always lies opposite the direction in which the subsequent diagonal will grow in output grid.",
            "To construct the input grid, begin by placing arrowhead-shaped objects such that each one can be extended diagonally in its orientation direction by several cells before reaching the grid boundary. Arrowheads always have priority and must be placed first. After placing an arrowhead, extend its diagonal by adding 4 to 5 same-colored cells in that direction.",
            "In some examples once the diagonal is partially extended, place a 3-cell vertical or horizontal strip exactly where you want to stop the extension. This strip acts as a stopper, and the diagonal line must be 4-way connected to the middle cell of the strip to ensure proper alignment. Then, reflect the pattern on the opposite side, oriented such that a new diagonal line begins in the opposite direction. Extend this new diagonal and again stop it using another vertical or horizontal strip, making sure the diagonal is 4-way connected to strip.",
            "Sometimes we use 1×1 blocks instead of vertical or horizontal strip, follow the same rule as vertical and horizontal strips and serve as part of the chain logic.",
            "No object should be placed randomly. All objects must be aligned with the direction of an inferred diagonal path.",
            "Arrowheads are positioned such that their diagonal extensions intersect with another object, forming a complete diagonal chain in the output. Objects that are part of a diagonal chain must follow consistent spacing, so the diagonal path between them can be filled seamlessly.",
            "In some cases only place arrow heads without any 1x1 blocks or vertical or horizontal strips, in this case the diagonal extension will continue until it reaches the grid boundary.",
            "Ensure no diagonal paths are interupted by each other.",
            "Once all arrowheads and strips have been added according to the logic, remove all the extended diagonal lines, ensuring that the grid contains only arrowheads, vertical or horizontal strips, and 1×1 blocks."
        ]

        self.transformation_reasoning_chain = [
            "The output grid is a copy of the input grid with additional single cells added along inferred diagonal paths, based on the structure and orientation of arrowhead-shaped objects",
            "The arrowhead can point in any of the four diagonal directions. In matrix form; [[c, c], [c, 0]] → tip points north-west (↖), [[c, 0], [c, c]] → tip points south-west (↙), [[0, c], [c, c]] → tip points south-east (↘), [[c, c], [0, c]] → tip points north-east (↗). The empty corner in the 2×2 block always lies opposite to the direction in which the diagonal will grow in the output grid.",
            "For every arrowhead in the input grid, the diagonal extension begins from its tip, in the arrow’s pointing direction. From there, the diagonal advances step by step, coloring each empty cell with the same color as the arrowhead.",
            "If the diagonal would land on the center cell of a 3-cell vertical or horizontal strip, the extension stops one cell before the strip. That final filled cell is 4-way connected to the strip’s center cell and becomes the pivot for reflection. The color of pivot cell and reflected diagonal colors are changed to match color of strip.",
            "If the diagonal reaches a 1×1 block such that the diagonal would land directly above the 1x1 block, the diagonal stops at the current cell. A reflection then occurs, and the diagonal continues in the opposite diagonal direction from that point.",
            "This reflection continues the chain logic until grid boundary has been reached.",
            "All inferred diagonals are filled to complete the visible chains between objects in the output.",
            "No new shapes are introduced — all added cells are extensions of the existing structures.",
            "Diagonal filling strictly follows the input logic; misaligned or arbitrary diagonals are not allowed."
        ]

        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)


    _ORI_MAP = {
        "NW": {
            "dx": -1, "dy": -1, "tip": (0, 0), "cells": [(0, 0), (0, 1), (1, 0)]
        },
        "NE": {
            "dx": -1, "dy": +1, "tip": (0, 1), "cells": [(0, 0), (0, 1), (1, 1)]
        },
        "SW": {
            "dx": +1, "dy": -1, "tip": (1, 0), "cells": [(0, 0), (1, 0), (1, 1)]
        },
        "SE": {
            "dx": +1, "dy": +1, "tip": (1, 1), "cells": [(0, 1), (1, 0), (1, 1)]
        }
    }

    def _can_place(self, grid: np.ndarray, coords: List[Tuple[int, int]]) -> bool:
        """True iff all coords are inside grid and currently empty."""
        s = grid.shape[0]
        return all(0 <= r < s and 0 <= c < s and grid[r, c] == 0 for r, c in coords)

    def _place_arrow(self, grid: np.ndarray, color: int, orientation: str, tip_r: int, tip_c: int):
        """Place an arrow on *grid* with *color* and given orientation such that the
        tip sits at (tip_r, tip_c)."""
        rels = self._ORI_MAP[orientation]
        # compute top-left of the 2×2 bounding box
        tip_rel_r, tip_rel_c = rels["tip"]
        box_r0 = tip_r - tip_rel_r
        box_c0 = tip_c - tip_rel_c
        coords = [(box_r0 + dr, box_c0 + dc) for dr, dc in rels["cells"]]
        if not self._can_place(grid, coords):
            raise ValueError("Cannot place arrow at requested location – overlap or OOB.")
        for r, c in coords:
            grid[r, c] = color
        return (tip_r, tip_c, rels["dx"], rels["dy"])

    def _place_block(self, grid: np.ndarray, color: int, r: int, c: int):
        if not self._can_place(grid, [(r, c)]):
            raise ValueError("Cannot place block.")
        grid[r, c] = color

    def _place_strip(self, grid: np.ndarray, color: int, r: int, c: int, vertical: bool):
        if vertical:
            coords = [(r - 1, c), (r, c), (r + 1, c)]
        else:
            coords = [(r, c - 1), (r, c), (r, c + 1)]
        if not self._can_place(grid, coords):
            raise ValueError("Cannot place strip.")
        for rr, cc in coords:
            grid[rr, cc] = color

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        size: int = taskvars['grid_size']
        grid = np.zeros((size, size), dtype=int)

        case: str = gridvars['case']
        rng = random.Random()  # local RNG for variety

        # sample a pool of distinct colors (we won't re-use colors for arrows & strips)
        available_colors = list(range(1, 10))
        rng.shuffle(available_colors)

        # convenience lambdas --------------------------------------------------
        def sample_tip(dx: int, dy: int, path_len: int) -> Tuple[int, int]:
            """Choose a tip so that a diagonal of *path_len* cells fits before hitting
            the boundary in (dx,dy) direction."""
            r_lo = path_len if dx == -1 else 1
            r_hi = size - 2 - path_len if dx == +1 else size - 3
            c_lo = path_len if dy == -1 else 1
            c_hi = size - 2 - path_len if dy == +1 else size - 3
            if r_lo > r_hi or c_lo > c_hi:
                raise ValueError("Grid too small for requested path length.")
            return rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi)


        if case == 'case4':
            # ❶ how many pairs can fit without overlap
            n_req    = size // 2 - 2            # theoretical maximum
            n_arrows = (n_req + 1) // 2         # ceil(n_req / 2)

            # ❷ RESEED the case-local RNG so each run starts from a new state
            rng.seed(random.getrandbits(64))

            # ❸ flip a coin with that fresh RNG
            side = rng.choice(['left', 'right'])   # left  ➜ top-left quadrant (↘)
                                                # right ➜ top-right quadrant (↙)

            # ❹ set all geometry parameters in one place
            if side == 'left':                       # ↘  southeast arrows
                ori      = 'SE'
                corr_dir = (+1, +1)                  # corridor goes down-right
                base_tl  = (n_req, 0)                # first arrow’s top-left
                stride   = (-2, +1)                  # up two rows, right one col
                blk_col  = size // 2                 # blocks in centre-right column
                tip_off  = (1, 1)                    # tip offset inside 2×2 box
                filled   = [(0, 1), (1, 0), (1, 1)]  # filled cells in SE arrow
            else:                                    # ↙  southwest arrows
                ori      = 'SW'
                corr_dir = (+1, -1)                  # corridor goes down-left
                base_tl  = (n_req, size - 2)         # near top-right edge
                stride   = (-2, -1)                  # up two rows, left one col
                blk_col  = size // 2 - 1             # blocks in centre-left column
                tip_off  = (1, 0)                    # tip offset for SW arrow
                filled   = [(0, 0), (1, 0), (1, 1)]  # filled cells in SW arrow

            # ❺ enough distinct colours – recycle palette if n_arrows > 9
            colours = (available_colors * ((n_arrows + 8) // 9))[:n_arrows]
            reserved: set[tuple[int, int]] = set()   # track every occupied cell

            for k, colour in enumerate(colours):
                # ---------- 1️⃣ place the arrow-head ------------------------------
                tl_r = base_tl[0] + stride[0] * k
                tl_c = base_tl[1] + stride[1] * k
                tip_r, tip_c = tl_r + tip_off[0], tl_c + tip_off[1]
                self._place_arrow(grid, colour, ori, tip_r, tip_c)

                # mark the three filled cells of the arrow
                reserved.update([(tl_r + dr, tl_c + dc) for dr, dc in filled])

                # ---------- 2️⃣ reserve the empty diagonal corridor ---------------
                dx, dy = corr_dir
                steps  = abs(blk_col - tip_c)        # ≥ 1 by construction
                corridor = [(tip_r + s * dx, tip_c + s * dy) for s in range(1, steps)]
                if any(cell in reserved or grid[cell] != 0 for cell in corridor):
                    raise RuntimeError("Case 4 corridor overlap – should never happen")
                reserved.update(corridor)

                # ---------- 3️⃣ place the matching 1×1 block ----------------------
                blk_r = tip_r + steps * dx+1           # lands exactly on the diagonal
                self._place_block(grid, colour, blk_r, blk_col)
                reserved.add((blk_r, blk_col))



        elif case == 'case2':
            orientations = ['NW', 'NE', 'SW', 'SE']
            arrow_colors = [available_colors.pop() for _ in range(4)]
            quadrant_ranges = {
                'NW': ((0, size // 2), (0, size // 2)),
                'NE': ((0, size // 2), (size // 2, size)),
                'SW': ((size // 2, size), (0, size // 2)),
                'SE': ((size // 2, size), (size // 2, size)),
            }

            placed_all = True
            for ori, color in zip(orientations, arrow_colors):
                dx, dy = self._ORI_MAP[ori]['dx'], self._ORI_MAP[ori]['dy']
                row_range, col_range = quadrant_ranges[ori]
                placed = False

                for _ in range(200):
                    # Sample tip within quadrant but ensure diagonal has space
                    try:
                        tip_r = random.randint(max(row_range[0] + 3, 1), min(row_range[1] - 3, size - 2))
                        tip_c = random.randint(max(col_range[0] + 3, 1), min(col_range[1] - 3, size - 2))

                        # Simulate path and check for clean diagonal
                        path_cells = set()
                        rr, cc = tip_r + dx, tip_c + dy
                        while 0 <= rr < size and 0 <= cc < size:
                            if grid[rr, cc] != 0:
                                break  # blocked
                            path_cells.add((rr, cc))
                            rr += dx
                            cc += dy

                        self._place_arrow(grid, color, ori, tip_r, tip_c)
                        placed = True
                        break
                    except ValueError:
                        continue

                if not placed:
                    placed_all = False
                    break

            # Fallback: same-orientation non-intersecting diagonals in one quadrant
            if not placed_all:
                ori = 'SE'  # choose any one
                dx, dy = self._ORI_MAP[ori]['dx'], self._ORI_MAP[ori]['dy']
                arrow_colors = [available_colors.pop() for _ in range(4)]
                existing_paths: List[set] = []

                for col in arrow_colors:
                    for _ in range(200):
                        tip_r = random.randint(1, size // 2 - 3)
                        tip_c = random.randint(1, size // 2 - 3)

                        path_cells = set()
                        rr, cc = tip_r + dx, tip_c + dy
                        while 0 <= rr < size and 0 <= cc < size:
                            if grid[rr, cc] != 0:
                                break
                            path_cells.add((rr, cc))
                            rr += dx
                            cc += dy

                        if any(path_cells & ep for ep in existing_paths):
                            continue

                        try:
                            self._place_arrow(grid, col, ori, tip_r, tip_c)
                            existing_paths.append(path_cells)
                            break
                        except ValueError:
                            continue
                    else:
                        raise RuntimeError("Could not place fallback arrows for case2.")

        elif case == 'case3':
            arrow_col   = available_colors.pop()
            n_strips    = 2
            strip_cols  = [available_colors.pop() for _ in range(n_strips)]

            # ── helper ─────────────────────────────────────────────────────
            def place_strip_with_end(end_r: int, end_c: int,
                                    dx: int, dy: int, col: int,
                                    stop_before: bool) -> tuple[bool, bool, tuple[int,int]]:
                """
                Place a 3-cell strip so that (end_r,end_c) is one of its ends.

                If stop_before==True  → caller’s pivot is (end_r−dx, end_c−dy)
                If stop_before==False → diagonal will land ON (end_r,end_c)

                Returns (placed?, is_vertical, pivot_rc)
                """
                for is_vert in (True, False):
                    if is_vert:
                        ctr_r = end_r + (+1 if dx < 0 else -1)
                        ctr_c = end_c
                        coords = [(ctr_r - 1, ctr_c),
                                (ctr_r,     ctr_c),
                                (ctr_r + 1, ctr_c)]
                    else:
                        ctr_r = end_r
                        ctr_c = end_c + (+1 if dy < 0 else -1)
                        coords = [(ctr_r, ctr_c - 1),
                                (ctr_r, ctr_c    ),
                                (ctr_r, ctr_c + 1)]

                    if 0 <= ctr_r - 1 and ctr_r + 1 < size and \
                    0 <= ctr_c - 1 and ctr_c + 1 < size and \
                    self._can_place(grid, coords):
                        self._place_strip(grid, col, ctr_r, ctr_c, is_vert)
                        pivot = (end_r - dx, end_c - dy) if stop_before else (end_r, end_c)
                        return True, is_vert, pivot
                return False, False, (-1, -1)

            # ── build the chain  (retry up to 400 times) ───────────────────
            for _ in range(400):
                snapshot = grid.copy()

                # 1) arrow orientation & first leg length -------------------
                ori         = rng.choice(list(self._ORI_MAP.keys()))
                dx, dy      = self._ORI_MAP[ori]['dx'], self._ORI_MAP[ori]['dy']
                L1          = rng.randint(4, 6)

                tip_r = rng.randint(1, size - 2)
                tip_c = rng.randint(1, size - 2)
                endA_r = tip_r + dx * (L1 + 1)          # strip A end
                endA_c = tip_c + dy * (L1 + 1)
                if not (1 <= endA_r < size - 1 and 1 <= endA_c < size - 1):
                    continue
                # corridor up to *pivot1* must be empty
                if any(grid[tip_r + dx*s, tip_c + dy*s] != 0 for s in range(1, L1 + 1)):
                    continue
                try:
                    self._place_arrow(grid, arrow_col, ori, tip_r, tip_c)
                except ValueError:
                    grid[:] = snapshot; continue

                ok, vertA, (pivot1_r, pivot1_c) = \
                    place_strip_with_end(endA_r, endA_c, dx, dy,
                                        strip_cols[0], stop_before=True)
                if not ok:
                    grid[:] = snapshot; continue

                # reflected direction after strip A
                dx2, dy2 = (dx, -dy) if vertA else (-dx, dy)

                # 2) second leg length & strip B ----------------------------
                L2 = rng.randint(4, 6)
                endB_r = pivot1_r + dx2 * (L2)          # diagonal lands ON this end
                endB_c = pivot1_c + dy2 * (L2)
                if not (1 <= endB_r < size - 1 and 1 <= endB_c < size - 1):
                    grid[:] = snapshot; continue
                # corridor to endB must be empty (excluding pivot1)
                if any(grid[pivot1_r + dx2*s, pivot1_c + dy2*s] != 0 for s in range(1, L2)):
                    grid[:] = snapshot; continue

                ok, _, _ = place_strip_with_end(endB_r, endB_c, dx2, dy2,
                                                strip_cols[1], stop_before=False)
                if not ok:
                    grid[:] = snapshot; continue

                # optional decorative third strip --------------------------
                if n_strips == 3:
                    col = strip_cols[2]
                    for _ in range(200):
                        v = rng.choice([True, False])
                        if v:
                            r = rng.randint(1, size - 2); c = rng.randint(0, size - 1)
                        else:
                            r = rng.randint(0, size - 1); c = rng.randint(1, size - 2)
                        try:
                            self._place_strip(grid, col, r, c, v)
                            break
                        except ValueError:
                            continue
                break
            else:
                raise RuntimeError("Case 3: failed to build arrow-strip chain")

        elif case == 'test':
            mid = size // 2                       # centre column
            colours = iter(available_colors * 3)  # endless colour supply

            def put_tip(r: int, c: int, ori: str):
                self._place_arrow(grid, next(colours), ori, r, c)

            # 0️⃣  TOP-LEFT corner  (tip 1,1)  ↘   fills (1,0) & (0,1)
            put_tip(1, 1, 'SE')

            # 1️⃣  LEFT ladder – odd rows 3,5,…,mid-3   column 1   ↘
            for r in range(3, mid - 1, 2):
                put_tip(r, 1, 'SE')

            # 2️⃣  TOP-LEFT edge – row 1,  odd columns 3,5,…,mid-3   ↘
            for c in range(3, mid - 1, 2):        # stops at mid-3
                put_tip(1, c, 'SE')

            # -------- 2×2 empty gap: columns mid-1 & mid, rows 0-1 --------

            # 3️⃣  TOP-RIGHT edge – row 1, odd columns mid+1, mid+3,… ≤ N-3  ↙
            for c in range(mid + 1, size - 2, 2):
                put_tip(1, c, 'SW')

            # 4️⃣  RIGHT ladder – odd rows 3,5,…,mid-3   column N-2  ↙
            for r in range(3, mid - 1, 2):
                put_tip(r, size - 2, 'SW')

            # 5️⃣  TOP-RIGHT corner  (tip 1,N-2)  ↙   fills (0,N-2) & (1,N-1)
            put_tip(1, size - 2, 'SW')
                # ------------------------------------------------------------------

   
        elif case == 'test2':
            mid = size // 2                      # vertical centre
            colour_A, colour_B = available_colors[0], available_colors[1]

            # 1️⃣  two top-corner arrow-heads
            self._place_arrow(grid, colour_A, 'SE', 1, 1)          # top-left  ↘
            self._place_arrow(grid, colour_B, 'SW', 1, size - 2)   # top-right ↙

            # 2️⃣  vertical strip A  (colour_A)  – column mid-1, rows mid-4..mid-2
            self._place_strip(grid, colour_A, mid - 2, mid - 1, vertical=True)


            # 3️⃣  vertical strip B  (colour_B)  – column mid+1, rows mid-4..mid-2
            self._place_strip(grid, colour_B, mid - 3, mid + 1, vertical=True)





        return grid
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        • Trace the diagonal that starts one step away from each arrow-head tip.
        • Re-colour the *pivot* cell that is four-way adjacent to a strip
          with the strip’s colour, then reflect once.
        • After the first reflection, landing on a second strip (end cell)
          stops the walk.
        • 1×1 blocks bounce vertically (dx *= –1) as in case 4.
        """
        size   = grid.shape[0]
        output = grid.copy()

        # ─── 1. classify every connected monochrome object ───────────────
        objs = find_connected_objects(grid,
                                      diagonal_connectivity=False,
                                      background=0,
                                      monochromatic=True)

        coord_to_type: Dict[Tuple[int, int], str] = {}
        arrow_objs = []

        for obj in objs:
            h, w, sz = obj.height, obj.width, obj.size
            if sz == 3 and h == 2 and w == 2:                 # arrow-head
                arrow_objs.append(obj)
                for r, c, _ in obj:
                    coord_to_type[(r, c)] = 'arrow'
            elif sz == 3 and ((h == 3 and w == 1) or (h == 1 and w == 3)):
                typ = 'vstrip' if h == 3 else 'hstrip'        # 3-cell strip
                for r, c, _ in obj:
                    coord_to_type[(r, c)] = typ
            elif sz == 1:                                     # 1×1 block
                for r, c, _ in obj:
                    coord_to_type[(r, c)] = 'block'
            else:                                             # barrier
                for r, c, _ in obj:
                    coord_to_type[(r, c)] = 'other'

        # ─── 2. helper: missing cell in 2×2 box → (dx,dy) ────────────────
        missing_to_vec = {(1, 1): (-1, -1), (1, 0): (-1, +1),
                          (0, 1): (+1, -1), (0, 0): (+1, +1)}

        # ─── 3. walk every arrow’s diagonal ──────────────────────────────
        for arrow in arrow_objs:
            # infer orientation & tip
            box = arrow.bounding_box
            arr = np.zeros((2, 2), dtype=int)
            for r, c, _ in arrow:
                arr[r - box[0].start, c - box[1].start] = 1
            missing = next((i, j) for i in range(2) for j in range(2)
                            if arr[i, j] == 0)

            dx, dy = missing_to_vec[missing]
            tip_r  = box[0].start + (1 - missing[0])
            tip_c  = box[1].start + (1 - missing[1])
            colour = next(iter(arrow.colors))

            cur_r, cur_c = tip_r + dx, tip_c + dy
            steps, max_steps = 0, size * 4      # safety

            while 0 <= cur_r < size and 0 <= cur_c < size and steps < max_steps:
                steps += 1
                cell_typ = (
                    coord_to_type.get((cur_r, cur_c))
                    if grid[cur_r, cur_c] != 0 else None
                )

                # ---------- empty cell ----------------------------------
                if cell_typ is None:
                    # look-ahead: will the next cell hit a v/h strip?
                    nxt_r, nxt_c = cur_r + dx, cur_c + dy
                    nxt_typ = (
                        coord_to_type.get((nxt_r, nxt_c))
                        if 0 <= nxt_r < size and 0 <= nxt_c < size
                           and grid[nxt_r, nxt_c] != 0 else None
                    )

                    # ★ pivot on LAST diagonal cell right before a strip
                    if nxt_typ in ('vstrip', 'hstrip'):
                        strip_colour = grid[nxt_r, nxt_c]
                        output[cur_r, cur_c] = strip_colour   # recolour pivot
                        colour = strip_colour                # continue with it
                        if nxt_typ == 'vstrip':
                            dy *= -1                         # mirror horizontally
                        else:  # hstrip
                            dx *= -1                         # mirror vertically
                        cur_r += dx
                        cur_c += dy
                        continue

                    # block-bounce rule: one cell **below** while descending
                    if dx == +1:
                        below_r = cur_r + 1
                        if below_r < size and coord_to_type.get((below_r, cur_c)) == 'block':
                            output[cur_r, cur_c] = colour
                            dx *= -1                       # vertical mirror
                            cur_r += dx
                            cur_c += dy
                            continue

                    # normal paint & advance
                    output[cur_r, cur_c] = colour
                    cur_r += dx
                    cur_c += dy
                    continue

                # ---------- pass through other arrow-heads ---------------
                if cell_typ == 'arrow':
                    cur_r += dx
                    cur_c += dy
                    continue

                # ---------- reflectors we LAND on ------------------------
                if cell_typ in ('vstrip', 'hstrip', 'block'):
                    # first strip already handled by pivot; landing here
                    # means second encounter → stop
                    break

                # ---------- barrier / unknown → stop --------------------
                break

        return output
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # fixed, even sizes the user asked for
        size = random.choice([14, 18, 22, 26])
        taskvars = {'grid_size': size}

        # ── 1. build the three training cases exactly as before ─────────
        train_pairs: List[GridPair] = []
        for case in ['case2', 'case3', 'case4']:
            inp = self.create_input(taskvars, {'case': case})
            train_pairs.append(
                {'input': inp, 'output': self.transform_input(inp, taskvars)}
            )

        # ── 2. build *two* test cases instead of one ───────────────────
        test_pairs: List[GridPair] = []
        for tcase in ['test', 'test2']:          # <── both branches invoked
            inp = self.create_input(taskvars, {'case': tcase})
            test_pairs.append(
                {'input': inp, 'output': self.transform_input(inp, taskvars)}
            )

        return taskvars, {'train': train_pairs, 'test': test_pairs}


