import random
from typing import Dict, Any, Tuple, List, Set
import numpy as np
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject

class Task182e5d0fGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square but have different sizes.",
            "Each grid contains n empty (0) strips (each made of three consecutive empty cells), n {color('object_color')} L-shaped objects, and n {color('cell_color')} cells, all placed on a {color('background_color')} background.",
            "The empty strips are located only at the grid edges (first or last row/column).",
            "The L shape is defined as a combination of one vertical strip and one horizontal strip connected perpendicularly. One of the two strips is mostly shorter than the other and sometimes can also be equal, with the shorter strip being either two or three cells long.",
            "The construction starts with the shorter strip, which is connected at the center of the empty (0) strip. Once the short strip is placed, the longer strip is added perpendicular to it, starting exactly from the cell adjacent to the last cell of the short strip.",
            "When adding the longer strip, there are two possible directions for extension. If it is vertical, it can extend either towards the first row or the last row; if it is horizontal, it can extend either towards the first column or the last column. The chosen direction must ensure that the strip does not intersect with any existing strips belonging to a different {color('object_color')} L-shaped object.",
            "The longer strip can be placed in three possible ways: reaching the edge, stopping one cell before the edge, or stopping two–three cells before the edge.",
            "The n {color('cell_color')} cells are placed in grid edges only.",
            "Each {color('cell_color')} cell corresponds to one {color('object_color')} L-shaped object and is placed near the end of its long strip. If the long strip is horizontal, the cell is added to either the first or last column, depending on the strip’s direction. When the strip extends toward the first column, the cell is placed in the first column, in a row either one above or one below the strip’s endpoint—usually outside the L shape’s region. If the long strip is vertical, the cell is added to either the first or last row, depending on the strip’s direction. When the strip extends toward the first row, the cell is placed in the first row, in a column either one left or one right of the strip’s endpoint—again usually outside the L shape’s region.",
            "This ensures that all L-shaped objects are placed without intersecting or overlapping each other."
        ]

        transformation_reasoning_chain = [
    "The output grid is created by copying the input grid and identifying the {color('cell_color')} cells and {color('object_color')} L-shaped objects.",
    "If a {color('object_color')} L-shaped object is 4-way connected to a {color('cell_color')} cell, a transformation occurs; otherwise, the respective object remains unchanged.",
    "In the transformation, the entire respective L-shaped object is removed except for the {color('object_color')} cell in the respective empty strip. The {color('cell_color')} cell is then moved so that it becomes connected to the remaining {color('object_color')} cell in the empty (0) strip.",
    "If the corresponding empty strip is vertical, the {color('cell_color')} cell is placed so that it is horizontally connected to the {color('object_color')} cell.",
    "If the corresponding empty strip is horizontal, the {color('cell_color')} cell is placed so that it is vertically connected to the {color('object_color')} cell.",
    "All removed {color('object_color')} cells, along with the original positions of the relocated {color('cell_color')} cells, are replaced with {color('background_color')}."
]


        super().__init__(input_reasoning_chain, transformation_reasoning_chain)


    @staticmethod
    def _adjacent(idx: int, max_idx: int) -> int:
        """Return idx ± 1 inside [0,max_idx] and different from idx."""
        if idx == 0:
            return 1
        if idx == max_idx:
            return max_idx - 1
        return idx + random.choice([-1, 1])

    @staticmethod
    def _with_buffer(cells: List[Tuple[int, int]], rows: int, cols: int) -> Set[Tuple[int, int]]:
        """Return given cells plus their 4‑neighbours (used for spacing)."""
        buf = set()
        for r, c in cells:
            buf.add((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    buf.add((nr, nc))
        return buf


    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        bg, obj, cel = taskvars['background_color'], taskvars['object_color'], taskvars['cell_color']

        # Keep original upper bound but choose requested_n first
        requested_n = random.randint(2, 4)

        force_dirs = list(gridvars.get('force_long_dirs', []))
        all_dirs = force_dirs + [d for d in ['left', 'right', 'up', 'down'] if d not in force_dirs]
        dir_vec = {'left': (0, -1), 'right': (0, 1), 'up': (-1, 0), 'down': (1, 0)}

        def get_corner(long_dir, anchor):
            ar, ac = anchor
            if long_dir == 'left':
                return 'NW' if ar == 0 else 'SW'
            elif long_dir == 'right':
                return 'NE' if ar == 0 else 'SE'
            elif long_dir == 'up':
                return 'NW' if ac == 0 else 'NE'
            elif long_dir == 'down':
                return 'SW' if ac == 0 else 'SE'
            return None

        # try odd/even sizes up to 30 as before
        for size in range(random.randint(12, 20), 31, 2):
            grid = np.full((size, size), bg, dtype=int)
            # Dynamic cap for n depending on side length
            # (small boards struggle to host 4 Ls with spacing)
            size_cap = 2 if size < 16 else (3 if size < 22 else 4)
            n = min(requested_n, size_cap)

            buffer: Set[Tuple[int, int]] = set()
            used_corners = set()

            def in_bounds(r, c):
                return 0 <= r < size and 0 <= c < size

            def halo_object_cells(cells):
                """1-cell halo only around object cells (no halo for zeros or edge-cell)."""
                for r, c in cells:
                    buffer.add((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if in_bounds(nr, nc):
                            buffer.add((nr, nc))

            def place_single(long_dir: str, force_option: int = None) -> bool:
                # Much higher attempt budget to dodge unlucky RNG
                for _ in range(10000):
                    # Anchor on edge
                    if long_dir in ('left', 'right'):
                        row_edge = random.choice([0, size - 1])
                        col_center = random.randint(2, size - 3)
                        anchor = (row_edge, col_center)
                        short_vec = (1, 0) if row_edge == 0 else (-1, 0)
                        horiz = True
                    else:
                        col_edge = random.choice([0, size - 1])
                        row_center = random.randint(2, size - 3)
                        anchor = (row_center, col_edge)
                        short_vec = (0, 1) if col_edge == 0 else (0, -1)
                        horiz = False

                    corner = get_corner(long_dir, anchor)

                    # Only enforce unique corners when we truly need four distinct ones
                    if n == 4 and corner in used_corners:
                        continue

                    if anchor in buffer:
                        continue

                    # 3-cell zero strip (exact cells reserved, no halo)
                    strip = [
                        (anchor[0] - 1, anchor[1]) if not horiz else (anchor[0], anchor[1] - 1),
                        anchor,
                        (anchor[0] + 1, anchor[1]) if not horiz else (anchor[0], anchor[1] + 1)
                    ]
                    if any(not in_bounds(r, c) or (r, c) in buffer for r, c in strip):
                        continue

                    # Short leg (object cells)
                    short_len = random.choice([2, 3])
                    short = [(anchor[0] + k * short_vec[0], anchor[1] + k * short_vec[1]) for k in range(short_len)]
                    if any(not in_bounds(r, c) or (r, c) in buffer for r, c in short):
                        continue

                    # Long leg
                    vec = dir_vec[long_dir]
                    start = short[-1]
                    max_reach = (
                        start[1] if long_dir == 'left' else
                        (size - 1 - start[1]) if long_dir == 'right' else
                        start[0] if long_dir == 'up' else
                        (size - 1 - start[0])
                    )
                    if max_reach < 1:
                        continue

                    option = force_option if force_option is not None else random.choice([1, 2, 3])
                    if option == 1:
                        length = max_reach
                    elif option == 2:
                        length = max_reach - 1
                    else:
                        length = max(1, max_reach - random.choice([2, 3]))  # clamp

                    long = [(start[0] + i * vec[0], start[1] + i * vec[1]) for i in range(1, length + 1)]
                    if any((r, c) in buffer or not in_bounds(r, c) for r, c in long):
                        continue

                    # Cell-color at edge (reserve exact cell, no halo)
                    end_r, end_c = long[-1]
                    if long_dir == 'left':
                        cell = (self._adjacent(end_r, size - 1), 0)
                    elif long_dir == 'right':
                        cell = (self._adjacent(end_r, size - 1), size - 1)
                    elif long_dir == 'up':
                        cell = (0, self._adjacent(end_c, size - 1))
                    else:
                        cell = (size - 1, self._adjacent(end_c, size - 1))

                    if not in_bounds(*cell):
                        continue
                    if cell in buffer or cell in strip or cell in short or cell in long:
                        continue

                    # Commit placement
                    # zeros: exact reserve only
                    for r, c in strip:
                        grid[r, c] = 0

                    # object legs: draw and halo them
                    grid[anchor] = obj
                    for r, c in short[1:]:
                        grid[r, c] = obj
                    for r, c in long:
                        grid[r, c] = obj

                    # edge cell: exact reserve only
                    grid[cell] = cel

                    # Update buffers:
                    #  - object cells get a 1-cell halo to prevent collisions
                    #  - zero strip and the edge cell are reserved exactly (no halo)
                    halo_object_cells(short + long)  # includes anchor because short starts at anchor
                    buffer.update(strip)            # exact
                    buffer.add(cell)                # exact

                    used_corners.add(corner)
                    return True
                return False

            success = True
            # Still place one "reach edge" first, but with relaxed spacing this is safer
            if not place_single(all_dirs[0], force_option=1):
                success = False
            else:
                for i in range(1, n):
                    if not place_single(all_dirs[i % len(all_dirs)]):
                        success = False
                        break

            if success:
                return grid

        # If we truly can’t place with all relaxations above, raise as before.
        raise RuntimeError("Unable to create grid up to 30×30; reduce n or constraints")

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
           
        bg, obj, cel = taskvars['background_color'], taskvars['object_color'], taskvars['cell_color']
        out = grid.copy()
        rows, cols = out.shape
        objects = find_connected_objects(out, diagonal_connectivity=False, background=bg, monochromatic=True)
        lookup = {(r, c): o for o in objects for r, c in o.coords}

        for cr, cc in zip(*np.where(out == cel)):
            touching = [lookup.get(nb) for nb in [(cr-1, cc), (cr+1, cc), (cr, cc-1), (cr, cc+1)] if nb in lookup]
            touching = [o for o in touching if o]
            if not touching:
                continue
            
            obj_comp = touching[0]

            # Find anchor (object cell between two empty cells)
            anchor = None
            zero_positions = []
            for r, c in obj_comp.coords:
                neighbors = [(r+dr, c+dc) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                            if 0 <= r+dr < rows and 0 <= c+dc < cols]
                zeros = [(nr, nc) for nr, nc in neighbors if out[nr, nc] == 0]
                if len(zeros) == 2:
                    anchor = (r, c)
                    zero_positions = zeros
                    break
            if not anchor:
                continue

            # Remove all other object cells
            for r, c in obj_comp.coords:
                if (r, c) != anchor:
                    out[r, c] = bg
            
            # Remove old cell_color
            out[cr, cc] = bg

            # Determine strip orientation
            zr1, zc1 = zero_positions[0]
            zr2, zc2 = zero_positions[1]
            horizontal_strip = (zr1 == zr2)  # both zeros share same row

            # Place cell_color perpendicular to strip
            ar, ac = anchor
            if horizontal_strip:
                # strip horizontal → cell_color above or below
                for dr in [-1, 1]:
                    nr = ar + dr
                    if 0 <= nr < rows and out[nr, ac] in (bg, 0):
                        out[nr, ac] = cel
                        break
            else:
                # strip vertical → cell_color left or right
                for dc in [-1, 1]:
                    nc = ac + dc
                    if 0 <= nc < cols and out[ar, nc] in (bg, 0):
                        out[ar, nc] = cel
                        break

        return out


    # ------------------------------------------------------------------
    # Train / test generation
    # ------------------------------------------------------------------
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        obj_col, cel_col, bg_col = random.sample(range(1, 10), 3)
        taskvars = {'object_color': obj_col, 'cell_color': cel_col, 'background_color': bg_col}

        dirs = ['left', 'right', 'up', 'down']
        random.shuffle(dirs)
        train: List[GridPair] = []
        for d in dirs:
            inp = self.create_input(taskvars, {'force_long_dirs': [d]})
            train.append({'input': inp, 'output': self.transform_input(inp, taskvars)})

        test_inp = self.create_input(taskvars, {})
        test_pair = {'input': test_inp, 'output': self.transform_input(test_inp, taskvars)}
        return taskvars, {'train': train, 'test': [test_pair]}


