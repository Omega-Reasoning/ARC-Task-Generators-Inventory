from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject
from Framework.input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List


class Taska48eeaf7Generator(ARCTaskGenerator):
   
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of different sizes.",
            "Each input grid contains exactly one {color('object_color')} 2×2 block and several {color('cell_color')} cells, with all remaining cells being empty.",
            "The {color('object_color')} 2×2 block can be placed anywhere in the grid except at the edges, but the {color('cell_color')} cells must be placed in specific allowed locations.",
            "The {color('cell_color')} cells may appear in specific rows, columns, or along diagonal paths.",
            "Each {color('cell_color')} cell must be placed in a row or column occupied by the 2×2 block. At most two {color('cell_color')} cells may appear in the same row or column, and two are allowed only if one appears before and the other after the block (or one on the left and the other on the right side).",
            "The {color('cell_color')} cells may also be added along a diagonal path that starts from a cell diagonally adjacent to the 2×2 block and extends strictly in the same diagonal direction.",
            "Each diagonal path may start from any of the four diagonally adjacent cells to the 2×2 block and extends outward until the boundary is reached. All cells on this diagonal path are diagonally connected in one consistent direction."
            
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the {color('object_color')} 2×2 block and all {color('cell_color')} cells.",
            "Each {color('cell_color')} cell is moved closer to the {color('object_color')} 2×2 block.",
            "If a {color('cell_color')} cell shares a row with the block, it moves horizontally (left or right) along that row toward the block.",
            "If it shares a column, it moves vertically (up or down) along that column toward the block.",
            "If it lies on the allowed diagonal path, it moves along that diagonal toward the block.",
            "Each {color('cell_color')} cell may move by one or more cells, but always directly in the direction of the block (row, column, or diagonal), without changing its direction."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create a single input grid according to the input reasoning chain."""
        # Grid size between 5 and 20 (keeps outputs readable and well inside ARC limit 30)
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        # Use task-level fixed colors when provided, otherwise pick and return them via gridvars
        if 'object_color' in taskvars and 'cell_color' in taskvars:
            object_color = taskvars['object_color']
            cell_color = taskvars['cell_color']
        else:
            object_color = random.randint(1, 9)
            cell_color = random.randint(1, 9)
            while cell_color == object_color:
                cell_color = random.randint(1, 9)

        # Place 2x2 object not on the edges: top-left row in [1, rows-3], col in [1, cols-3]
        br = random.randint(1, rows - 3)
        bc = random.randint(1, cols - 3)

        grid = np.zeros((rows, cols), dtype=int)
        # Place 2x2 block
        grid[br:br+2, bc:bc+2] = object_color

        # Prepare allowed positions for cell_color cells
        allowed_positions = []  # list of (r,c,reason)

        # Same-row allowed positions for the two rows occupied by the block
        for r in (br, br + 1):
            # left side (must not touch the block orthogonally) -> cols 0..bc-2
            for c in range(0, max(0, bc - 1)):
                if c <= bc - 2:
                    allowed_positions.append((r, c, 'row_left'))
            # right side -> cols bc+3..cols-1
            for c in range(bc + 3, cols):
                allowed_positions.append((r, c, 'row_right'))

        # Same-column allowed positions for the two columns occupied by the block
        for c in (bc, bc + 1):
            # above
            for r in range(0, max(0, br - 1)):
                if r <= br - 2:
                    allowed_positions.append((r, c, 'col_top'))
            # below
            for r in range(br + 3, rows):
                allowed_positions.append((r, c, 'col_bottom'))

        # ---------- FIXED DIAGONAL LOGIC STARTS HERE ----------
        # Diagonal paths starting from cells diagonally adjacent to the 2×2 block.
        # Each path is a straight diagonal ray extending in a single direction.

        # Top-left diagonal: start at (br-1, bc-1) and go up-left
        r, c = br - 1, bc - 1
        while r >= 0 and c >= 0:
            allowed_positions.append((r, c, 'diag_tl'))
            r -= 1
            c -= 1

        # Top-right diagonal: start at (br-1, bc+2) and go up-right
        r, c = br - 1, bc + 2
        while r >= 0 and c < cols:
            allowed_positions.append((r, c, 'diag_tr'))
            r -= 1
            c += 1

        # Bottom-left diagonal: start at (br+2, bc-1) and go down-left
        r, c = br + 2, bc - 1
        while r < rows and c >= 0:
            allowed_positions.append((r, c, 'diag_bl'))
            r += 1
            c -= 1

        # Bottom-right diagonal: start at (br+2, bc+2) and go down-right
        r, c = br + 2, bc + 2
        while r < rows and c < cols:
            allowed_positions.append((r, c, 'diag_br'))
            r += 1
            c += 1
        # ---------- FIXED DIAGONAL LOGIC ENDS HERE ----------

        # Filter duplicates and keep unique positions (preserve reason is not critical)
        seen = set()
        deduped = []
        for r, c, reason in allowed_positions:
            if (r, c) not in seen:
                seen.add((r, c))
                deduped.append((r, c, reason))
        allowed_positions = deduped

        # Choose number of cell_color cells between 2 and 5
        n_cells = random.randint(2, 5)

        placed = []  # list of (r,c)

        # To ensure the transform results are deterministic and non-overlapping, compute targets
        def compute_target(r, c):
            # If in same row as block
            if r == br or r == br + 1:
                if c < bc:
                    return (r, bc - 1)
                else:
                    return (r, bc + 2)
            # If in same column
            if c == bc or c == bc + 1:
                if r < br:
                    return (br - 1, c)
                else:
                    return (br + 2, c)
            # Diagonals: determine which diagonal we are on by relation to diagonally adjacent cells
            # Top-left diagonal path: towards (br-1, bc-1)
            if abs(r - (br - 1)) == abs(c - (bc - 1)) and r <= br - 1 and c <= bc - 1:
                return (br - 1, bc - 1)
            # Top-right
            if abs(r - (br - 1)) == abs(c - (bc + 2)) and r <= br - 1 and c >= bc + 2:
                return (br - 1, bc + 2)
            # Bottom-left
            if abs(r - (br + 2)) == abs(c - (bc - 1)) and r >= br + 2 and c <= bc - 1:
                return (br + 2, bc - 1)
            # Bottom-right
            if abs(r - (br + 2)) == abs(c - (bc + 2)) and r >= br + 2 and c >= bc + 2:
                return (br + 2, bc + 2)
            # Fallback: no movement
            return (r, c)

        # Build candidate list from allowed_positions and shuffle
        candidates = allowed_positions.copy()
        random.shuffle(candidates)

        attempts = 0
        while len(placed) < n_cells and attempts < 500 and candidates:
            attempts += 1
            r, c, reason = candidates.pop()
            # Skip if position already used or is part of the block
            if (r, c) in placed:
                continue
            if grid[r, c] != 0:
                continue

            # Compute target and skip if target is inside the block or already used by another target
            tgt = compute_target(r, c)
            # if target is inside block coordinates, skip
            if br <= tgt[0] <= br + 1 and bc <= tgt[1] <= bc + 1:
                continue
            if any(compute_target(pr, pc) == tgt for pr, pc in placed):
                continue

            # Enforce at most two per row/column for rows/cols of block and they must be on different sides
            def violates_row_col_constraints(r0, c0):
                # Count existing in same row if row is one of block rows
                if r0 in (br, br + 1):
                    existing = [pc for pr, pc in placed if pr == r0]
                    if len(existing) >= 2:
                        return True
                    if len(existing) == 1:
                        # ensure one is left and one is right
                        existing_side = existing[0] < bc
                        this_side = c0 < bc
                        if existing_side == this_side:
                            return True
                if c0 in (bc, bc + 1):
                    existing = [pr for pr, pc in placed if pc == c0]
                    if len(existing) >= 2:
                        return True
                    if len(existing) == 1:
                        existing_side = existing[0] < br
                        this_side = r0 < br
                        if existing_side == this_side:
                            return True
                return False

            if violates_row_col_constraints(r, c):
                continue

            # Place the cell
            grid[r, c] = cell_color
            placed.append((r, c))

        # If we failed to place enough, retry the whole grid generation once (keeps variety)
        if len(placed) < 2:
            return self.create_input(taskvars, gridvars)

        # Save variables needed by transform (but task-level colors should be used when present)
        gridvars['object_color'] = object_color
        gridvars['cell_color'] = cell_color
        gridvars['block_top_left'] = (br, bc)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input grid into output grid by moving cell_color cells toward the 2x2 block."""
        out = grid.copy()

        # Identify the 2x2 block by searching for a 2x2 monochromatic object of size 4
        rows, cols = out.shape
        object_obj = None
        for r in range(rows - 1):
            for c in range(cols - 1):
                sub = out[r:r+2, c:c+2]
                vals = set(np.unique(sub))
                vals.discard(0)
                if len(vals) == 1 and np.count_nonzero(sub) == 4:
                    object_obj = (r, c, list(vals)[0])
                    break
            if object_obj:
                break

        if object_obj is None:
            # Nothing to do
            return out

        br, bc, object_color = object_obj

        # Find all cell_color cells (single-cell objects)
        cell_positions = []
        for r in range(rows):
            for c in range(cols):
                if out[r, c] != 0:
                    if not (br <= r <= br + 1 and bc <= c <= bc + 1):
                        cell_positions.append((r, c, out[r, c]))

        # To avoid overwriting moved cells, build a new output and paste block first
        result = np.zeros_like(out)
        # Paste the block
        result[br:br+2, bc:bc+2] = object_color

        occupied_targets = set()

        def target_for(r, c):
            # Same row
            if r == br or r == br + 1:
                if c < bc:
                    return (r, bc - 1)
                else:
                    return (r, bc + 2)
            # Same column
            if c == bc or c == bc + 1:
                if r < br:
                    return (br - 1, c)
                else:
                    return (br + 2, c)
            # Diagonals by quadrant
            if r < br and c < bc:
                return (br - 1, bc - 1)
            if r < br and c > bc + 1:
                return (br - 1, bc + 2)
            if r > br + 1 and c < bc:
                return (br + 2, bc - 1)
            if r > br + 1 and c > bc + 1:
                return (br + 2, bc + 2)
            return (r, c)

        # Move cells to their targets
        for r, c, col in cell_positions:
            tgt_r, tgt_c = target_for(r, c)
            # If target is out of bounds, keep in place
            if not (0 <= tgt_r < rows and 0 <= tgt_c < cols):
                tgt_r, tgt_c = r, c
            # If target already occupied by block or another moved cell, step back along path to last free cell
            if result[tgt_r, tgt_c] == 0 and not (br <= tgt_r <= br + 1 and bc <= tgt_c <= bc + 1):
                result[tgt_r, tgt_c] = col
                occupied_targets.add((tgt_r, tgt_c))
            else:
                # find nearest free cell along line from original towards target
                cur_r, cur_c = r, c
                step_r = 0 if tgt_r == r else (1 if tgt_r > r else -1)
                step_c = 0 if tgt_c == c else (1 if tgt_c > c else -1)
                last_free = (r, c)
                while (cur_r, cur_c) != (tgt_r, tgt_c):
                    cur_r += step_r
                    cur_c += step_c
                    if result[cur_r, cur_c] == 0 and not (br <= cur_r <= br + 1 and bc <= cur_c <= bc + 1):
                        last_free = (cur_r, cur_c)
                    else:
                        break
                result[last_free[0], last_free[1]] = col
                occupied_targets.add(last_free)

        return result

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Generate 3-5 training examples and 1 test example."""
        taskvars: Dict[str, Any] = {}

        num_train = random.randint(3, 5)
        train = []
        for _ in range(num_train):
            gridvars = {}
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)
            # store the colours in taskvars for template instantiation
            taskvars.setdefault('object_color', gridvars.get('object_color', None))
            taskvars.setdefault('cell_color', gridvars.get('cell_color', None))
            train.append({'input': inp, 'output': out})

        # Single test
        test_gridvars = {}
        test_in = self.create_input(taskvars, test_gridvars)
        test_out = self.transform_input(test_in, taskvars)
        taskvars.setdefault('object_color', test_gridvars.get('object_color', taskvars.get('object_color')))
        taskvars.setdefault('cell_color', test_gridvars.get('cell_color', taskvars.get('cell_color')))

        return taskvars, {
            'train': train,
            'test': [{'input': test_in, 'output': test_out}]
        }


