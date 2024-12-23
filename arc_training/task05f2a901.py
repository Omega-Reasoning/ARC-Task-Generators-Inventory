import numpy as np
import random
from typing import Dict, Any, Tuple, List

from arc_task_generator import ARCTaskGenerator, MatrixPair, TrainTestData

class ARCTask05f2a901Generator(ARCTaskGenerator):
    """
    Generator for 'slide object until it touches another object' tasks,
    allowing the moving object to have a more flexible shape within its
    bounding box. Specifically:
      - For left/right slides, the object's bounding-box height is used,
        meaning each row has at least one colored cell.
      - For up/down slides, the object's bounding-box width is used,
        meaning each column has at least one colored cell.
    The final position of the object actually touches (becomes adjacent to)
    the static block.
    """

    def __init__(self):
        observation_chain = [
            "There is a static rectangular {color('static_object')} {vars['rows_static_object']}x{vars['columns_static_object']} block in each input matrix.",
            "In each input matrix, there is one {color('moving_object')} object which can have different shapes.",
            "Both objects seem to share either a row or a column."
        ]
        reasoning_chain = [
            "The output matrix has the same shape as the input matrix.",
            "The {color('static_object')} object can be directly copied into the output matrix as it remains static.",
            "The {color('moving_object')} object moves either up, down, left or right towards the static object.",
            "It moving object moves in the direction which causes it to touch the static object.",
            "It moves exactly until it touches the static object."
        ]
        super().__init__(observation_chain, reasoning_chain)

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        1) Pick colors for static vs. moving object.
        2) Generate at least 3 distinct directions in the training set, plus one extra.
        3) Generate a single test example with a random direction.
        """

        # Distinct colors for static/moving from [1..9]
        possible_colors = list(range(1, 10))
        color_static = random.choice(possible_colors)
        color_moving = random.choice([c for c in possible_colors if c != color_static])

        # Static block size
        rs = random.randint(2, 4)
        cs = random.randint(2, 4)

        # Directions for training
        all_dirs = ["up", "down", "left", "right"]
        distinct_dirs = random.sample(all_dirs, k=3)
        fourth_dir = random.choice(all_dirs)
        train_dirs = distinct_dirs + [fourth_dir]

        # One test direction
        test_dir = random.choice(all_dirs)

        taskvars = {
            "static_object": color_static,
            "moving_object": color_moving,
            "rows_static_object": rs,
            "columns_static_object": cs,
        }

        # Build train data
        train_data: List[MatrixPair] = []
        for d in train_dirs:
            matrixvars = {"direction": d}
            inp = self.create_input(taskvars, matrixvars)
            out = self.transform_input(inp, {**taskvars, "direction": d})
            train_data.append(MatrixPair(input=inp, output=out))

        # Build test data
        test_data: List[MatrixPair] = []
        matrixvars_test = {"direction": test_dir}
        inp_test = self.create_input(taskvars, matrixvars_test)
        out_test = self.transform_input(inp_test, {**taskvars, "direction": test_dir})
        test_data.append(MatrixPair(input=inp_test, output=out_test))

        return taskvars, TrainTestData(train=train_data, test=test_data)

    def create_input(self,
                     taskvars: Dict[str, Any],
                     matrixvars: Dict[str, Any]) -> np.ndarray:
        """
        Create a grid [10..20]x[10..20]. Place:
          - A static rectangle (rs x cs) in color_static.
          - A *flexible* moving shape of bounding box size 2..4 x 2..4 in color_moving,
            but with some cells possibly empty. The rule is:
              - If direction is left/right, each row in bounding box has at least one colored cell.
              - If direction is up/down, each column in bounding box has at least one colored cell.
        Place the moving shape so that it has at least 1 cell of free space to slide toward the static block.
        """
        color_static = taskvars["static_object"]
        color_moving = taskvars["moving_object"]
        rs = taskvars["rows_static_object"]
        cs = taskvars["columns_static_object"]
        direction = matrixvars["direction"]

        n = random.randint(10, 20)
        m = random.randint(10, 20)
        grid = np.zeros((n, m), dtype=int)

        # Place static block in a random location (with some margin)
        for _attempt in range(50):
            grid[:, :] = 0

            # If the grid is too small, fallback
            if n - rs - 2 <= 0 or m - cs - 2 <= 0:
                rS, cS = 0, 0
            else:
                rS = random.randint(1, n - rs - 1)
                cS = random.randint(1, m - cs - 1)

            # Fill the static block
            grid[rS:rS+rs, cS:cS+cs] = color_static

            # Decide bounding box size for moving shape
            mh = random.randint(2, 4)
            mw = random.randint(2, 4)
            sprite = self._make_arbitrary_sprite(mh, mw, direction, color_moving)

            # Now place sprite according to direction constraints
            if direction == "left":
                # same row => pick row in [rS..rS+rs-1]
                row_candidates = list(range(rS, rS + rs))
                if not row_candidates:
                    continue
                row = random.choice(row_candidates)
                # must place it to the right of static => c >= cS+cs+1
                # ensuring c+mw <= m
                valid_c = []
                for cM in range(cS + cs + 1, m - mw + 1):
                    sub = grid[row:row+mh, cM:cM+mw] if row+mh<=n else None
                    if sub is not None and np.all(sub == 0):
                        valid_c.append(cM)
                if not valid_c:
                    continue
                chosen_c = random.choice(valid_c)
                grid[row:row+mh, chosen_c:chosen_c+mw] = self._overlay(grid[row:row+mh, chosen_c:chosen_c+mw], sprite)
                return grid

            elif direction == "right":
                # same row => pick row in [rS..rS+rs-1], place block to the left
                row_candidates = list(range(rS, rS + rs))
                if not row_candidates:
                    continue
                row = random.choice(row_candidates)
                valid_c = []
                # c+mw <= cS - 1
                for cM in range(0, cS - mw):
                    sub = grid[row:row+mh, cM:cM+mw] if row+mh<=n else None
                    if sub is not None and np.all(sub == 0):
                        valid_c.append(cM)
                if not valid_c:
                    continue
                chosen_c = random.choice(valid_c)
                grid[row:row+mh, chosen_c:chosen_c+mw] = self._overlay(grid[row:row+mh, chosen_c:chosen_c+mw], sprite)
                return grid

            elif direction == "up":
                # same column => pick col in [cS..cS+cs-1], place block below
                col_candidates = list(range(cS, cS + cs))
                if not col_candidates:
                    continue
                col = random.choice(col_candidates)
                valid_r = []
                # must place it so r >= rS+rs+1
                for rM in range(rS + rs + 1, n - mh + 1):
                    sub = grid[rM:rM+mh, col:col+mw] if col+mw<=m else None
                    if sub is not None and np.all(sub == 0):
                        valid_r.append(rM)
                if not valid_r:
                    continue
                chosen_r = random.choice(valid_r)
                # overlay
                grid[chosen_r:chosen_r+mh, col:col+mw] = self._overlay(grid[chosen_r:chosen_r+mh, col:col+mw], sprite)
                return grid

            else:
                # direction == "down"
                col_candidates = list(range(cS, cS + cs))
                if not col_candidates:
                    continue
                col = random.choice(col_candidates)
                valid_r = []
                # place above => r+mh <= rS-1
                for rM in range(0, rS - mh):
                    sub = grid[rM:rM+mh, col:col+mw] if col+mw<=m else None
                    if sub is not None and np.all(sub == 0):
                        valid_r.append(rM)
                if not valid_r:
                    continue
                chosen_r = random.choice(valid_r)
                grid[chosen_r:chosen_r+mh, col:col+mw] = self._overlay(grid[chosen_r:chosen_r+mh, col:col+mw], sprite)
                return grid

        # fallback: if no valid placement found, return the partially filled grid
        return grid

    def transform_input(self,
                        matrix: np.ndarray,
                        taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Slide the shape of color_moving in the chosen direction until:
         - If next shift is out of bounds or overlaps static_object, do NOT perform it => STOP.
         - If next shift causes adjacency (touching) with static_object, DO perform it => STOP.
        That yields a final position where the blocks actually contact.
        """
        direction = taskvars["direction"]
        color_static = taskvars["static_object"]
        color_moving = taskvars["moving_object"]

        out = matrix.copy()

        # gather coords
        static_coords = set(zip(*np.where(out == color_static)))
        moving_coords = set(zip(*np.where(out == color_moving)))

        def overlap(coordsA, coordsB) -> bool:
            return not coordsA.isdisjoint(coordsB)

        def adjacent(coordsA, coordsB) -> bool:
            # True if any pair from coordsA and coordsB has Manhattan-distance=1
            for (rA, cA) in coordsA:
                for (rB, cB) in coordsB:
                    if abs(rA - rB) + abs(cA - cB) == 1:
                        return True
            return False

        # direction offsets
        dr, dc = 0, 0
        if direction == "up":
            dr = -1
        elif direction == "down":
            dr = 1
        elif direction == "left":
            dc = -1
        elif direction == "right":
            dc = 1

        # shift until invalid or adjacency
        while True:
            next_coords = {(r+dr, c+dc) for (r, c) in moving_coords}
            # bounds
            if any(nr<0 or nr>=out.shape[0] or nc<0 or nc>=out.shape[1] for nr, nc in next_coords):
                break
            # overlap => stop
            if overlap(next_coords, static_coords):
                break
            # adjacency => do shift then stop
            if adjacent(next_coords, static_coords):
                moving_coords = next_coords
                break

            # otherwise, we can shift
            moving_coords = next_coords

        # rebuild output
        out[out == color_moving] = 0
        for (r, c) in moving_coords:
            out[r, c] = color_moving

        return out

    def _make_arbitrary_sprite(self, mh: int, mw: int, direction: str, color: int) -> np.ndarray:
        """
        Builds a shape in a bounding box of size (mh x mw) while respecting:
        - If direction is left or right, each row has at least one colored cell.
        - If direction is up or down, each column has at least one colored cell.
        Then ensures the shape is contiguous in an 8-way sense by keeping only the largest connected component.
        """

        import random
        import numpy as np
        from scipy.ndimage import label

        sprite = np.zeros((mh, mw), dtype=int)

        # Assign cells to color within each row/column
        if direction in ["left", "right"]:
            # Each row has at least one colored cell
            for r in range(mh):
                num_cols = random.randint(1, mw)
                chosen_cols = random.sample(range(mw), num_cols)
                for c in chosen_cols:
                    sprite[r, c] = color
        else:
            # direction in ["up", "down"]
            # Each column has at least one colored cell
            for c in range(mw):
                num_rows = random.randint(1, mh)
                chosen_rows = random.sample(range(mh), num_rows)
                for r in chosen_rows:
                    sprite[r, c] = color

        # Enforce 8-way contiguity: keep only the largest connected component
        structure = np.ones((3, 3), dtype=int)  # 8-way connectivity
        labeled, n_obj = label(sprite == color, structure=structure)
        if n_obj > 1:
            sizes = [(labeled == i).sum() for i in range(1, n_obj + 1)]
            largest_idx = 1 + np.argmax(sizes)
            sprite[(labeled != largest_idx) & (sprite == color)] = 0

        return sprite

    def _overlay(self, region: np.ndarray, sprite: np.ndarray) -> np.ndarray:
        """
        Overlay the sprite (same shape as region) onto region, 
        placing the non-zero sprite cells. (If region is zero, no conflict.)
        """
        # We don't do any blending, just put sprite color wherever sprite is non-zero.
        # Return a new array for cleanliness.
        out = region.copy()
        mask = (sprite != 0)
        out[mask] = sprite[mask]
        return out
