import numpy as np
import random
from typing import Dict, Any, Set, Tuple, List

from arc_task_generator import ARCTaskGenerator, MatrixPair, TrainTestData
from input_library import Contiguity, create_object, enforce_object_height, enforce_object_width
from transformation_library import GridObject, find_connected_objects

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
            "It is moving object moves in the direction which causes it to touch the static object.",
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
            # sprite = create_sprite(
            #    height=mh,
            #    width=mw,
            #    color_palette=color_moving,
            #    enforce_height=(direction in ["up", "down"]),
            #    enforce_width=(direction in ["left", "right"]),
            #    contiguity=Contiguity.EIGHT
            #)
            #sprite = retry(
            #    lambda: create_sprite(height=mh, width=mw, color_palette=color_moving, contiguity=Contiguity.EIGHT),
            #    (lambda x: np.all(np.any(x != 0, axis=0))) if direction in ["up", "down"] else
            #    (lambda x: np.all(np.any(x != 0, axis=1)))
            #)
            sprite = (enforce_object_height if direction in ["up", "down"] else enforce_object_width)(
                lambda: create_object(height=mh, width=mw, color_palette=color_moving, contiguity=Contiguity.EIGHT)
            )


            # sprite = create_sprite_old(mh, mw, direction, color_moving)

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

    def transform_input(self, matrix: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Slide the moving shape towards the static shape until they touch.
        The direction is determined by relative positions of the objects.
        """
        color_static = taskvars["static_object"]
        color_moving = taskvars["moving_object"]
        
        # Create GridObjects from the matrix
        objects = find_connected_objects(matrix, diagonal_connectivity=True)
        static_obj = objects.with_color(color_static)[0]
        moving_obj = objects.with_color(color_moving)[0]
        
        def direction_to_offset(direction: str) -> Tuple[int, int]:
            """Convert direction string to (dr, dc) offset."""
            return {
                "up": (-1, 0),
                "down": (1, 0),
                "left": (0, -1),
                "right": (0, 1)
            }[direction]

        def compute_movement_direction(static: GridObject, moving: GridObject) -> str:
            """Determine movement direction based on objects' bounding boxes."""
            s_box = static.bounding_box
            m_box = moving.bounding_box
            
            dr = (s_box[0].start + s_box[0].stop) / 2 - (m_box[0].start + m_box[0].stop) / 2
            dc = (s_box[1].start + s_box[1].stop) / 2 - (m_box[1].start + m_box[1].stop) / 2
            
            return "down" if abs(dr) > abs(dc) and dr > 0 else \
                "up" if abs(dr) > abs(dc) else \
                "right" if dc > 0 else "left"

        # Determine direction based on relative positions
        direction = compute_movement_direction(static_obj, moving_obj)
        dr, dc = direction_to_offset(direction)
        
        # Create output grid and initialize coordinates
        out = matrix.copy()
        moving_coords = moving_obj.coords
        
        # Shift until invalid or adjacency
        while True:
            next_coords = {(r+dr, c+dc) for (r, c) in moving_coords}
            
            # Check bounds
            if not all(0 <= r < out.shape[0] and 0 <= c < out.shape[1] 
                    for r, c in next_coords):
                break
            
            # Create temporary GridObject for next position
            next_obj = GridObject.from_grid(matrix, next_coords)
            
            # Check overlap or adjacency
            if next_obj.coords & static_obj.coords:  # overlap
                break
            if next_obj.is_adjacent_to(static_obj):  # adjacency
                moving_coords = next_coords
                break
                
            # Perform shift
            moving_coords = next_coords
        
        # Update output grid
        out[out == color_moving] = 0
        for r, c in moving_coords:
            out[r, c] = color_moving
        
        return out

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
