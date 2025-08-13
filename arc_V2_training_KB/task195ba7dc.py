from typing import Dict, Any, Tuple, List
import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, Contiguity
from transformation_library import GridObject, GridObjects, find_connected_objects

class Task195ba7dcTask(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are size {vars['rows']}×{vars['cols']}.",
            "The grid is split along the vertical axis by filling the middle column with {color('column_color')}, creating left and right halves.",
            "The left and right halves contain several objects of varying shapes and sizes made of {color('object_color')}; all other cells are empty (0).",
            "The shapes and sizes of objects vary across examples.",
        ]
        transformation_reasoning_chain = [
            "The output grid is of size {vars['rows']}×(({vars['cols']}-1)//2).",
            "It is constructed by identifying the {color('column_color')} column and the two halves containing {color('object_color')} objects.",
            "Copy the right half and paste it over the left half, allowing overlapping cells — this is intentional.",
            "Cells that are empty (0) in both halves remain empty (0) in the output.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)


    def _place_objects_in_half(self, grid: np.ndarray, rows: int, cols: int, half: str,
                               object_color: int,
                               min_objects: int = 2,
                               max_objects: int = 5) -> None:
        """Fill a half (left/right) of the grid with several non-overlapping objects."""
        assert half in {"left", "right"}
        mid_col = cols // 2
        if half == "left":
            c0, c1 = 0, mid_col  # [0, mid_col)
        else:
            c0, c1 = mid_col + 1, cols  # (mid_col, cols]

        # Occupancy mask to avoid overlaps within the half
        occ = np.zeros_like(grid, dtype=bool)

        target_n = random.randint(min_objects, max_objects)

        def try_place_once():
            """Attempt to place a single random object inside the chosen half without overlap."""
            # Random object size, encourage variety and keep within bounds
            max_h = max(1, min(rows, 6 + random.randint(-2, 4)))
            max_w = max(1, min((c1 - c0), 6 + random.randint(-2, 4)))
            h = random.randint(1, max(1, max_h))
            w = random.randint(1, max(1, max_w))
            # Ensure object fits in half
            h = min(h, rows)
            w = min(w, (c1 - c0))
            # Random contiguity preference for shape variety
            cont = random.choice([Contiguity.EIGHT, Contiguity.FOUR])
            obj = create_object(h, w, color_palette=object_color, contiguity=cont, background=0)

            # Compute possible placement window (top-left inclusive)
            max_r = rows - obj.shape[0]
            max_c = (c1 - c0) - obj.shape[1]
            if max_r < 0 or max_c < 0:
                return False
            r_off = random.randint(0, max_r)
            c_off = random.randint(0, max_c)
            r0 = r_off
            c_start = c0 + c_off

            # Check overlap within the half and avoid the divider column
            sub = grid[r0:r0 + obj.shape[0], c_start:c_start + obj.shape[1]]
            sub_occ = occ[r0:r0 + obj.shape[0], c_start:c_start + obj.shape[1]]
            if np.any((obj != 0) & sub_occ):
                return False

            # Place
            mask = obj != 0
            sub[mask] = object_color
            occ[r0:r0 + obj.shape[0], c_start:c_start + obj.shape[1]][mask] = True
            return True

        placed = 0
        attempts = 0
        while placed < target_n and attempts < 200:
            if try_place_once():
                placed += 1
            attempts += 1

        # Ensure at least two cells exist in this half
        if not np.any(grid[:, c0:c1] != 0):
            # Fallback: draw a small 1x1 or 1x2 object
            r = random.randrange(rows)
            c = random.randrange(c0, c1)
            grid[r, c] = object_color
            if c + 1 < c1 and random.random() < 0.5:
                grid[r, c + 1] = object_color

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        object_color = taskvars['object_color']
        column_color = taskvars['column_color']

        grid = np.zeros((rows, cols), dtype=int)
        mid_col = cols // 2
        grid[:, mid_col] = column_color  # divider

        # Populate left and right halves with several objects
        self._place_objects_in_half(grid, rows, cols, 'left', object_color)
        self._place_objects_in_half(grid, rows, cols, 'right', object_color)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        object_color = taskvars['object_color']
        column_color = taskvars['column_color']

        mid_col = cols // 2
        assert np.all(grid[:, mid_col] == column_color), "Middle column must be the split marker"

        left = grid[:, :mid_col]
        right = grid[:, mid_col + 1:]
        assert left.shape[1] == right.shape[1] == (cols - 1) // 2

        # Overlay rule: if either half has a non-zero (object) cell, output has object_color.
        # (Both halves use the same object_color by construction.)
        out = np.where((left != 0) | (right != 0), object_color, 0)
        return out

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Global task variables (shared across all pairs)
        rows = random.randint(5, 30)
        # Ensure odd cols between 5 and 30
        odd_choices = [c for c in range(5, 31) if c % 2 == 1]
        cols = random.choice(odd_choices)

        # Colors 1..9 distinct
        object_color = random.randint(1, 9)
        column_color = random.randint(1, 9)
        while column_color == object_color:
            column_color = random.randint(1, 9)

        taskvars: Dict[str, Any] = {
            'rows': rows,
            'cols': cols,
            'object_color': object_color,
            'column_color': column_color,
        }

        # Number of examples
        n_train = random.randint(3, 5)
        n_test = 1

        def gen_pair() -> GridPair:
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            # Sanity checks
            assert inp.shape == (rows, cols)
            assert out.shape == (rows, (cols - 1) // 2)
            return {'input': inp, 'output': out}

        train_pairs = [gen_pair() for _ in range(n_train)]
        test_pairs = [gen_pair() for _ in range(n_test)]

        return taskvars, {'train': train_pairs, 'test': test_pairs}



