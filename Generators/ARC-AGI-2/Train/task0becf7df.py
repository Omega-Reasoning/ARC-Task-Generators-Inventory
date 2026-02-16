from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from scipy.ndimage import label


class Task0becf7dfGenerator(ARCTaskGenerator):
   
    def __init__(self) -> None:
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each grid contains a 2×2 block of four differently colored cells placed in the top-left corner; the colors of these cells vary across examples.",
            "Additionally, there is a multi-colored object constructed using all four colors from the 2×2 block.",
            "The multi-colored object is positioned near the center of the grid, fully surrounded by at least one layer of empty (0) cells.",
            "All remaining cells are 0."
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the colors in the top-left 2×2 block of the input grid. Based on these colors, the colors of the multi-colored object positioned near the center are changed.",
            "The color used in the top-left cell of the 2×2 block is changed to the color of the top-right cell of the block within the multi-colored object.",
            "The color used in the bottom-left cell of the 2×2 block is changed to the color of the bottom-right cell of the block within the multi-colored object.",
            " The 2×2 block itself remains unchanged."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # ───────────────────── MAIN FUNCTION 1: CREATE INPUT ─────────────────────────────── #

    def create_input(self, taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        
        def _is_connected(grid: np.ndarray, background: int = 0) -> bool:
            mask = grid != background
            if not np.any(mask):
                return False
            structure = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]])
            _, n = label(mask, structure=structure)
            return n == 1

        def _get_neighbors(r: int, c: int, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    yield nr, nc

        def _find_path(sr: int, sc: int, er: int, ec: int,
                       shape: Tuple[int, int]) -> List[Tuple[int, int]]:
            path, r, c = [], sr, sc
            while r != er:
                r += 1 if r < er else -1
                path.append((r, c))
            while c != ec:
                c += 1 if c < ec else -1
                path.append((r, c))
            return path[:-1]      # exclude endpoint already coloured

        def _connect_components(grid: np.ndarray, colours: List[int]) -> np.ndarray:
            mask = np.isin(grid, colours)
            structure = np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]])
            labelled, n_comp = label(mask, structure=structure)
            if n_comp <= 1:
                return grid

            sizes = [(labelled == i).sum() for i in range(1, n_comp + 1)]
            hub = np.argmax(sizes) + 1

            for comp_id in range(1, n_comp + 1):
                if comp_id == hub:
                    continue

                comp_coords = np.column_stack(np.where(labelled == comp_id))
                hub_coords = np.column_stack(np.where(labelled == hub))

                r1, c1, r2, c2, _ = min(
                    ((r1, c1, r2, c2,
                      abs(r1 - r2) + abs(c1 - c2))
                     for r1, c1 in comp_coords
                     for r2, c2 in hub_coords),
                    key=lambda t: t[4]
                )

                path = _find_path(r1, c1, r2, c2, grid.shape)
                for pr, pc in path:
                    if grid[pr, pc] == 0:
                        grid[pr, pc] = random.choice(colours)

            return grid

        def make_object() -> np.ndarray:
            h, w = max_r - min_r + 1, max_c - min_c + 1
            obj = np.zeros((h, w), dtype=int)

            sr, sc = h // 2, w // 2
            obj[sr, sc] = colours[0]
            frontier = [(sr, sc)]
            usage = {c: 0 for c in colours}
            usage[colours[0]] = 1

            target = random.randint(12, 20)
            while len(frontier) < target:
                fr, fc = random.choice(frontier)
                nbrs = [(nr, nc) for nr, nc in _get_neighbors(fr, fc, obj.shape)
                        if obj[nr, nc] == 0]
                if not nbrs:
                    frontier.remove((fr, fc))
                    if not frontier:
                        break
                    continue
                nr, nc = random.choice(nbrs)
                least = min(usage.values())
                pool = [c for c, n in usage.items() if n == least]
                colour = random.choice(pool)
                obj[nr, nc] = colour
                usage[colour] += 1
                frontier.append((nr, nc))

            # ensure every colour is present
            missing = [c for c, n in usage.items() if n == 0]
            if missing:
                filled = [(r, c) for r in range(h) for c in range(w) if obj[r, c]]
                for mcol in missing:
                    r, c = random.choice(filled)
                    usage[obj[r, c]] -= 1
                    obj[r, c] = mcol
                    usage[mcol] += 1

            if not _is_connected(obj):
                obj = _connect_components(obj, colours)

            return obj

        rows, cols = taskvars['rows'], taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)

        # 2 × 2 reference block
        colours = gridvars['colours']
        grid[0, 0], grid[0, 1], grid[1, 0], grid[1, 1] = colours

        # window around centre
        cr, cc = rows // 2, cols // 2
        min_r = max(4, cr - 4)
        max_r = min(rows - 2, cr + 4)
        min_c = max(4, cc - 4)
        max_c = min(cols - 2, cc + 4)

        # build centre object
        object_grid = retry(
            make_object,
            lambda x: (
                _is_connected(x) and
                len(set(x.flatten()) - {0}) == 4 and
                (x != 0).sum() >= 10
            ),
            max_attempts=50
        )

        for r in range(object_grid.shape[0]):
            for c in range(object_grid.shape[1]):
                if object_grid[r, c]:
                    grid[min_r + r, min_c + c] = object_grid[r, c]

        return grid

   

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        out = grid.copy()
        tl, tr = grid[0, 0], grid[0, 1]
        bl, br = grid[1, 0], grid[1, 1]
        cmap = {tl: tr, tr: tl, bl: br, br: bl}

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if r < 2 and c < 2:
                    continue
                col = grid[r, c]
                if col in cmap:
                    out[r, c] = cmap[col]
        return out

  

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {'rows': random.randint(12, 25),
                    'cols': random.randint(12, 25)}

        train, n_train = [], random.randint(3, 6)
        for _ in range(n_train):
            colours = random.sample(range(1, 10), 4)
            gv = {'colours': colours}
            inp = self.create_input(taskvars, gv)
            out = self.transform_input(inp, taskvars)
            train.append({'input': inp, 'output': out})

        colours = random.sample(range(1, 10), 4)
        gv = {'colours': colours}
        test_inp = self.create_input(taskvars, gv)
        test_out = self.transform_input(test_inp, taskvars)

        data: TrainTestData = {'train': train,
                               'test': [{'input': test_inp,
                                         'output': test_out}]}
        return taskvars, data


