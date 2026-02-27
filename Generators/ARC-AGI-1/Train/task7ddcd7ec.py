import random
from typing import Any, Dict, List, Tuple, Set

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData


class Task7ddcd7ecGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} × {vars['n']}.",
            "Each grid contains exactly one filled square block of size {vars['k']} × {vars['k']}.",
            "The color of the block may differ across grids.",
            "The block is placed away from the grid border, leaving at least two empty-cell layers on all four sides of the block.",
            "In addition to the block, there are one or more single cells (diagonal markers) with the same color as the block.",
            "Each diagonal marker lies on a diagonal extending outward from one of the block’s corners and is placed immediately outside the block (distance 1).",
            "At most one diagonal marker appears per block-corner diagonal direction.",
        ]
        transformation_reasoning_chain = [
            "The output grid is of size {vars['n']} × {vars['n']}.",
            "Locate the filled square block and identify its color.",
            "Check the four diagonal-adjacent positions immediately outside the block corners (one step outward along each corner diagonal).",
            "For each position that contains a diagonal marker cell in the block color, extend that color further along the same diagonal direction until the grid boundary is reached.",
            "All remaining cells stay empty (0).",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def _block_corners(r0: int, c0: int, k: int) -> Dict[str, Tuple[int, int]]:
        return {
            "tl": (r0, c0),
            "tr": (r0, c0 + k - 1),
            "bl": (r0 + k - 1, c0),
            "br": (r0 + k - 1, c0 + k - 1),
        }

    @staticmethod
    def _diag_dirs() -> Dict[str, Tuple[int, int]]:
        return {
            "tl": (-1, -1),
            "tr": (-1, +1),
            "bl": (+1, -1),
            "br": (+1, +1),
        }

    @staticmethod
    def _choose_marker_dirs(count: int) -> Set[str]:
        dirs = ["tl", "tr", "bl", "br"]
        return set(random.sample(dirs, k=count))

    # -------------------------
    # required API
    # -------------------------
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # pick task variables (fixed for the whole task)
        n = random.randint(9, 30)
        k = random.choice([2, 3, 4])
        taskvars = {"n": n, "k": k}

        nr_train = random.randint(3, 6)

        # Constraint: marker set/count differs between train and test.
        # We'll enforce COUNT differs (1..4 markers).
        all_counts = [1, 2, 3, 4]
        test_count = random.choice(all_counts)
        train_pool = [c for c in all_counts if c != test_count]
        train_counts = [random.choice(train_pool) for _ in range(nr_train)]

        train: List[GridPair] = []
        for i in range(nr_train):
            color = random.randint(1, 9)

            # place block so that 2-cell margin holds AND diagonal markers (distance 1) fit in-bounds
            # need r0 in [2 .. n-k-3]
            r0 = random.randint(2, n - k - 3)
            c0 = random.randint(2, n - k - 3)

            marker_dirs = self._choose_marker_dirs(train_counts[i])

            gridvars = {
                "color": color,
                "r0": r0,
                "c0": c0,
                "marker_dirs": marker_dirs,
            }
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)
            train.append({"input": inp, "output": out})

        # Test (count differs from training by construction)
        color = random.randint(1, 9)
        r0 = random.randint(2, n - k - 3)
        c0 = random.randint(2, n - k - 3)
        marker_dirs = self._choose_marker_dirs(test_count)

        test_inp = self.create_input(
            taskvars, {"color": color, "r0": r0, "c0": c0, "marker_dirs": marker_dirs}
        )
        test_out = self.transform_input(test_inp, taskvars)

        data: TrainTestData = {"train": train, "test": [{"input": test_inp, "output": test_out}]}
        return taskvars, data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = int(taskvars["n"])
        k = int(taskvars["k"])
        col = int(gridvars["color"])
        r0 = int(gridvars["r0"])
        c0 = int(gridvars["c0"])
        marker_dirs: Set[str] = set(gridvars["marker_dirs"])

        grid = np.zeros((n, n), dtype=int)

        # filled k×k block
        grid[r0 : r0 + k, c0 : c0 + k] = col

        # place diagonal markers immediately outside block corners (distance 1)
        corners = self._block_corners(r0, c0, k)
        dirs = self._diag_dirs()

        for name in marker_dirs:
            cr, cc = corners[name]
            dr, dc = dirs[name]
            mr, mc = cr + dr, cc + dc
            if 0 <= mr < n and 0 <= mc < n:
                grid[mr, mc] = col

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        n = int(taskvars["n"])
        k = int(taskvars["k"])
        out = grid.copy()

        # identify block color: most frequent non-zero color (block dominates markers)
        nonzero = out[out != 0]
        if nonzero.size == 0:
            return out
        block_color = int(max(set(nonzero.tolist()), key=nonzero.tolist().count))

        # find the k×k filled block of that color
        found = None
        for r0 in range(n - k + 1):
            for c0 in range(n - k + 1):
                patch = out[r0 : r0 + k, c0 : c0 + k]
                if np.all(patch == block_color):
                    found = (r0, c0)
                    break
            if found:
                break
        if not found:
            return out

        r0, c0 = found
        corners = self._block_corners(r0, c0, k)
        dirs = self._diag_dirs()

        for name, (cr, cc) in corners.items():
            dr, dc = dirs[name]
            mr, mc = cr + dr, cc + dc  # marker position
            if 0 <= mr < n and 0 <= mc < n and out[mr, mc] == block_color:
                rr, cc2 = mr + dr, mc + dc
                while 0 <= rr < n and 0 <= cc2 < n:
                    out[rr, cc2] = block_color
                    rr += dr
                    cc2 += dc

        return out


