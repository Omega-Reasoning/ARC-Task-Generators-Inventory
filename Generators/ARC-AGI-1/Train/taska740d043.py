import numpy as np
import random
from typing import Dict, Any, Tuple
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import create_object, retry, Contiguity, random_cell_coloring


class Taska740d043Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has a background filled with color {color('background')}.",
            "Some cells are non-background, colored with one or more distinct colors, forming a pattern.",
            "The non-background cells occupy some rectangular bounding box region within the grid."
        ]
        transformation_reasoning_chain = [
            "Identify all cells whose color is not the background color {color('background')}.",
            "Compute the minimal bounding box that contains all these non-background cells.",
            "Extract the sub-grid corresponding to this bounding box.",
            "In the extracted sub-grid, replace all cells that still have the background color {color('background')} with 0 (empty).",
            "The resulting sub-grid is the output."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        background = taskvars['background']
        # Available foreground colors (exclude background and 0)
        available = [c for c in range(1, 10) if c != background]

        def gen():
            H = random.randint(5, 15)
            W = random.randint(5, 15)
            grid = np.full((H, W), background, dtype=int)

            # Pick foreground colors (1-3)
            n_colors = random.randint(1, 3)
            fg_colors = random.sample(available, n_colors)

            # Place an object region somewhere inside the grid (avoid touching edges so
            # bounding box clearly sits inside)
            obj_h = random.randint(1, min(H - 2, 5))
            obj_w = random.randint(1, min(W - 2, 5))
            r0 = random.randint(1, H - obj_h - 1)
            c0 = random.randint(1, W - obj_w - 1)

            # Draw some non-background cells within that region
            # Ensure at least 2 non-background cells and that bbox spans >= 2 in at least one dim
            placed = []
            for r in range(obj_h):
                for c in range(obj_w):
                    if random.random() < 0.55:
                        grid[r0 + r, c0 + c] = random.choice(fg_colors)
                        placed.append((r0 + r, c0 + c))

            # Guarantee bounding box equals chosen region by filling the 4 corner-hits:
            # place at top-left-most and bottom-right-most to anchor the bbox
            grid[r0, c0 + random.randint(0, obj_w - 1)] = random.choice(fg_colors)  # top row
            grid[r0 + obj_h - 1, c0 + random.randint(0, obj_w - 1)] = random.choice(fg_colors)  # bottom row
            grid[r0 + random.randint(0, obj_h - 1), c0] = random.choice(fg_colors)  # left col
            grid[r0 + random.randint(0, obj_h - 1), c0 + obj_w - 1] = random.choice(fg_colors)  # right col

            return grid

        def valid(grid):
            mask = grid != background
            if mask.sum() < 2:
                return False
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            bbox_h = rmax - rmin + 1
            bbox_w = cmax - cmin + 1
            # Require that the output grid is different from the input (smaller)
            if bbox_h == grid.shape[0] and bbox_w == grid.shape[1]:
                return False
            # Require at least one background cell inside the bbox so that output has a 0
            sub = grid[rmin:rmax + 1, cmin:cmax + 1]
            if not np.any(sub == background):
                # still valid but less interesting; allow sometimes
                return random.random() < 0.3
            return True

        return retry(gen, valid, max_attempts=200)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        background = taskvars['background']
        mask = grid != background
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        sub = grid[rmin:rmax + 1, cmin:cmax + 1].copy()
        sub[sub == background] = 0
        return sub

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        background = random.randint(1, 9)
        taskvars = {'background': background}
        n_train = random.randint(3, 5)
        return taskvars, self.create_grids_default(n_train, 1, taskvars)


if __name__ == "__main__":
    gen = BoundingBoxCropGenerator()
    taskvars, data = gen.create_grids()
    print("Task variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(data)