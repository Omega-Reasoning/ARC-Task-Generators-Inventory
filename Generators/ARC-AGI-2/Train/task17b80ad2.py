from typing import Dict, Any, Tuple
import random
import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
class Task17b80ad2Generator(ARCTaskGenerator):
 

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains multiple single-colored cells, randomly distributed within the grid but never in the last row.",
            "Several {color('cell_color')} cells are added to the last row of the input grid (never in adjacent columns).",
            "Finally, when checking the columns containing {color('cell_color')} cells, there are empty (0) cells directly above as well as other coloured cells further up to demonstrate the transformation.",
        ]

        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and identifying the {color('cell_color')} cells in the last row of the input grid.",
            "The transformation involves adding more cells to completely fill the columns containing {color('cell_color')} cells.",
            "The filling starts from the last row and moves upward toward the first row.",
            "Starting with {color('cell_color')} cells, additional {color('cell_color')} cells are added in the same column until another differently colored cell is encountered, at which point the extension continues with the new color.",
            "Each time a new colored cell is encountered in a column, the extension color changes accordingly to match the new colored cell.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)


    @staticmethod
    def _random_grid_size() -> Tuple[int, int]:
        """Random height × width in the inclusive 5–30 range."""
        return random.randint(5, 30), random.randint(5, 30)

    @staticmethod
    def _fill_columns(grid: np.ndarray, cell_color: int) -> np.ndarray:
        """Return a new grid with the column‑filling transformation applied."""
        out = grid.copy()
        h, w = out.shape
        for col in range(w):
            if out[-1, col] != cell_color:
                continue  # skip non‑marker columns
            current = cell_color
            for row in range(h - 1, -1, -1):
                if out[row, col] == 0:
                    out[row, col] = current
                else:
                    current = out[row, col]
        return out


    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:  # noqa: D401
        cell_color: int = taskvars["cell_color"]
        height, width = self._random_grid_size()
        grid = np.zeros((height, width), dtype=int)

        # 1. Place non‑adjacent markers
        max_markers = max(2, width // 2)
        num_markers = random.randint(2, max_markers)
        available_cols = set(range(width))
        marker_cols = []
        while len(marker_cols) < num_markers and available_cols:
            c = random.choice(list(available_cols))
            marker_cols.append(c)
            for n in (c - 1, c, c + 1):
                available_cols.discard(n)
        marker_cols.sort()
        for c in marker_cols:
            grid[-1, c] = cell_color
            grid[-2, c] = 0  # keep immediate cell blank

        # 2. Guarantee at least one other colour in each marker column
        palette = [c for c in range(1, 10) if c != cell_color]
        for c in marker_cols:
            rows = list(range(height - 2))
            random.shuffle(rows)
            for r in rows:
                if grid[r, c] == 0:
                    grid[r, c] = random.choice(palette)
                    break

        # 3. Scatter extra non‑cell_color dots
        total = height * width
        n_extra = random.randint(max(2, total // 20), total // 3)
        placed = 0
        while placed < n_extra:
            r = random.randint(0, height - 3)
            c = random.randint(0, width - 1)
            if grid[r, c] == 0:
                grid[r, c] = random.choice(palette)
                placed += 1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        return self._fill_columns(grid, taskvars["cell_color"])

   
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars: Dict[str, Any] = {"cell_color": random.randint(1, 9)}
        num_train = random.randint(3, 5)
        return taskvars, self.create_grids_default(num_train, 1, taskvars)

