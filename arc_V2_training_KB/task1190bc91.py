import random
from typing import Dict, Any, Tuple, List
import numpy as np

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData


class Task1190bc91Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each grid contains one multi-colored vertical or horizontal line, along with one or two additional lines formed by 4-way connecting two same-colored blocks.",
            "The multi-colored line is between 3 and {vars['grid_size'] - 2} cells long and, if horizontal, it must start from the first or last column and lie on one of the middle rows; if vertical, it must start from the first or last row and lie on one of the middle columns.",
            "The one or two additional lines, formed by 4-way connecting two same-colored blocks, can be placed **only** in three specific border regions determined by the multi‑coloured line’s orientation and start side.",
            "If the multi-colored line is horizontal and starts from the last column, the additional lines may appear as (1) a horizontal line in the first row, (2) a horizontal line in the last row, or (3) a vertical line in the first column—never in the last column.  Symmetric rules apply to the other three orientation/side combinations.",
            "Whenever a horizontal additional line is chosen, its two cells occupy consecutive middle columns that are also occupied by the multi‑coloured line.  Whenever a vertical additional line is chosen, its two cells occupy consecutive middle rows that are also occupied by the multi‑coloured line.",
            "All colours are unique: every cell of the multi‑coloured line has a distinct colour, and every additional line uses a colour absent from the spine and from any other additional line."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying one multi-colored vertical or horizontal line, along with the additional border lines.",
            "Each cell of the multi-coloured line is extended diagonally in all four directions using its own colour, stopping at coloured cells or the grid boundary.",
            "These diagonal rays partition the grid into three regions.",
            "Empty (0) cells *inside* any region that contains at least one additional 2‑cell border line are flood‑filled with that border‑line’s colour.",
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def _place_multi_coloured_line(self, grid: np.ndarray, gridvars: Dict[str, Any]) -> Dict[str, Any]:
        n = grid.shape[0]
        orient = gridvars.get("orientation", random.choice(["horizontal", "vertical"]))
        # force start_side to always be "last"
        start_side = "last"

        mid = n // 2
        middle_band = [max(0, mid - 1), mid, min(n - 1, mid + 1)]

        info: Dict[str, Any] = {"orient": orient, "start_side": start_side}

        if orient == "horizontal":
            row = gridvars.get("row", random.choice(middle_band))
            length = gridvars.get("length", random.randint(3, min(6, n - 2)))  # capped to 6 for color safety
            colours = random.sample(range(1, 10), length)

            cols = list(range(n - 1, n - length - 1, -1))  # always from last column
            for c_idx, col in zip(cols, colours):
                grid[row, c_idx] = col

            info.update({"row": row, "cols": cols, "length": length, "colours": colours})
        else:  # vertical
            col = gridvars.get("col", random.choice(middle_band))
            length = gridvars.get("length", random.randint(3, min(6, n - 2)))
            colours = random.sample(range(1, 10), length)

            rows = list(range(n - 1, n - length - 1, -1))  # always from last row
            for r_idx, col_val in zip(rows, colours):
                grid[r_idx, col] = col_val

            info.update({"col": col, "rows": rows, "length": length, "colours": colours})

        return info


    def _middle_two_indices(self, seq: List[int]) -> Tuple[int, int]:
        L = len(seq)
        mid = L // 2 - 1 if L % 2 == 0 else L // 2 - 1
        mid = max(0, mid)
        return seq[mid], seq[mid + 1]

    def _choose_new_colour(self, used: set) -> int:
        available = [c for c in range(1, 10) if c not in used]
        if not available:
            raise ValueError("No available colours left to choose from. Reduce spine or additional lines.")
        colour = random.choice(available)
        used.add(colour)
        return colour

    def _place_additional_lines(self, grid: np.ndarray, spine_info: Dict[str, Any], used_colours: set):
        n = grid.shape[0]
        orient = spine_info["orient"]
        start_side = spine_info["start_side"]

        if orient == "horizontal":
            horiz_rows = [0, n - 1]
            vert_col = 0 if start_side == "last" else n - 1
            regions = [("h", r) for r in horiz_rows]
            if vert_col != 0:
                regions.append(("v", vert_col))
        else:
            vert_cols = [0, n - 1]
            horiz_row = 0 if start_side == "last" else n - 1
            regions = [("v", c) for c in vert_cols] + [("h", horiz_row)]

        num_lines = random.choice([1, 2])
        chosen_regions = random.sample(regions, k=num_lines)
        chosen_regions.sort(key=lambda x: 0 if x[0] == "h" else 1)

        horiz_line_row = None
        vert_line_col = None

        for kind, coord in chosen_regions:
            if kind == "h":
                row = coord
                horiz_line_row = row

                if orient == "horizontal":
                    col1, col2 = self._middle_two_indices(spine_info["cols"])
                else:
                    spine_col = spine_info["col"]
                    if row == 0:
                        col1, col2 = (spine_col, spine_col + 1) if spine_col + 1 < n else (spine_col - 1, spine_col)
                    else:
                        col1, col2 = (spine_col - 1, spine_col) if spine_col - 1 >= 0 else (spine_col, spine_col + 1)

                colour = self._choose_new_colour(used_colours)
                grid[row, col1] = colour
                grid[row, col2] = colour

            else:
                col = coord
                vert_line_col = col

                if orient == "horizontal":
                    spine_row = spine_info["row"]
                    if horiz_line_row is None:
                        direction = random.choice(["up", "down"])
                    else:
                        direction = "down" if horiz_line_row == 0 else "up"

                    if direction == "down":
                        start_row = min(spine_row + 1, n - 2)
                        rows = [start_row, start_row + 1]
                    else:
                        start_row = max(spine_row - 2, 0)
                        rows = [start_row, start_row + 1]
                else:
                    row1, row2 = self._middle_two_indices(spine_info["rows"])
                    rows = [row1, row2]

                colour = self._choose_new_colour(used_colours)
                for r in rows:
                    grid[r, col] = colour

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars["grid_size"]
        grid = np.zeros((n, n), dtype=int)

        spine_info = self._place_multi_coloured_line(grid, gridvars)
        used = set(spine_info["colours"])

        self._place_additional_lines(grid, spine_info, used)
        return grid

    def _find_multi_coloured_line(self, grid: np.ndarray) -> List[Tuple[int, int]]:
        n = grid.shape[0]
        for r in range(n):
            c = 0
            while c < n:
                if grid[r, c] == 0:
                    c += 1
                    continue
                start = c
                colours = set()
                while c < n and grid[r, c] != 0:
                    colours.add(grid[r, c])
                    c += 1
                length = c - start
                if length >= 3 and len(colours) == length:
                    return [(r, cc) for cc in range(start, c)]

        for c in range(n):
            r = 0
            while r < n:
                if grid[r, c] == 0:
                    r += 1
                    continue
                start = r
                colours = set()
                while r < n and grid[r, c] != 0:
                    colours.add(grid[r, c])
                    r += 1
                length = r - start
                if length >= 3 and len(colours) == length:
                    return [(rr, c) for rr in range(start, r)]
        raise ValueError("Malformed input: multi‑coloured line not found")

    @staticmethod
    def _extend_diagonally(grid: np.ndarray, positions: List[Tuple[int, int]]):
        n = grid.shape[0]
        for r, c in positions:
            colour = grid[r, c]
            for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                while 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == 0:
                    grid[nr, nc] = colour
                    nr += dr
                    nc += dc

    @staticmethod
    def _flood_fill(grid: np.ndarray, seeds: List[Tuple[int, int]], fill_colour: int):
        n = grid.shape[0]
        stack = seeds[:]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr, nc] == 0:
                    grid[nr, nc] = fill_colour
                    stack.append((nr, nc))

    def _fill_regions_with_border_colours(self, grid: np.ndarray):
        n = grid.shape[0]
        visited = set()
        for row in (0, n - 1):
            for c in range(1, n - 1):
                if (row, c) in visited or grid[row, c] == 0:
                    continue
                if grid[row, c] == grid[row, c + 1]:
                    colour = grid[row, c]
                    seeds = [(row, c), (row, c + 1)]
                    visited.update(seeds)
                    self._flood_fill(grid, seeds, colour)
        for col in (0, n - 1):
            for r in range(1, n - 1):
                if (r, col) in visited or grid[r, col] == 0:
                    continue
                if grid[r, col] == grid[r + 1, col]:
                    colour = grid[r, col]
                    seeds = [(r, col), (r + 1, col)]
                    visited.update(seeds)
                    self._flood_fill(grid, seeds, colour)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        out = grid.copy()
        spine_cells = self._find_multi_coloured_line(out)
        self._extend_diagonally(out, spine_cells)
        self._fill_regions_with_border_colours(out)
        return out

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {"grid_size": random.randint(7, 30)}
        n = taskvars["grid_size"]
        mid = n // 2
        middle_band = [max(0, mid - 1), mid, min(n - 1, mid + 1)]

        train_gridvars: List[Dict[str, Any]] = [
            {
                "orientation": "horizontal",
                "start_side": "last",
                "row": random.choice(middle_band),
            },
            {
                "orientation": "vertical",
                "start_side": "last",
                "col": random.choice(middle_band),
            }
        ]

        extra = random.randint(0, 1)
        for _ in range(extra):
            train_gridvars.append({})  # fully random but still constrained in method

        random.shuffle(train_gridvars)

        train_pairs = []
        for gvars in train_gridvars:
            inp = self.create_input(taskvars, gvars)
            out = self.transform_input(inp, taskvars)
            train_pairs.append({"input": inp, "output": out})

        test_inp = self.create_input(taskvars, {})
        test_out = self.transform_input(test_inp, taskvars)

        return taskvars, {"train": train_pairs, "test": [{"input": test_inp, "output": test_out}]}



