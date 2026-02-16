from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List


class Task0a938d79Generator(ARCTaskGenerator):
    

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "Input grids contain exactly two single‑coloured cells, with the remaining cells being empty (0).",
            "The cells differ in colour and follow one of four templates: (i) first & last row, (ii) first & last column, (iii) same first/last row, (iv) same first/last column.",
            "If the two colored cells are positioned in the first and last rows, they must be placed so that there is enough space on the right half of the grid to add vertical stripes with the same gap as in the columns containing the two cells.",
            "If the two colored cells are positioned in the first and last columns, they must be placed so that there is enough space on the lower half of the grid to add horizontal stripes with the same gap as in the rows containing the two cells.",
            "If the two colored cells are positioned in the same row, they must be placed so that there is enough space on the right half of the grid to add  vertical stripes with the same gap as in the columns containing the two cells.",
            "If the two colored cells are positioned in the same column, they must be placed so that there is enough space on the lower half of the grid to add  horizontal stripes with the same gap as in the rows containing the two cells."
        ]

        transformation_reasoning_chain = [
            "The output grids are created by copying the input grids and identifying the two single-colored cells, which can be located in one of four templates: (i) first & last row, (ii) first & last column, (iii) same first/last row, (iv) same first/last column.",
            "If the two colored cells are in the first and last rows or in the same first/last row, vertical stripes are added on the right half of the grid; if they are in the first and last columns or in the same first/last column, horizontal stripes are added on the lower half of the grid.",
            "Let *k* be the gap between cells along the varying axis (≥ 1).",
            "Vertical: add vertical stripes every *k* columns across the width, starting at the left‑most cell column and alternating the two colours.",
            "Horizontal: add horizontal stripes every *k* rows, starting at the upper‑most cell row and alternating the two colours.",
            "All stripes are exactly one cell thick and perfectly evenly spaced." 
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        color1, color2 = taskvars['color1'], taskvars['color2']
        placement_type = gridvars['placement_type']

        grid = np.zeros((rows, cols), dtype=int)

        if placement_type == "first_last_row":  # seeds in first & last rows
            left_half_end = max(1, cols // 2 - 1)  # last index included in left half
            candidates = list(range(1, left_half_end + 1))  # avoid column 0 to guarantee gap from border
            
            # Pick two indices with gap between 1 and min(4, cols//4) to ensure multiple stripes
            max_gap = min(4, cols // 4, left_half_end)
            pairs = [(a, b) for a in candidates for b in candidates 
                    if a != b and 1 <= abs(a - b) <= max_gap]
            if not pairs:
                # Fallback: just ensure they're different
                pairs = [(a, b) for a in candidates for b in candidates if a != b]
            c1, c2 = random.choice(pairs)
            
            grid[0,        c1] = color1
            grid[rows - 1, c2] = color2

        elif placement_type == "first_last_col":  # seeds in first & last columns
            top_half_end = max(1, rows // 2 - 1)
            candidates = list(range(1, top_half_end + 1))
            
            # Pick two indices with gap between 1 and min(4, rows//4) to ensure multiple stripes
            max_gap = min(4, rows // 4, top_half_end)
            pairs = [(a, b) for a in candidates for b in candidates 
                    if a != b and 1 <= abs(a - b) <= max_gap]
            if not pairs:
                pairs = [(a, b) for a in candidates for b in candidates if a != b]
            r1, r2 = random.choice(pairs)
            
            grid[r1, 0]        = color1
            grid[r2, cols - 1] = color2

        elif placement_type == "same_row":        # both seeds in row 0 or last row
            row = random.choice([0, rows - 1])
            candidates = list(range(1, cols))  # exclude column 0 to keep distance from edge
            
            # Pick two indices with gap between 1 and min(4, cols//4) to ensure multiple stripes
            max_gap = min(4, cols // 4)
            pairs = [(a, b) for a in candidates for b in candidates 
                    if a != b and 1 <= abs(a - b) <= max_gap]
            if not pairs:
                pairs = [(a, b) for a in candidates for b in candidates if a != b]
            c1, c2 = random.choice(pairs)
            
            grid[row, c1] = color1
            grid[row, c2] = color2

        elif placement_type == "same_col":        # both seeds in column 0 or last column
            col = random.choice([0, cols - 1])
            candidates = list(range(1, rows))  # exclude row 0
            
            # Pick two indices with gap between 1 and min(4, rows//4) to ensure multiple stripes
            max_gap = min(4, rows // 4)
            pairs = [(a, b) for a in candidates for b in candidates 
                    if a != b and 1 <= abs(a - b) <= max_gap]
            if not pairs:
                pairs = [(a, b) for a in candidates for b in candidates if a != b]
            r1, r2 = random.choice(pairs)
            
            grid[r1, col] = color1
            grid[r2, col] = color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = grid.shape
        output = grid.copy()

        seeds = [(r, c, grid[r, c])
                 for r in range(rows)
                 for c in range(cols)
                 if grid[r, c] != 0]
        if len(seeds) != 2:
            return output

        (r1, c1, col1), (r2, c2, col2) = seeds

        # vertical‑stripe cases ───────────────────────────────────────
        if (r1 in {0, rows - 1} and r2 in {0, rows - 1}) or (r1 == r2):
            output[:, c1] = col1
            output[:, c2] = col2

            k = abs(c1 - c2) or 1  # k ≥ 1
            first_col = min(c1, c2)
            first_colour, second_colour = (col1, col2) if c1 == first_col else (col2, col1)

            start = max(c1, c2) + k
            colour = second_colour if ((start - first_col) // k) % 2 else first_colour
            for cc in range(start, cols, k):
                output[:, cc] = colour
                colour = second_colour if colour == first_colour else first_colour

        # horizontal‑stripe cases ─────────────────────────────────────
        else:  # seeds in first/last columns or same column
            output[r1, :] = col1
            output[r2, :] = col2

            k = abs(r1 - r2) or 1
            first_row = min(r1, r2)
            top_colour, bottom_colour = (col1, col2) if r1 == first_row else (col2, col1)

            colour = top_colour
            for rr in range(first_row, rows, k):
                output[rr, :] = colour
                colour = bottom_colour if colour == top_colour else top_colour

        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {
            'rows': random.randint(7, 30),
            'cols': random.randint(7, 30),
            'color1': random.choice(range(1, 10)),
            'color2': None
        }
        taskvars['color2'] = random.choice([c for c in range(1, 10) if c != taskvars['color1']])

        required = ["first_last_row", "first_last_col", "same_col"]
        placements = required.copy()
        while len(placements) < random.randint(3, 6):
            placements.append(random.choice(["first_last_row", "first_last_col", "same_row", "same_col"]))

        train: List[GridPair] = []
        for p in placements:
            inp = self.create_input(taskvars, {'placement_type': p})
            out = self.transform_input(inp, taskvars)
            train.append({'input': inp, 'output': out})

        test_type = random.choice(["first_last_row", "first_last_col", "same_row", "same_col"])
        test_inp = self.create_input(taskvars, {'placement_type': test_type})
        test_out = self.transform_input(test_inp, taskvars)
        test = [{'input': test_inp, 'output': test_out}]

        return taskvars, {'train': train, 'test': test}


