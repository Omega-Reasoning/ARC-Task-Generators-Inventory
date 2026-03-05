from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List


class Taske76a88a6(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "In each input grid, there are multiple rectangles of equal size, all colored {color('background_color')}, except for one rectangle, which is divided into two randomly selected colors.",
            "In each input grid, the rectangles are positioned so that they do not touch each other.",
            "The two random colors vary across different input grids."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all the rectangle shapes, including one rectangle that consists of two colors.",
            "The two-color pattern from that rectangle is then applied to the other rectangles that originally had only a single color."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # ------------------------------------------------------------
    # INPUT GENERATION (Guaranteed Valid)
    # ------------------------------------------------------------

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        background_color = taskvars['background_color']
        color1 = gridvars['color1']
        color2 = gridvars['color2']

        grid = np.zeros((n, n), dtype=int)

        rect_height = random.randint(2, max(2, n // 4))
        rect_width = random.randint(2, max(2, n // 4))

        num_rectangles = random.randint(3, 4)

        placed = []

        for _ in range(num_rectangles):
            for _ in range(200):
                r = random.randint(0, n - rect_height)
                c = random.randint(0, n - rect_width)

                new_rect = (r, c, r + rect_height, c + rect_width)

                valid = True
                for pr in placed:
                    if not (new_rect[2] + 1 <= pr[0] or
                            pr[2] + 1 <= new_rect[0] or
                            new_rect[3] + 1 <= pr[1] or
                            pr[3] + 1 <= new_rect[1]):
                        valid = False
                        break

                if valid:
                    placed.append(new_rect)
                    break

        special_index = random.randint(0, len(placed) - 1)

        for i, (r1, c1, r2, c2) in enumerate(placed):
            if i == special_index:
                for r in range(r1, r2):
                    for c in range(c1, c2):
                        grid[r, c] = random.choice([color1, color2])
            else:
                grid[r1:r2, c1:c2] = background_color

        return grid

    # ------------------------------------------------------------
    # TRANSFORMATION (Self-contained, Dummy-safe)
    # ------------------------------------------------------------

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        background_color = taskvars['background_color']

        visited = set()
        rectangles = []

        rows, cols = grid.shape

        for r in range(rows):
            for c in range(cols):
                if (r, c) in visited:
                    continue
                if grid[r, c] == 0:
                    continue

                stack = [(r, c)]
                min_r = max_r = r
                min_c = max_c = c

                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited:
                        continue
                    if not (0 <= cr < rows and 0 <= cc < cols):
                        continue
                    if grid[cr, cc] == 0:
                        continue

                    visited.add((cr, cc))
                    min_r = min(min_r, cr)
                    max_r = max(max_r, cr)
                    min_c = min(min_c, cc)
                    max_c = max(max_c, cc)

                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        stack.append((cr+dr, cc+dc))

                rectangles.append((min_r, min_c, max_r+1, max_c+1))

        pattern = None

        for r1, c1, r2, c2 in rectangles:
            region = grid[r1:r2, c1:c2]
            unique = np.unique(region)
            if len(unique) == 2 and background_color not in unique:
                pattern = region.copy()
                break

        if pattern is None:
            return output

        for r1, c1, r2, c2 in rectangles:
            region = grid[r1:r2, c1:c2]
            unique = np.unique(region)

            if len(unique) == 1 and background_color in unique:
                if pattern.shape == region.shape:
                    output[r1:r2, c1:c2] = pattern
                else:
                    ph, pw = pattern.shape
                    th, tw = region.shape
                    scaled = np.zeros((th, tw), dtype=int)
                    for i in range(th):
                        for j in range(tw):
                            si = int(i * ph / th)
                            sj = int(j * pw / tw)
                            scaled[i, j] = pattern[si, sj]
                    output[r1:r2, c1:c2] = scaled

        return output

    # ------------------------------------------------------------
    # TASK CREATION (Never Fails)
    # ------------------------------------------------------------

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:

        taskvars = {
            'n': random.randint(12, 20),
            'background_color': random.randint(1, 9)
        }

        num_train = random.randint(3, 5)

        train = []
        used_pairs = set()

        while len(train) < num_train:
            color1 = random.randint(1, 9)
            color2 = random.randint(1, 9)

            if color1 == color2 or color1 == taskvars['background_color'] or color2 == taskvars['background_color']:
                continue
            if (color1, color2) in used_pairs:
                continue

            gridvars = {'color1': color1, 'color2': color2}
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)

            train.append({'input': inp, 'output': out})
            used_pairs.add((color1, color2))

        # Test example
        while True:
            color1 = random.randint(1, 9)
            color2 = random.randint(1, 9)
            if color1 != color2 and color1 != taskvars['background_color'] and color2 != taskvars['background_color']:
                break

        test_gridvars = {'color1': color1, 'color2': color2}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)

        return taskvars, {
            'train': train,
            'test': [{'input': test_input, 'output': test_output}]
        }