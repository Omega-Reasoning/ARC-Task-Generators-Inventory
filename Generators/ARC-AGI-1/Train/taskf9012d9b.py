from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random


class Taskf9012d9bGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid is square and filled with a repeating periodic 2D pattern.",
            "The pattern uses exactly {vars['n']} distinct non-zero colors.",
            "One square region located at a corner of the grid is replaced with 0s (empty cells).",
        ]
        transformation_reasoning_chain = [
            "Identify the small repeating block (the period) of the pattern from the non-zero portion of the grid.",
            "Locate the square region of 0 cells (the hole).",
            "For each cell inside the hole, determine its value by looking at the corresponding cell of the repeating block, based on its position within the period.",
            "The output grid is the reconstructed content of the hole, with the same dimensions as the hole.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        S = gridvars['S']
        pr, pc = gridvars['pr'], gridvars['pc']
        base = gridvars['base']
        hr0, hr1 = gridvars['hr0'], gridvars['hr1']
        hc0, hc1 = gridvars['hc0'], gridvars['hc1']

        grid = np.zeros((S, S), dtype=int)
        for i in range(S):
            for j in range(S):
                grid[i, j] = base[i % pr, j % pc]
        grid[hr0:hr1, hc0:hc1] = 0
        return grid

    def transform_input(self, grid, taskvars):
        H, W = grid.shape
        zero_mask = (grid == 0)
        rows = np.where(zero_mask.any(axis=1))[0]
        cols = np.where(zero_mask.any(axis=0))[0]
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1

        def try_period(pr, pc):
            base = {}
            for i in range(H):
                for j in range(W):
                    if grid[i, j] != 0:
                        key = (i % pr, j % pc)
                        if key in base and base[key] != grid[i, j]:
                            return None
                        base[key] = int(grid[i, j])
            if len(base) < pr * pc:
                return None
            return base

        found_base, found_pr, found_pc = None, None, None
        for pr in range(1, H + 1):
            for pc in range(1, W + 1):
                result = try_period(pr, pc)
                if result is not None:
                    found_base = result
                    found_pr, found_pc = pr, pc
                    break
            if found_base is not None:
                break

        output = np.zeros((r1 - r0, c1 - c0), dtype=int)
        for i in range(r0, r1):
            for j in range(c0, c1):
                output[i - r0, j - c0] = found_base[(i % found_pr, j % found_pc)]
        return output

    def create_grids(self):
        n = random.randint(2, 4)
        taskvars = {'n': n}

        num_train = random.randint(3, 4)

        def make_gridvars():
            valid_periods = [(pr, pc) for pr in [2, 3] for pc in [2, 3] if pr * pc >= n]
            pr, pc = random.choice(valid_periods)

            min_side = 2 * max(pr, pc)
            S = random.randint(min_side, 8)

            colors = random.sample(range(1, 10), n)

            base = None
            for _ in range(500):
                candidate = np.array([[random.choice(colors) for _ in range(pc)]
                                      for _ in range(pr)])
                if set(candidate.flatten()) == set(colors):
                    base = candidate
                    break
            if base is None:
                flat = list(colors) + [random.choice(colors)
                                       for _ in range(pr * pc - n)]
                random.shuffle(flat)
                base = np.array(flat).reshape(pr, pc)

            max_hole_side = min(pr, pc)
            hole_side = random.randint(1, max_hole_side)
            hh = hw = hole_side

            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            if corner == 'tl':
                hr0, hc0 = 0, 0
            elif corner == 'tr':
                hr0, hc0 = 0, S - hw
            elif corner == 'bl':
                hr0, hc0 = S - hh, 0
            else:
                hr0, hc0 = S - hh, S - hw

            return {
                'S': S, 'pr': pr, 'pc': pc, 'base': base,
                'hr0': hr0, 'hr1': hr0 + hh,
                'hc0': hc0, 'hc1': hc0 + hw,
            }

        def make_pair():
            for _ in range(50):
                gridvars = make_gridvars()
                inp = self.create_input(taskvars, gridvars)
                non_zero_colors = set(int(v) for v in np.unique(inp) if v != 0)
                if len(non_zero_colors) == n:
                    out = self.transform_input(inp, taskvars)
                    return {'input': inp, 'output': out}
            gridvars = make_gridvars()
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)
            return {'input': inp, 'output': out}

        train = [make_pair() for _ in range(num_train)]
        test = [make_pair()]
        return taskvars, {'train': train, 'test': test}


