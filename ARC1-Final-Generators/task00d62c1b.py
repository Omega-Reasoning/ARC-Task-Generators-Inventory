from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from input_library import create_object, retry


class Task00d62c1bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids have different sizes.",
            "Each input grid contains several {color('cell_color1')} objects, while the remaining cells are empty (0).",
            "The {color('cell_color1')} objects have different shapes, with at least one structure in each input that encloses one or more empty (0) cells from all sides.",
            "The shape and position of these objects vary across examples."
        ]

        transformation_reasoning_chain = [
            "The output grids are created by copying the input grids and identifying all empty (0) regions that are fully enclosed on all sides by {color('cell_color1')} cells.",
            "All such enclosed empty (0) cells are recolored with {color('cell_color2')} color.",
            "The remaining cells stay unchanged."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        # pick two distinct non-zero colors
        colors = random.sample(range(1, 10), 2)
        taskvars = {
            'cell_color1': colors[0],
            'cell_color2': colors[1]
        }

        nr_train = random.randint(3, 5)

        train = []
        for _ in range(nr_train):
            grid = self.create_input(taskvars, {})
            out = self.transform_input(grid.copy(), taskvars)
            train.append({'input': grid, 'output': out})

        # ensure test example is different in size/placement
        test_grid = self.create_input(taskvars, {})
        test_out = self.transform_input(test_grid.copy(), taskvars)

        return taskvars, {'train': train, 'test': [{'input': test_grid, 'output': test_out}]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create a grid containing multiple {cell_color1} objects, at least one of which
        forms a closed enclosure (a ring) that leaves one or more zero cells fully enclosed.
        We randomise sizes, positions and enclosure types to ensure variety.
        """
        # grid size between 5 and 20 (keeps things compact but varied)
        rows = random.randint(5, 20)
        cols = random.randint(5, 20)
        grid = np.zeros((rows, cols), dtype=int)

        color1 = taskvars['cell_color1']

        def compute_enclosed(mask_grid: np.ndarray) -> np.ndarray:
            # mark zeros reachable from border
            visited = np.zeros_like(mask_grid, dtype=bool)
            stack = []
            for r in range(rows):
                for c in (0, cols - 1):
                    if mask_grid[r, c] == 0:
                        stack.append((r, c))
            for c in range(cols):
                for r in (0, rows - 1):
                    if mask_grid[r, c] == 0:
                        stack.append((r, c))

            while stack:
                r, c = stack.pop()
                if visited[r, c]:
                    continue
                if mask_grid[r, c] != 0:
                    continue
                visited[r, c] = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and mask_grid[nr, nc] == 0:
                        stack.append((nr, nc))

            return (mask_grid == 0) & (~visited)

        def place_rectangle_ring(top, left, h, w):
            # ensure inside area exists
            if h < 3 or w < 3:
                return False
            # perimeter coords
            r0, c0 = top, left
            r1, c1 = top + h - 1, left + w - 1
            coords = [(r0, c) for c in range(c0, c1 + 1)] + [(r1, c) for c in range(c0, c1 + 1)] + \
                     [(r, c0) for r in range(r0, r1 + 1)] + [(r, c1) for r in range(r0, r1 + 1)]
            enclosed_now = compute_enclosed(grid)
            # don't place if any perimeter cell is inside an already enclosed region
            for (r, c) in coords:
                if enclosed_now[r, c]:
                    return False
            for (r, c) in coords:
                grid[r, c] = color1
            return True

        def place_partial_enclosure(top, left, h, w, open_sides: tuple):
            # open_sides: tuple of sides to leave open, e.g. ('top',) or ('left','top')
            if h < 2 or w < 2:
                return False
            r0, c0 = top, left
            r1, c1 = top + h - 1, left + w - 1
            coords = set()
            # top
            if 'top' not in open_sides:
                coords.update({(r0, c) for c in range(c0, c1 + 1)})
            # bottom
            if 'bottom' not in open_sides:
                coords.update({(r1, c) for c in range(c0, c1 + 1)})
            # left
            if 'left' not in open_sides:
                coords.update({(r, c0) for r in range(r0, r1 + 1)})
            # right
            if 'right' not in open_sides:
                coords.update({(r, c1) for r in range(r0, r1 + 1)})

            enclosed_now = compute_enclosed(grid)
            for (r, c) in coords:
                if enclosed_now[r, c]:
                    return False
            for (r, c) in coords:
                grid[r, c] = color1
            return True

        def place_irregular_enclosure(top, left, h, w):
            # create a jagged enclosure by drawing two overlapping rectangle rings
            ok = place_rectangle_ring(top, left, h, w)
            if not ok:
                return False
            # add an inner small rectangle that blocks some interior adjacency to force diagonal-only holes
            if h >= 5 and w >= 5:
                inner_h = max(1, h - 4)
                inner_w = max(1, w - 4)
                inner_top = top + 2
                inner_left = left + 2
                center_r = inner_top + inner_h // 2
                center_c = inner_left + inner_w // 2
                # avoid writing inside enclosed area
                enclosed_now = compute_enclosed(grid)
                for c in range(inner_left, inner_left + inner_w):
                    if enclosed_now[center_r, c]:
                        return True
                for r in range(inner_top, inner_top + inner_h):
                    if enclosed_now[r, center_c]:
                        return True
                grid[center_r, inner_left:inner_left + inner_w] = color1
                grid[inner_top:inner_top + inner_h, center_c] = color1
            return True

        # We'll try to place 1-3 enclosures (favoring partial 2- or 3-sided enclosures)
        nr_enclosures = random.randint(1, 3)
        attempts = 0
        placed = 0
        partial_target = random.randint(1, max(1, nr_enclosures))  # ensure at least one partial enclosure
        partial_placed = 0
        while placed < nr_enclosures and attempts < 120:
            attempts += 1
            h = random.randint(3, min(rows - 2, 8))
            w = random.randint(3, min(cols - 2, 8))
            top = random.randint(0, max(0, rows - h))
            left = random.randint(0, max(0, cols - w))
            # bias: 30% full ring, 30% irregular, 40% partial
            choice = random.random()
            ok = False
            if choice < 0.3:
                ok = place_rectangle_ring(top, left, h, w)
            elif choice < 0.6:
                ok = place_irregular_enclosure(top, left, h, w)
            else:
                # partial: choose 1-2 sides to be open (so enclosure on 2 or 3 sides)
                sides = ['top', 'bottom', 'left', 'right']
                n_open = random.choice([1, 2])
                open_sides = tuple(random.sample(sides, n_open))
                ok = place_partial_enclosure(top, left, h, w, open_sides)
                if ok:
                    partial_placed += 1
            if ok:
                placed += 1

        # If we failed to place any partial enclosure, try explicitly to add one
        if partial_placed < 1:
            attempts = 0
            while partial_placed < 1 and attempts < 100:
                attempts += 1
                h = random.randint(3, min(rows - 2, 8))
                w = random.randint(3, min(cols - 2, 8))
                top = random.randint(0, max(0, rows - h))
                left = random.randint(0, max(0, cols - w))
                sides = ['top', 'bottom', 'left', 'right']
                n_open = random.choice([1, 2])
                open_sides = tuple(random.sample(sides, n_open))
                if place_partial_enclosure(top, left, h, w, open_sides):
                    partial_placed += 1

        # add small 3x3 rings (1x1 holes) while avoiding enclosed interiors
        for _ in range(random.randint(0, 2)):
            if rows < 3 or cols < 3:
                break
            top = random.randint(0, rows - 3)
            left = random.randint(0, cols - 3)
            place_rectangle_ring(top, left, 3, 3)

        # recompute enclosed mask and avoid scattering blobs inside enclosed regions
        enclosed_mask = compute_enclosed(grid)

        def is_adjacent_to_color1(rr, cc):
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = rr+dr, cc+dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == color1:
                    return True
            return False

        # scatter a larger number of isolated single color1 cells (singletons),
        # avoiding enclosed holes and avoiding adjacency to other color1 cells so they remain 1x1
        n_singletons = random.randint(3, 7)
        for _ in range(n_singletons):
            tries = 0
            placed_single = False
            while tries < 60:
                tries += 1
                r = random.randint(0, rows - 1)
                c = random.randint(0, cols - 1)
                if enclosed_mask[r, c]:
                    continue
                if grid[r, c] != 0:
                    continue
                if is_adjacent_to_color1(r, c):
                    continue
                grid[r, c] = color1
                placed_single = True
                break
            # if couldn't find an isolated cell, relax adjacency constraint once
            if not placed_single:
                tries = 0
                while tries < 30:
                    tries += 1
                    r = random.randint(0, rows - 1)
                    c = random.randint(0, cols - 1)
                    if enclosed_mask[r, c] or grid[r, c] != 0:
                        continue
                    grid[r, c] = color1
                    break

        # ensure at least one fully enclosed empty region exists
        enclosed_mask = compute_enclosed(grid)
        if not enclosed_mask.any():
            # try placing rings on temporary copies until one produces an enclosed region
            success = False
            for _ in range(200):
                temp = grid.copy()
                h = random.randint(3, min(rows - 2, 8))
                w = random.randint(3, min(cols - 2, 8))
                top = random.randint(0, max(0, rows - h))
                left = random.randint(0, max(0, cols - w))
                # draw perimeter on temp
                r0, c0 = top, left
                r1, c1 = top + h - 1, left + w - 1
                for c in range(c0, c1 + 1):
                    temp[r0, c] = color1
                    temp[r1, c] = color1
                for r in range(r0, r1 + 1):
                    temp[r, c0] = color1
                    temp[r, c1] = color1

                new_enclosed = compute_enclosed(temp)
                if new_enclosed.any():
                    grid[:] = temp
                    enclosed_mask = new_enclosed
                    success = True
                    break

            # fallback: if still no enclosed region, and grid is large enough, place central 3x3 ring
            if not success and rows >= 3 and cols >= 3:
                top = (rows - 3) // 2
                left = (cols - 3) // 2
                r0, c0 = top, left
                r1, c1 = top + 2, left + 2
                for c in range(c0, c1 + 1):
                    grid[r0, c] = color1
                    grid[r1, c] = color1
                for r in range(r0, r1 + 1):
                    grid[r, c0] = color1
                    grid[r, c1] = color1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Recolor all zero cells that are not reachable from the grid border (i.e. enclosed holes)
        with taskvars['cell_color2'].
        """
        out = grid.copy()
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        # BFS/DFS from border zeros to mark reachable empty cells
        stack = []
        for r in range(rows):
            for c in (0, cols - 1):
                if grid[r, c] == 0:
                    stack.append((r, c))
        for c in range(cols):
            for r in (0, rows - 1):
                if grid[r, c] == 0:
                    stack.append((r, c))

        while stack:
            r, c = stack.pop()
            if visited[r, c]:
                continue
            if grid[r, c] != 0:
                continue
            visited[r, c] = True
            # neighbors 4-way
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc] and grid[nr, nc] == 0:
                    stack.append((nr, nc))

        # enclosed zeros are zeros not visited
        enclosed = (grid == 0) & (~visited)
        out[enclosed] = taskvars['cell_color2']

        return out



