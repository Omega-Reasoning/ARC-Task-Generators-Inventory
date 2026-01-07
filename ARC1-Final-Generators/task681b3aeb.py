from arc_task_generator import ARCTaskGenerator
import numpy as np
import random
from transformation_library import find_connected_objects


class Task681b3aebGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain exactly two objects, where each object is made of 4-way connected cells of the same color, and all remaining cells are empty (0).",
            "The objects are formed by first creating an imaginary {vars['block_size']}x{vars['block_size']} grid completely filled with exactly two different colored objects.",
            "Once this imaginary grid is created, both colored objects are randomly placed within the actual grid.",
            "The shapes and sizes of both objects are different from each other but strictly such that, when connected, they form a completely colored {vars['block_size']}x{vars['block_size']} grid."
        ]

        transformation_reasoning_chain = [
            "The output grids are of size {vars['block_size']}x{vars['block_size']}.",
            "They are constructed by identifying the two colored objects in the input grids.",
            "Once identified, both objects are placed into the output grid and are connected such that they form a completely filled grid with no empty (0) cells remaining.",
            "The positioning ensures that both objects fit together perfectly with no truncation."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        grid_size = random.randint(7, 30)
        block_size = random.choice([3, 4, 5, 6])

        taskvars = {
            'grid_size': grid_size,
            'block_size': block_size
        }

        train_count = random.randint(3, 4)
        train_data = []

        for _ in range(train_count):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            train_data.append({'input': inp, 'output': out})

        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)

        return taskvars, {
            'train': train_data,
            'test': [{'input': test_input, 'output': test_output}]
        }

    # ---------------------------------------------------------------------

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        block_size = taskvars['block_size']

        grid = np.zeros((grid_size, grid_size), dtype=int)
        colors = random.sample(range(1, 10), 2)

        # --- generate a valid 2-object partition of an NxN block ---
        while True:
            template = np.zeros((block_size, block_size), dtype=int)

            # random connected seed
            all_cells = [(r, c) for r in range(block_size) for c in range(block_size)]
            random.shuffle(all_cells)

            split_point = random.randint(block_size**2 // 3, 2 * block_size**2 // 3)
            cells1 = set(all_cells[:split_point])
            cells2 = set(all_cells[split_point:])

            if self._is_connected(cells1) and self._is_connected(cells2):
                for r, c in cells1:
                    template[r, c] = 1
                for r, c in cells2:
                    template[r, c] = 2
                break

        # random symmetry
        template = np.rot90(template, k=random.randint(0, 3))
        if random.choice([True, False]):
            template = 3 - template

        obj1 = {(r, c) for r in range(block_size) for c in range(block_size) if template[r, c] == 1}
        obj2 = {(r, c) for r in range(block_size) for c in range(block_size) if template[r, c] == 2}

        # place objects far apart
        for _ in range(200):
            r1, c1 = random.randint(0, grid_size - block_size), random.randint(0, grid_size - block_size)
            r2, c2 = random.randint(0, grid_size - block_size), random.randint(0, grid_size - block_size)

            obj1_cells = {(r + r1, c + c1) for r, c in obj1}
            obj2_cells = {(r + r2, c + c2) for r, c in obj2}

            if obj1_cells & obj2_cells:
                continue

            if self._are_adjacent(obj1_cells, obj2_cells):
                continue

            temp = np.zeros_like(grid)
            for r, c in obj1_cells:
                temp[r, c] = colors[0]
            for r, c in obj2_cells:
                temp[r, c] = colors[1]

            if len(find_connected_objects(temp, diagonal_connectivity=False, background=0)) == 2:
                return temp

        return self.create_input(taskvars, gridvars)

    # ---------------------------------------------------------------------

    def transform_input(self, grid, taskvars):
        block_size = taskvars['block_size']
        output = np.zeros((block_size, block_size), dtype=int)

        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        if len(objects) != 2:
            raise ValueError("Expected exactly two objects")

        arrays = []
        for obj in objects:
            arr = obj.to_array()
            col = list(obj.colors)[0]
            filled = np.zeros_like(arr)
            filled[arr > 0] = col
            arrays.append((filled, col))

        for r1 in range(4):
            for r2 in range(4):
                a1 = np.rot90(arrays[0][0], r1)
                a2 = np.rot90(arrays[1][0], r2)
                c1, c2 = arrays[0][1], arrays[1][1]

                for i1 in range(-a1.shape[0] + 1, block_size):
                    for j1 in range(-a1.shape[1] + 1, block_size):
                        for i2 in range(-a2.shape[0] + 1, block_size):
                            for j2 in range(-a2.shape[1] + 1, block_size):
                                test = np.zeros_like(output)
                                if not self._place(test, a1, c1, i1, j1):
                                    continue
                                if not self._place(test, a2, c2, i2, j2):
                                    continue
                                if np.all(test > 0):
                                    return test

        raise ValueError("Could not reconstruct full block")

    # ---------------------------------------------------------------------

    @staticmethod
    def _place(grid, obj, color, r0, c0):
        for r in range(obj.shape[0]):
            for c in range(obj.shape[1]):
                if obj[r, c] > 0:
                    rr, cc = r0 + r, c0 + c
                    if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                        return False
                    if grid[rr, cc] != 0:
                        return False
                    grid[rr, cc] = color
        return True

    @staticmethod
    def _is_connected(cells):
        visited = {next(iter(cells))}
        stack = list(visited)
        while stack:
            r, c = stack.pop()
            for nr, nc in [(r+1,c), (r-1,c), (r,c+1), (r,c-1)]:
                if (nr, nc) in cells and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    stack.append((nr, nc))
        return len(visited) == len(cells)

    @staticmethod
    def _are_adjacent(a, b):
        for r1, c1 in a:
            for r2, c2 in b:
                if abs(r1-r2) + abs(c1-c2) == 1:
                    return True
        return False
