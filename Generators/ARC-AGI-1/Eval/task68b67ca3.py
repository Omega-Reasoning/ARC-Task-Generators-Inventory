import random
import numpy as np
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

class Task68b67ca3Generator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "Input grids are larger squares than output grids, exactly twice their size minus one.",
            "Input grid is created by taking a small pattern and separating its cells with empty rows and columns.",
            "Removing those empty rows and columns reconstructs the output pattern in the output grid."
        ]
        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids are squares of the same sizes {vars['rows']}, {vars['columns']}.",
            "This grid contains a small random connected shape with some repeating and some unique colors, surrounded by empty cells."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        full = gridvars['full_output']
        rows, cols = full.shape
        inp = np.zeros((2*rows-1, 2*cols-1), dtype=int)
        for r in range(rows):
            for c in range(cols):
                inp[2*r, 2*c] = full[r, c]
        return inp

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        return grid[::2, ::2]

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        size = random.randint(6, 10)
        taskvars = {'rows': size, 'columns': size}
        rows = cols = size

        def random_shape(max_cells=8):
            k = random.randint(4, max_cells)
            r0, c0 = rows//2, cols//2
            shape = {(r0, c0)}
            while len(shape) < k:
                r, c = random.choice(tuple(shape))
                dr, dc = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    shape.add((nr, nc))
            return list(shape)

        train_examples: list[GridPair] = []
        for _ in range(random.randint(3,5)):
            full = np.zeros((rows, cols), dtype=int)
            coords = random_shape()
            k = len(coords)
            base = random.sample(range(1,10), k-1)
            dup = random.choice(base)
            colors = base + [dup]
            random.shuffle(colors)
            for (r, c), color in zip(coords, colors):
                full[r, c] = color
            inp = self.create_input(taskvars, {'full_output': full})
            train_examples.append(GridPair(input=inp, output=full))

        full = np.zeros((rows, cols), dtype=int)
        coords = random_shape()
        k = len(coords)
        base = random.sample(range(1,10), k-1)
        dup = random.choice(base)
        colors = base + [dup]
        random.shuffle(colors)
        for (r, c), color in zip(coords, colors):
            full[r, c] = color
        test_inp = self.create_input(taskvars, {'full_output': full})
        test_examples: list[GridPair] = [GridPair(input=test_inp, output=full)]

        return taskvars, {'train': train_examples, 'test': test_examples}
