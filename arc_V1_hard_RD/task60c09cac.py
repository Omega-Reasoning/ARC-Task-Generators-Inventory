from arc_task_generator import ARCTaskGenerator, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random

class Task60c09cacGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares of size {vars['rows']} by {vars['cols']}.",
            "Every input grid has many joint L-shaped patterns in various orientations and styles."
        ]
        transformation_reasoning_chain = [
            "The output grid is double the size of the input grid.",
            "Each input cell becomes a 2×2 block of the same color in the output."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_l_shape(self, size: int, color: int) -> np.ndarray:
        """Generate a single L-shape of given size and color."""
        l = np.zeros((size, size), dtype=int)
        half = size // 2
        v = random.randint(1, 4)
        if v == 1:  # bottom-left
            l[size-1, :half+1] = color
            l[half:, 0] = color
        elif v == 2:  # bottom-right
            l[size-1, half:] = color
            l[half:, size-1] = color
        elif v == 3:  # top-left
            l[0, :half+1] = color
            l[:half+1, 0] = color
        else:       # top-right
            l[0, half:] = color
            l[:half+1, size-1] = color
        return l

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Place 2–4 random L-shapes (and occasional noise) into a zero grid."""
        n = taskvars['rows']
        grid = np.zeros((n, n), dtype=int)
        shapes = random.randint(2, 4)
        colors = random.sample(range(1, 10), shapes)

        for color in colors:
            size = random.randint(3, min(6, n//2))
            l = self.create_l_shape(size, color)

            placed = False
            for _ in range(50):
                r0 = random.randint(0, n-size)
                c0 = random.randint(0, n-size)
                area = grid[r0:r0+size, c0:c0+size]
                if np.count_nonzero(area) < size//2:
                    for i in range(size):
                        for j in range(size):
                            if l[i,j]:
                                grid[r0+i, c0+j] = color
                    placed = True
                    break
            if not placed:
                r0 = random.randint(0, n-size)
                c0 = random.randint(0, n-size)
                grid[r0:r0+size, c0:c0+size][l!=0] = l[l!=0]

        # 30% chance of sparse noise
        if random.random() < 0.3:
            noise_color = random.randint(1,9)
            random_cell_coloring(grid, noise_color, density=0.05, overwrite=False)

        return grid

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        """Double every cell into a 2×2 block, yielding a grid twice as wide and tall."""
        rows, cols = grid.shape
        out = np.zeros((rows*2, cols*2), dtype=int)
        for r in range(rows):
            for c in range(cols):
                out[r*2:r*2+2, c*2:c*2+2] = grid[r, c]
        return out

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Generate 3–5 train examples and 1 test example with consistent size."""
        size = random.randint(8, 12)
        taskvars = {'rows': size, 'cols': size}

        train = []
        for _ in range(random.randint(3,5)):
            inp = self.create_input(taskvars, {})
            train.append({'input': inp, 'output': self.transform_input(inp)})

        inp = self.create_input(taskvars, {})
        test = [{'input': inp, 'output': self.transform_input(inp)}]

        return taskvars, {'train': train, 'test': test}


if __name__ == "__main__":
    gen = LShapeDoublingTaskGenerator()
    vars, data = gen.create_grids()
    print("Task vars:", vars)
    print("Train examples:", len(data['train']))
    print("Test examples:", len(data['test']))
    try:
        ARCTaskGenerator.visualize_train_test_data(data)
    except:
        pass
