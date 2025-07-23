import random
import numpy as np
from arc_task_generator import ARCTaskGenerator, TrainTestData

class ScatterCornersTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "Input grids are squares of same sizes {vars['rows']}, {vars['columns']}.",
            "Every input grid has exactly one 2x2 block at the middle of the grid, so make sure you keep that in mind and create the grid sizes accordingly. So this 2x2 block has 4 cells in which each cell has a different color."
        ]
        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is the same size of the input grid.",
            "Transformation is very simple, just scatter those 4 cells on each corner of the grid with respect to their position."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['columns']
        # initialize background
        grid = np.zeros((rows, cols), dtype=int)
        # center 2x2 block start indices
        start_r = rows // 2 - 1
        start_c = cols // 2 - 1
        # choose 4 distinct non-zero colors
        colors = random.sample(list(range(1, 10)), 4)
        # place colors in the 2x2 central block
        grid[start_r,     start_c]     = colors[0]  # top-left
        grid[start_r,     start_c + 1] = colors[1]  # top-right
        grid[start_r + 1, start_c]     = colors[2]  # bottom-left
        grid[start_r + 1, start_c + 1] = colors[3]  # bottom-right
        return grid

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        rows, cols = grid.shape
        # initialize output grid
        output = np.zeros_like(grid)
        # compute center block positions
        mid_r = rows // 2
        mid_c = cols // 2
        # map each of the 4 central cells to its corresponding corner
        positions = [
            (mid_r - 1, mid_c - 1, 0,      0     ),  # top-left -> corner (0,0)
            (mid_r - 1, mid_c,     0,      cols-1),  # top-right -> corner (0,last)
            (mid_r,     mid_c - 1, rows-1, 0     ),  # bottom-left -> corner (last,0)
            (mid_r,     mid_c,     rows-1, cols-1)   # bottom-right -> corner (last,last)
        ]
        for r_src, c_src, r_dst, c_dst in positions:
            color = grid[r_src, c_src]
            output[r_dst, c_dst] = color
        return output

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # choose an even grid size between 4 and 10, invariant for all examples
        size = random.choice([4, 6, 8, 10])
        rows = size
        columns = size
        taskvars = {'rows': rows, 'columns': columns}

        # generate 3-5 training examples
        num_train = random.randint(3, 5)
        train_examples = []
        for _ in range(num_train):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp)
            train_examples.append({'input': inp, 'output': out})

        # generate one test example
        test_inp = self.create_input(taskvars, {})
        test_out = self.transform_input(test_inp)
        test_examples = [{'input': test_inp, 'output': test_out}]

        return taskvars, {'train': train_examples, 'test': test_examples}
