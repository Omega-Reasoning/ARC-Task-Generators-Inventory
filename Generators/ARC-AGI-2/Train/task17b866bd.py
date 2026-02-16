from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class Task17b866bdGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
             "Input grids are of different sizes.",
    "Each input contains a repeating pattern made of flower-shaped structures: a central empty (0) tile surrounded by {color('grid_color')} cells, uniformly distributed across the grid.",
    "The grid is constructed by tiling this flower-shaped pattern from the top-left corner, till bottom-right corner.",
    "This results in several empty (0) cells outside the flower tiles, which are located at rows and columns that are multiples of 5 (including the (0, 0) cell).",
    "Once a fully uniform grid is formed, some of these empty cells are filled with different colors — including the {color('grid_color')} color itself."
]


        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all flower-shaped tiles: each consists of a central empty (0) tile surrounded by {color('grid_color')} cells, uniformly distributed across the grid.",
    
    "These tiles are created by tiling the flower-shaped pattern from the top-left corner to the bottom-right corner of the grid.",
    
    "Next, examine every row at column positions where i = 0, 5, 10, 15, 20, 25, and so on.",
    
    "If a cell at any of these positions contains a color (i.e., is non-zero), then the corresponding flower-shaped tile in the bottom-right direction of that position should have its empty (0) cells filled with the same color.",
    
    "This operation ensures that color information from special marker positions (in the first row or column) is propagated into their respective flower-shaped tiles.",
            "After all flower-shaped tiles have been filled based on the above conditions, the output grid removes (sets to 0) any colored cell that appears at tile-aligned positions (i, j) where both i and j are multiples of 5 — including (0,0) — and also remove any colored cell that was used as the (1,1) interior of a 3×3 block to trigger filling."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_base_pattern(self, grid_color: int) -> np.ndarray:
        return np.array([
            [0, grid_color, grid_color, grid_color, grid_color, 0],
            [grid_color, grid_color, 0, 0, grid_color, grid_color],
            [grid_color, 0, 0, 0, 0, grid_color],
            [grid_color, 0, 0, 0, 0, grid_color],
            [grid_color, grid_color, 0, 0, grid_color, grid_color],
            [0, grid_color, grid_color, grid_color, grid_color, 0]
        ])

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = gridvars['rows']
        cols = gridvars['cols']
        grid_color = taskvars['grid_color']
        base_pattern = self.create_base_pattern(grid_color)
        grid = np.zeros((rows, cols), dtype=int)

        for r in range(0, rows - 5, 5):
            for c in range(0, cols - 5, 5):
                grid[r:r+6, c:c+6] = base_pattern

        border_positions = []
        interior_positions = []

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    if r in range(0, rows, 5) and c == 0:
                        border_positions.append((r, c))
                    elif c in range(0, cols, 5) and r == 0:
                        border_positions.append((r, c))

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                block = grid[r-1:r+2, c-1:c+2]
                if block.shape == (3, 3) and (r, c) == (r, c):
                    count_grid_color = np.sum(block == grid_color)
                    if count_grid_color >= 7 and grid[r, c] == 0:
                        interior_positions.append((r, c))

        all_positions = interior_positions + border_positions
        if not all_positions:
            return grid

        num_to_color = min(random.randint(2, 5), len(all_positions))
        positions_to_color = random.sample(all_positions, num_to_color)

        available_colors = [c for c in range(1, 10) if c != grid_color]
        available_colors.append(grid_color)

        for (r, c) in positions_to_color:
            color = random.choice(available_colors)
            grid[r, c] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        rows, cols = grid.shape
        grid_color = taskvars['grid_color']

        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                block = grid[r-1:r+2, c-1:c+2]
                if block.shape == (3, 3) and grid[r, c] != 0 and np.sum(block == grid_color) >= 7:
                    r_base = (r // 5) * 5
                    c_base = (c // 5) * 5
                    if r_base + 6 <= rows and c_base + 6 <= cols:
                        for i in range(r_base, r_base + 6):
                            for j in range(c_base, c_base + 6):
                                if output_grid[i, j] == 0:
                                    output_grid[i, j] = grid[r, c]

        for j in range(0, cols, 5):
            if grid[0, j] != 0:
                r_base = 0
                c_base = (j // 5) * 5
                if r_base + 6 <= rows and c_base + 6 <= cols:
                    for i in range(r_base, r_base + 6):
                        for k in range(c_base, c_base + 6):
                            if output_grid[i, k] == 0:
                                output_grid[i, k] = grid[0, j]

        for i in range(0, rows, 5):
            if grid[i, 0] != 0:
                r_base = (i // 5) * 5
                c_base = 0
                if r_base + 6 <= rows and c_base + 6 <= cols:
                    for m in range(r_base, r_base + 6):
                        for n in range(c_base, c_base + 6):
                            if output_grid[m, n] == 0:
                                output_grid[m, n] = grid[i, 0]

        for i in range(0, rows, 5):
            for j in range(0, cols, 5):
                output_grid[i, j] = 0

        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        grid_color = random.choice(range(1, 10))
        valid_sizes = [11, 16, 21, 26]

        taskvars = {'grid_color': grid_color}
        train_examples = []

        for _ in range(3):
            rows = random.choice(valid_sizes)
            cols = random.choice(valid_sizes)
            gridvars = {'rows': rows, 'cols': cols}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        test_rows = random.choice(valid_sizes)
        test_cols = random.choice(valid_sizes)
        test_input = self.create_input(taskvars, {'rows': test_rows, 'cols': test_cols})
        test_output = self.transform_input(test_input, taskvars)

        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }

        return taskvars, train_test_data
