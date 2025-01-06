# checkerboard_task_generator.py
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TasktaskgiRuPb6jZtoUE6BGRXvQCdGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids can have different sizes.",
            "They only contain a single {color('object_color')} rectangular object, 4-way connected cells, surrounded by empty (0) cells."
        ]
        reasoning_chain = [
            "The output grid is constructed by copying the input grid and emptying (0) some of the {color('object_color')} cells to create a checkerboard pattern.",
            "The checkerboard pattern alternates between {color('object_color')} and empty (0) cells in each row and column.",
            "The checkerboard pattern must always have the top-left corner cell filled with {color('object_color')} color."
        ]
        super().__init__(observation_chain, reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        rows = gridvars['rows']
        cols = gridvars['cols']
        rect_row = gridvars['rect_row']
        rect_col = gridvars['rect_col']
        rect_h = gridvars['rect_h']
        rect_w = gridvars['rect_w']
        color = taskvars['object_color']
        grid = np.zeros((rows, cols), dtype=int)
        for r in range(rect_row, rect_row + rect_h):
            for c in range(rect_col, rect_col + rect_w):
                grid[r, c] = color
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        color = taskvars['object_color']
        out_grid = grid.copy()
        rows, cols = out_grid.shape
        min_r, min_c = rows, cols
        max_r, max_c = -1, -1
        for r in range(rows):
            for c in range(cols):
                if out_grid[r, c] == color:
                    min_r = min(min_r, r)
                    max_r = max(max_r, r)
                    min_c = min(min_c, c)
                    max_c = max(max_c, c)
        if min_r > max_r or min_c > max_c:
            return out_grid
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                row_offset = r - min_r
                col_offset = c - min_c
                if (row_offset + col_offset) % 2 == 0:
                    out_grid[r, c] = color
                else:
                    out_grid[r, c] = 0
        return out_grid

    def create_grids(self) -> tuple:
        taskvars = {}
        taskvars['object_color'] = random.randint(1, 9)
        nr_train_examples = random.randint(3, 6)
        used_sizes = set()
        train_pairs = []
        for _ in range(nr_train_examples):
            rows, cols = self._pick_unique_grid_size(used_sizes)
            gridvars = self._create_random_rect_in_grid(rows, cols)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        rows_test, cols_test = self._pick_unique_grid_size(used_sizes)
        gridvars_test = self._create_random_rect_in_grid(rows_test, cols_test)
        test_input = self.create_input(taskvars, gridvars_test)
        test_output = self.transform_input(test_input, taskvars)
        train_test_data = TrainTestData(
            train=train_pairs,
            test=[GridPair(input=test_input, output=test_output)]
        )
        return taskvars, train_test_data

    def _pick_unique_grid_size(self, used_sizes, min_dim=5, max_dim=30) -> tuple:
        while True:
            rows = random.randint(min_dim, max_dim)
            cols = random.randint(min_dim, max_dim)
            if (rows, cols) not in used_sizes:
                used_sizes.add((rows, cols))
                return rows, cols

    def _create_random_rect_in_grid(self, rows: int, cols: int) -> dict:
        max_rect_h = max(2, rows // 2)
        max_rect_w = max(2, cols // 2)
        rect_h = random.randint(2, max_rect_h if max_rect_h < rows else rows-1)
        rect_w = random.randint(2, max_rect_w if max_rect_w < cols else cols-1)
        rect_row = random.randint(0, rows - rect_h)
        rect_col = random.randint(0, cols - rect_w)
        return {
            'rows': rows,
            'cols': cols,
            'rect_row': rect_row,
            'rect_col': rect_col,
            'rect_h': rect_h,
            'rect_w': rect_w
        }

if __name__ == "__main__":
    generator = CheckerboardTaskGenerator()
    taskvars, train_test_data = generator.create_grids()
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)