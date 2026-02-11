from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskWRuL6EJHB7kpStmXyCaTtxGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid contains a 2x2 {color('object_color')} square object and a single {color('cell_color')} cell, while all remaining cells are empty (0).",
            "The {color('object_color')} object and the single {color('cell_color')} cell are separated by empty (0) cells, with one positioned in the top half of the grid and the other in the bottom half of the grid."
        ]
        transformation_reasoning_chain = [
            "The output grids are created by relocating the {color('object_color')} object and the single {color('cell_color')} cell so that they are vertically connected and aligned with the top-left corner of the grid.",
            "Their placement order in the output grid is determined by their position in the input grid.",
            "If the 2x2 object is in the top half of the input grid, it is placed above the single cell and aligned with the top-left corner, starting from (0,0).",
            "Otherwise, the single cell is positioned at (0,0), with the 2x2 object placed directly below it, starting at (1,0)."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars, force_top_object=False, force_bottom_object=False) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        object_color = taskvars['object_color']
        cell_color = taskvars['cell_color']
        grid = np.zeros((rows, cols), dtype=int)

        top_half_max_row = rows // 2
        bottom_half_min_row = rows // 2

        if force_top_object:
            place_block_top = True
        elif force_bottom_object:
            place_block_top = False
        else:
            place_block_top = random.choice([True, False])

        if place_block_top:
            block_row = random.randint(0, max(0, top_half_max_row - 2))
        else:
            block_row = random.randint(bottom_half_min_row, rows - 2)

        block_col = random.randint(0, max(0, cols - 2))
        grid[block_row:block_row + 2, block_col:block_col + 2] = object_color

        if place_block_top:
            cell_row = random.randint(bottom_half_min_row, rows - 1)
        else:
            cell_row = random.randint(0, top_half_max_row - 1)

        cell_col = random.randint(0, cols - 1)
        grid[cell_row, cell_col] = cell_color
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        object_color = taskvars['object_color']
        cell_color = taskvars['cell_color']

        object_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == object_color]
        cell_coords = [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == cell_color]
        
        if not object_coords or not cell_coords:
            return grid.copy()

        block_min_row = min(r for r, _ in object_coords)
        cell_row, cell_col = cell_coords[0]

        top_half_max_row = rows // 2
        block_is_top = (block_min_row < top_half_max_row)

        out_grid = np.zeros((rows, cols), dtype=int)

        if block_is_top:
            out_grid[0:2, 0:2] = object_color
            out_grid[2, 0] = cell_color
        else:
            out_grid[0, 0] = cell_color
            out_grid[1:3, 0:2] = object_color

        return out_grid

    def create_grids(self):
        rows = random.randint(8, 30)
        cols = random.randint(8, 30)
        object_color = random.randint(1, 9)
        while True:
            cell_color_candidate = random.randint(1, 9)
            if cell_color_candidate != object_color:
                cell_color = cell_color_candidate
                break

        taskvars = {
            'rows': rows,
            'cols': cols,
            'object_color': object_color,
            'cell_color': cell_color
        }

        nr_train = random.choice([3, 4])
        nr_test = 2  # Ensuring 2 test cases with different configurations

        train_data = []
        test_data = []

        for _ in range(nr_train):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            train_data.append(GridPair(input=inp, output=out))

        # Ensuring test cases have both configurations
        test_inputs = [
            self.create_input(taskvars, {}, force_top_object=True),  # Object in top half
            self.create_input(taskvars, {}, force_bottom_object=True)  # Object in bottom half
        ]

        for inp in test_inputs:
            out = self.transform_input(inp, taskvars)
            test_data.append(GridPair(input=inp, output=out))

        data: TrainTestData = {
            'train': train_data,
            'test': test_data
        }
        return taskvars, data



