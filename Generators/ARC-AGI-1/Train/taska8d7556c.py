from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random


class Taska8d7556cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a noisy rectangular field of {color('field_color')} and empty cells.",
            "2. Hidden among the noise are all-empty rectangles at least {vars['minimum_side']} cells high and wide.",
            "3. Some maximal empty rectangles are disjoint, while an example may contain two orthogonally overlapping candidates.",
            "4. Canvas size, noise, and rectangle dimensions vary between examples."
        ]
        transformation_reasoning_chain = [
            "1. Enumerate all-empty rectangles with both dimensions at least {vars['minimum_side']}.",
            "2. Keep only maximal rectangles not strictly contained in another all-empty rectangle.",
            "3. Process maximal rectangles by decreasing area and reject any that overlaps an already selected rectangle.",
            "4. Recolor every selected rectangle {color('highlight_color')} and preserve all remaining cells."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        field_color, highlight_color = random.sample(range(1, 10), 2)
        taskvars = {
            'field_color': field_color,
            'highlight_color': highlight_color,
            'minimum_side': 2,
        }
        train_gridvars = [
            {'rows': 14, 'cols': 14, 'rectangles': [(1, 1, 2, 2), (8, 9, 3, 2)]},
            {'rows': 16, 'cols': 15, 'rectangles': [(2, 8, 2, 4), (10, 2, 2, 2)]},
            {'rows': 17, 'cols': 17, 'rectangles': [(4, 6, 4, 2), (5, 5, 2, 3)]},
            {'rows': 18, 'cols': 16, 'rectangles': [(2, 2, 3, 2), (12, 10, 2, 3)]},
        ]
        test_gridvars = {'rows': 19, 'cols': 18, 'rectangles': [(3, 11, 5, 2), (13, 2, 3, 3)]}
        train = []
        for gridvars in train_gridvars:
            input_grid = self.create_input(taskvars, gridvars)
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        test_input = self.create_input(taskvars, test_gridvars)
        return taskvars, {
            'train': train,
            'test': [{'input': test_input, 'output': self.transform_input(test_input, taskvars)}],
        }

    def create_input(self, taskvars, gridvars):
        rows, cols = gridvars['rows'], gridvars['cols']
        field_color = taskvars['field_color']
        target_cells = set()
        for row, col, height, width in gridvars['rectangles']:
            target_cells.update(
                (target_row, target_col)
                for target_row in range(row, row + height)
                for target_col in range(col, col + width)
            )
        collar_cells = set()
        for row, col in target_cells:
            for row_step in (-1, 0, 1):
                for col_step in (-1, 0, 1):
                    neighbor = (row + row_step, col + col_step)
                    if (
                        0 <= neighbor[0] < rows
                        and 0 <= neighbor[1] < cols
                        and neighbor not in target_cells
                    ):
                        collar_cells.add(neighbor)

        def sample_grid():
            candidate = np.zeros((rows, cols), dtype=int)
            random_cell_coloring(
                candidate,
                field_color,
                density=random.uniform(0.72, 0.86),
                background=0,
                overwrite=False,
            )
            for row, col in collar_cells:
                candidate[row, col] = field_color
            for row, col in target_cells:
                candidate[row, col] = 0
            return candidate

        def valid_grid(candidate):
            zero_fraction = float(np.mean(candidate == 0))
            if not (0.08 <= zero_fraction <= 0.35 and np.any(candidate == field_color)):
                return False
            for row in range(rows - 1):
                for col in range(cols - 1):
                    block = {
                        (row, col),
                        (row + 1, col),
                        (row, col + 1),
                        (row + 1, col + 1),
                    }
                    if np.all(candidate[row:row + 2, col:col + 2] == 0):
                        if not block.issubset(target_cells):
                            return False
            return True

        try:
            sampled_grid = retry(sample_grid, valid_grid, max_attempts=80)
        except ValueError:
            sampled_grid = np.full((rows, cols), field_color, dtype=int)
            for row, col in target_cells:
                sampled_grid[row, col] = 0

        noise_grid = np.asarray(
            gridvars.get("noise_grid", sampled_grid),
            dtype=int,
        )
        if noise_grid.shape != (rows, cols):
            raise ValueError("noise_grid must match the requested canvas dimensions")
        if not set(np.unique(noise_grid)).issubset({0, field_color}):
            raise ValueError("noise_grid may contain only background and field colors")
        return noise_grid.copy()

    def transform_input(self, grid, taskvars):
        highlight_color = taskvars['highlight_color']
        minimum_side = taskvars['minimum_side']
        rows, cols = grid.shape
        empty = (grid == 0).astype(int)
        prefix = np.pad(empty, ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)
        candidates = []
        for row0 in range(rows - minimum_side + 1):
            for row1 in range(row0 + minimum_side, rows + 1):
                for col0 in range(cols - minimum_side + 1):
                    for col1 in range(col0 + minimum_side, cols + 1):
                        area = (row1 - row0) * (col1 - col0)
                        count = (
                            prefix[row1, col1]
                            - prefix[row0, col1]
                            - prefix[row1, col0]
                            + prefix[row0, col0]
                        )
                        if int(count) == area:
                            candidates.append((row0, row1, col0, col1))
        maximal = []
        for candidate in candidates:
            row0, row1, col0, col1 = candidate
            contained = False
            for other in candidates:
                if other == candidate:
                    continue
                if (
                    other[0] <= row0
                    and other[1] >= row1
                    and other[2] <= col0
                    and other[3] >= col1
                ):
                    contained = True
                    break
            if not contained:
                maximal.append(candidate)
        maximal.sort(
            key=lambda item: (
                -((item[1] - item[0]) * (item[3] - item[2])),
                item[0],
                item[2],
                item[1],
                item[3],
            )
        )
        output = grid.copy()
        selected = np.zeros_like(grid, dtype=bool)
        for row0, row1, col0, col1 in maximal:
            if not np.any(selected[row0:row1, col0:col1]):
                output[row0:row1, col0:col1] = highlight_color
                selected[row0:row1, col0:col1] = True
        return output
