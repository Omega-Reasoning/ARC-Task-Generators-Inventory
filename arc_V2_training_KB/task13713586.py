from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task13713586Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have different sizes.",
            "Each grid contains exactly one {color('strip_color')} strip that is either horizontal or vertical, created by completely filling either the first or last row or column respectively.",
            "If the first or last row is filled, then there are {vars['no_of_strips']} additional horizontal strips, each one cell wide and shorter in length the closer they are to the main {color('strip_color')} strip.",
            "If the first or last column is filled, then there are {vars['no_of_strips']} additional vertical strips, each one cell wide and shorter in length the closer they are to the main {color('strip_color')} strip.",
            "These additional strips are completely separated from each other by empty cells (0).",
            "All additional strips must be differently colored from each other.",
            "All remaining cells in the grid are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the {color('strip_color')} strip and the additional colored strips.",
            "All additional colored strips are expanded towards the {color('strip_color')} strip using the same color as the strip being expanded, until they reach the {color('strip_color')} strip.",
            "If the {color('strip_color')} strip is vertical, the additional strips are expanded vertically; otherwise, they are expanded horizontally.",
            "The expansion results in rectangular blocks. In case of overlap, smaller blocks take priority and are not overwritten by larger blocks.",
            "The transformation does not affect the {color('strip_color')} strip itself."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        height = gridvars['height']
        width = gridvars['width']
        strip_color = taskvars['strip_color']
        no_of_strips = gridvars['no_of_strips']
        strip_position = gridvars['strip_position']
        strip_colors = gridvars['strip_colors']

        grid = np.zeros((height, width), dtype=int)

        if strip_position == 'first_row':
            grid[0, :] = strip_color
            is_horizontal = True
        elif strip_position == 'last_row':
            grid[-1, :] = strip_color
            is_horizontal = True
        elif strip_position == 'first_col':
            grid[:, 0] = strip_color
            is_horizontal = False
        else:
            grid[:, -1] = strip_color
            is_horizontal = False

        strips_created = 0
        max_length = 5
        min_length = 2
        length_range = max_length - min_length + 1
        total_strips = min(no_of_strips, len(strip_colors))

        if is_horizontal:
            available_rows = list(range(2, height - 1)) if strip_position == 'first_row' else list(range(1, height - 2))
            spaced_rows = []
            for i, row in enumerate(available_rows):
                if i == 0 or row - spaced_rows[-1] >= 2:
                    spaced_rows.append(row)
            if strip_position == 'first_row':
                spaced_rows.sort(key=lambda r: r)
            else:
                spaced_rows.sort(key=lambda r: -r)

            for i, row in enumerate(spaced_rows[:total_strips]):
                relative_index = i / max(1, total_strips - 1)
                strip_length = min_length + int(relative_index * (length_range - 1))
                strip_length = min(strip_length, width - 2)
                max_start = max(0, width - strip_length)
                attempts = 0
                placed = False
                while attempts < 20 and not placed:
                    start_col = random.randint(1, max(1, max_start - 1))
                    buffer_start = max(0, start_col - 1)
                    buffer_end = min(width, start_col + strip_length + 1)
                    if np.all(grid[row, buffer_start:buffer_end] == 0):
                        grid[row, start_col:start_col + strip_length] = strip_colors[i]
                        strips_created += 1
                        placed = True
                    attempts += 1
        else:
            available_cols = list(range(2, width - 1)) if strip_position == 'first_col' else list(range(1, width - 2))
            spaced_cols = []
            for i, col in enumerate(available_cols):
                if i == 0 or col - spaced_cols[-1] >= 2:
                    spaced_cols.append(col)
            if strip_position == 'first_col':
                spaced_cols.sort(key=lambda c: c)
            else:
                spaced_cols.sort(key=lambda c: -c)

            for i, col in enumerate(spaced_cols[:total_strips]):
                relative_index = i / max(1, total_strips - 1)
                strip_length = min_length + int(relative_index * (length_range - 1))
                strip_length = min(strip_length, height - 2)
                max_start = max(0, height - strip_length)
                attempts = 0
                placed = False
                while attempts < 20 and not placed:
                    start_row = random.randint(1, max(1, max_start - 1))
                    buffer_start = max(0, start_row - 1)
                    buffer_end = min(height, start_row + strip_length + 1)
                    if np.all(grid[buffer_start:buffer_end, col] == 0):
                        grid[start_row:start_row + strip_length, col] = strip_colors[i]
                        strips_created += 1
                        placed = True
                    attempts += 1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        strip_color = taskvars['strip_color']
        main_strip_pos = None
        is_horizontal = False

        if np.all(grid[0, :] == strip_color):
            main_strip_pos = 0
            is_horizontal = True
            strip_location = 'first_row'
        elif np.all(grid[-1, :] == strip_color):
            main_strip_pos = grid.shape[0] - 1
            is_horizontal = True
            strip_location = 'last_row'
        elif np.all(grid[:, 0] == strip_color):
            main_strip_pos = 0
            is_horizontal = False
            strip_location = 'first_col'
        elif np.all(grid[:, -1] == strip_color):
            main_strip_pos = grid.shape[1] - 1
            is_horizontal = False
            strip_location = 'last_col'

        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        additional_strips = [obj for obj in objects if strip_color not in obj.colors]
        additional_strips.sort(key=lambda x: len(x.cells))

        for strip_obj in additional_strips:
            strip_cells = list(strip_obj.cells)
            if not strip_cells:
                continue
            color = strip_cells[0][2]
            if is_horizontal:
                strip_rows = {r for r, c, col in strip_cells}
                strip_cols = {c for r, c, col in strip_cells}
                min_row = min(strip_rows)
                max_row = max(strip_rows)
                if strip_location == 'first_row':
                    for c in strip_cols:
                        for r in range(min_row - 1, -1, -1):
                            if output_grid[r, c] == 0:
                                output_grid[r, c] = color
                            else:
                                break
                else:
                    for c in strip_cols:
                        for r in range(max_row + 1, output_grid.shape[0]):
                            if output_grid[r, c] == 0:
                                output_grid[r, c] = color
                            else:
                                break
            else:
                strip_rows = {r for r, c, col in strip_cells}
                strip_cols = {c for r, c, col in strip_cells}
                min_col = min(strip_cols)
                max_col = max(strip_cols)
                if strip_location == 'first_col':
                    for r in strip_rows:
                        for c in range(min_col - 1, -1, -1):
                            if output_grid[r, c] == 0:
                                output_grid[r, c] = color
                            else:
                                break
                else:
                    for r in strip_rows:
                        for c in range(max_col + 1, output_grid.shape[1]):
                            if output_grid[r, c] == 0:
                                output_grid[r, c] = color
                            else:
                                break

        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        grid_sizes = [
            (random.randint(10, 16), random.randint(12, 18)),
            (random.randint(12, 18), random.randint(10, 16)),
            (random.randint(10, 16), random.randint(12, 18)),
            (random.randint(14, 20), random.randint(16, 22))
        ]

        min_dimension = min(min(h, w) for h, w in grid_sizes)
        no_of_strips = max(2, min_dimension // 3)
        strip_color = random.randint(1, 9)
        taskvars = {
            'strip_color': strip_color,
            'no_of_strips': no_of_strips
        }

        available_colors = [i for i in range(1, 10) if i != strip_color]
        train_configs = []

        train_configs.append({
            'height': grid_sizes[0][0],
            'width': grid_sizes[0][1],
            'strip_position': 'first_row',
            'no_of_strips': no_of_strips,
            'strip_colors': random.sample(available_colors, min(no_of_strips, len(available_colors)))
        })

        train_configs.append({
            'height': grid_sizes[1][0],
            'width': grid_sizes[1][1],
            'strip_position': 'last_col',
            'no_of_strips': no_of_strips,
            'strip_colors': random.sample(available_colors, min(no_of_strips, len(available_colors)))
        })

        third_position = random.choice(['last_row', 'first_col'])
        train_configs.append({
            'height': grid_sizes[2][0],
            'width': grid_sizes[2][1],
            'strip_position': third_position,
            'no_of_strips': no_of_strips,
            'strip_colors': random.sample(available_colors, min(no_of_strips, len(available_colors)))
        })

        used_positions = {config['strip_position'] for config in train_configs}
        all_positions = {'first_row', 'last_row', 'first_col', 'last_col'}
        remaining_positions = list(all_positions - used_positions)
        test_position = random.choice(remaining_positions) if remaining_positions else random.choice(list(all_positions))

        test_no_of_strips = no_of_strips + 1
        test_config = {
            'height': grid_sizes[3][0],
            'width': grid_sizes[3][1],
            'strip_position': test_position,
            'no_of_strips': test_no_of_strips,
            'strip_colors': random.sample(available_colors, min(test_no_of_strips, len(available_colors)))
        }

        train_pairs = []
        for config in train_configs:
            input_grid = self.create_input(taskvars, config)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})

        test_input = self.create_input(taskvars, test_config)
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_pairs, 'test': test_pairs}


