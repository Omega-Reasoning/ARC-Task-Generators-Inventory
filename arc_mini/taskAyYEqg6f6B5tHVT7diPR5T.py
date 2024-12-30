# my_arc_agi_task_generator.py

import numpy as np
import random

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, BorderBehavior

class TasktaskAyYEqg6f6B5tHVT7diPR5TGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input matrix contains a {vars['length']}x{vars['width']} rectangular object, 4-way connected cells of {color('cell_color1')} color, with the remaining cells being empty (0)."
        ]
        reasoning_chain = [
            "To construct the output matrix, move the {color('cell_color1')} rectangular object such that it touches the bottom-right corner of the matrix, while keeping its shape and size unchanged."
        ]
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        grid_nrows = random.randint(10, 30)
        grid_ncols = random.randint(10, 30)
        cell_color1 = random.randint(1, 9)
        rect_length = random.randint(3, 5)
        rect_width = random.randint(3, 5)

        taskvars = {
            'grid_nrows': grid_nrows,
            'grid_ncols': grid_ncols,
            'cell_color1': cell_color1,
            'length': rect_length,
            'width': rect_width
        }

        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1

        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        nrows = taskvars['grid_nrows']
        ncols = taskvars['grid_ncols']
        color = taskvars['cell_color1']
        rect_length = taskvars['length']
        rect_width = taskvars['width']

        grid = np.zeros((nrows, ncols), dtype=int)

        max_row_placement = nrows - rect_length - 1
        max_col_placement = ncols - rect_width - 1

        if max_row_placement < 0 or max_col_placement < 0:
            rmin = 0
            rmax = nrows - rect_length
            cmin = 0
            cmax = ncols - rect_width
        else:
            rmin = 0
            rmax = max_row_placement
            cmin = 0
            cmax = max_col_placement

        row_start = random.randint(rmin, rmax) if rmax >= rmin else 0
        col_start = random.randint(cmin, cmax) if cmax >= cmin else 0

        grid[row_start:row_start + rect_length, col_start:col_start + rect_width] = color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        out_grid = grid.copy()

        objects = find_connected_objects(out_grid, diagonal_connectivity=False, background=0)
        color_of_interest = taskvars['cell_color1']
        matching_objs = objects.with_color(color_of_interest)

        if len(matching_objs) == 0:
            return out_grid

        obj = matching_objs[0]

        nrows, ncols = out_grid.shape
        obj_bb = obj.bounding_box
        obj_bottom = obj_bb[0].stop - 1
        obj_right = obj_bb[1].stop - 1

        dx = (nrows - 1) - obj_bottom
        dy = (ncols - 1) - obj_right

        obj.cut(out_grid, background=0)
        obj.translate(dx, dy, border_behavior=BorderBehavior.CLIP, grid_shape=(nrows, ncols))
        obj.paste(out_grid, overwrite=True, background=0)

        return out_grid

