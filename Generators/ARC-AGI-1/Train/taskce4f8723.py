from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
import numpy as np
import random


class Taskce4f8723Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains two {vars['panel_rows']} by {vars['panel_cols']} panels separated by one complete row of {color('divider_color')}.",
            "2. The upper panel contains a scattered mask of {color('upper_color')} cells on {color('background_color')}.",
            "3. The lower panel contains an independently scattered mask of {color('lower_color')} cells on {color('background_color')}.",
            "4. Both masks are nonempty, structurally varied, and may overlap when corresponding coordinates are aligned.",
            "5. The panel masks vary between examples while the dimensions and role colors remain fixed within one episode.",
        ]
        transformation_reasoning_chain = [
            "1. Split the input at the full {color('divider_color')} divider row into its equal upper and lower panels.",
            "2. Align corresponding rows and columns of the lower panel directly with the upper panel without reflection or rotation.",
            "3. Mark every coordinate occupied in either aligned panel.",
            "4. Return a {vars['panel_rows']} by {vars['panel_cols']} grid whose marked cells are {color('output_color')} and whose other cells are {color('background_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        colors = random.sample(range(1, 10), 4)
        taskvars = {
            "panel_rows": random.randint(4, 7),
            "panel_cols": random.randint(4, 7),
            "background_color": 0,
            "upper_color": colors[0],
            "lower_color": colors[1],
            "divider_color": colors[2],
            "output_color": colors[3],
        }
        cells = taskvars["panel_rows"] * taskvars["panel_cols"]
        train_gridvars = [
            {"upper_range": (3, max(4, cells // 2)), "lower_range": (3, max(4, cells // 2))},
            {"upper_range": (max(4, cells // 2), cells - 2), "lower_range": (3, max(4, cells // 2))},
            {"upper_range": (3, max(4, cells // 2)), "lower_range": (max(4, cells // 2), cells - 2)},
            {"upper_range": (max(4, cells // 3), cells - 3), "lower_range": (max(4, cells // 3), cells - 3)},
        ]
        test_gridvars = {
            "upper_range": (max(5, cells // 3), cells - 2),
            "lower_range": (max(5, cells // 3), cells - 2),
        }

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        return taskvars, {
            "train": [make_pair(gridvars) for gridvars in train_gridvars],
            "test": [make_pair(test_gridvars)],
        }

    def create_input(self, taskvars, gridvars):
        rows = taskvars["panel_rows"]
        cols = taskvars["panel_cols"]
        background = taskvars["background_color"]

        def sample_masks():
            upper = create_object(
                rows,
                cols,
                1,
                contiguity=Contiguity.NONE,
                background=0,
            )
            lower = create_object(
                rows,
                cols,
                1,
                contiguity=Contiguity.NONE,
                background=0,
            )
            return upper, lower

        def valid_masks(masks):
            upper, lower = masks
            upper_count = int(np.count_nonzero(upper))
            lower_count = int(np.count_nonzero(lower))
            reflected_union = (upper != 0) | (np.flipud(lower) != 0)
            direct_union = (upper != 0) | (lower != 0)
            return bool(
                gridvars["upper_range"][0] <= upper_count <= gridvars["upper_range"][1]
                and gridvars["lower_range"][0] <= lower_count <= gridvars["lower_range"][1]
                and np.count_nonzero(reflected_union) < rows * cols
                and not np.array_equal(reflected_union, direct_union)
            )

        try:
            upper_mask, lower_mask = gridvars.get(
                "masks",
                retry(sample_masks, valid_masks, max_attempts=80),
            )
        except ValueError:
            upper_mask = np.zeros((rows, cols), dtype=int)
            lower_mask = np.zeros((rows, cols), dtype=int)
            for index in range(min(rows, cols)):
                upper_mask[index, index] = 1
                lower_mask[index, (index + 1) % cols] = 1
            upper_mask[0, cols - 1] = 1
            lower_mask[rows - 1, 0] = 1
            upper_mask, lower_mask = gridvars.get(
                "masks",
                (upper_mask, lower_mask),
            )
        upper_mask = np.asarray(upper_mask, dtype=int)
        lower_mask = np.asarray(lower_mask, dtype=int)

        grid = np.full((2 * rows + 1, cols), background, dtype=int)
        grid[:rows][upper_mask != 0] = taskvars["upper_color"]
        grid[rows, :] = taskvars["divider_color"]
        grid[rows + 1 :][lower_mask != 0] = taskvars["lower_color"]
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        divider = taskvars["divider_color"]
        divider_rows = np.where(np.all(grid == divider, axis=1))[0]
        if len(divider_rows) != 1:
            return grid.copy()
        split = int(divider_rows[0])
        upper = grid[:split]
        lower = grid[split + 1 :]
        if upper.shape != lower.shape:
            return grid.copy()
        occupied = (upper == taskvars["upper_color"]) | (
            lower == taskvars["lower_color"]
        )
        output = np.full(upper.shape, background, dtype=int)
        output[occupied] = taskvars["output_color"]
        return output
