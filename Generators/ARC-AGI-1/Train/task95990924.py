from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.transformation_library import GridObject, find_connected_objects


class Task95990924Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is an otherwise {color('background_color')} grid containing one or more separate {color('block_color')} blocks.",
            "2. Every block is a solid {vars['block_size']} by {vars['block_size']} square.",
            "3. Blocks may occur near boundaries, so some diagonal corner locations can lie outside the grid.",
            "4. Grid size, block count, positions, and clipping regimes vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Find every {color('block_color')} {vars['block_size']} by {vars['block_size']} block.",
            "2. At offset {vars['corner_offset']} beyond its northwest and northeast corners, paint {color('northwest_color')} and {color('northeast_color')}.",
            "3. At the same offset beyond its southwest and southeast corners, paint {color('southwest_color')} and {color('southeast_color')}.",
            "4. Clip out-of-bounds corners, never overwrite a {color('block_color')} block, and preserve all other cells.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        colors = random.sample(range(1, 10), 6)
        taskvars = {
            "background_color": 0,
            "block_color": colors[0],
            "block_size": 2,
            "corner_offset": 1,
            "northwest_color": colors[1],
            "northeast_color": colors[2],
            "southwest_color": colors[3],
            "southeast_color": colors[4],
        }
        spread_a = random.randint(0, 3)
        spread_b = random.randint(0, 2)
        layouts = [
            {"rows": 10, "cols": 10, "positions": [(3, 3)]},
            {"rows": 10, "cols": 10, "positions": [(0, 3)]},
            {"rows": 13, "cols": 13, "positions": [(3, 0), (7 + spread_a, 8)]},
            {"rows": 15, "cols": 15, "positions": [(1, 1), (10 + spread_b, 10), (5, 4 + spread_a)]},
        ]
        test_layout = {
            "rows": 14,
            "cols": 14,
            "positions": [(0, 1), (12, 10), (4, 0), (7, 12)],
        }

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {"input": input_grid, "output": self.transform_input(input_grid, taskvars)}

        return taskvars, {
            "train": [make_pair(layout) for layout in layouts],
            "test": [make_pair(test_layout)],
        }

    def create_input(self, taskvars, gridvars):
        grid = np.full(
            (gridvars["rows"], gridvars["cols"]),
            taskvars["background_color"],
            dtype=int,
        )
        block = np.full(
            (taskvars["block_size"], taskvars["block_size"]),
            taskvars["block_color"],
            dtype=int,
        )
        for position in gridvars["positions"]:
            GridObject.from_array(block, offset=position).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        block_color = taskvars["block_color"]
        offset = taskvars["corner_offset"]
        output = grid.copy()
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        ).with_color(block_color)
        for obj in objects:
            box = obj.bounding_box
            top = box[0].start
            bottom = box[0].stop - 1
            left = box[1].start
            right = box[1].stop - 1
            corners = [
                (top - offset, left - offset, taskvars["northwest_color"]),
                (top - offset, right + offset, taskvars["northeast_color"]),
                (bottom + offset, left - offset, taskvars["southwest_color"]),
                (bottom + offset, right + offset, taskvars["southeast_color"]),
            ]
            for row, col, color in corners:
                if (
                    0 <= row < output.shape[0]
                    and 0 <= col < output.shape[1]
                    and output[row, col] == background
                ):
                    output[row, col] = color
        return output
