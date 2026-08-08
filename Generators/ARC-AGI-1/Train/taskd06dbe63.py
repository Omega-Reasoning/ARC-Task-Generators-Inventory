from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Taskd06dbe63Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['rows']} by {vars['cols']} field of {color('background_color')}.",
            "2. Exactly one cell is colored {color('marker_color')} and acts as a path anchor.",
            "3. The marker position varies between examples and can be off center.",
            "4. There are no other foreground objects or path cells in the input.",
        ]
        transformation_reasoning_chain = [
            "1. Locate the single {color('marker_color')} anchor and preserve it.",
            "2. Draw a {color('path_color')} branch by alternating {vars['segment_length']}-cell upward and rightward segments from the marker.",
            "3. Draw the opposite branch by alternating {vars['segment_length']}-cell downward and leftward segments from the marker.",
            "4. Stop each branch once its continuing path lies outside the grid and leave every other cell {color('background_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        marker_color, path_color = random.sample(range(1, 10), 2)
        rows = random.randint(11, 17)
        cols = random.randint(11, 17)
        taskvars = {
            "rows": rows,
            "cols": cols,
            "background_color": 0,
            "marker_color": marker_color,
            "path_color": path_color,
            "segment_length": random.choice([2, 3]),
        }
        train_gridvars = [
            {"position": (rows // 3, cols // 3)},
            {"position": (rows // 3, 2 * cols // 3)},
            {"position": (2 * rows // 3, cols // 3)},
            {"position": (rows // 2, cols // 2)},
        ]
        test_gridvars = {"position": (max(2, rows // 4), max(2, 3 * cols // 4))}

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
        grid = np.full(
            (taskvars["rows"], taskvars["cols"]),
            taskvars["background_color"],
            dtype=int,
        )
        marker = np.asarray(
            gridvars.get(
                "marker",
                create_object(
                    1,
                    1,
                    taskvars["marker_color"],
                    contiguity=Contiguity.FOUR,
                    background=0,
                ),
            ),
            dtype=int,
        )
        if marker.shape != (1, 1) or marker[0, 0] != taskvars["marker_color"]:
            raise ValueError("marker must be the task's singleton marker cell")
        GridObject.from_array(marker, offset=gridvars["position"]).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        markers = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        ).with_color(taskvars["marker_color"])
        if len(markers) != 1:
            return grid.copy()
        marker = markers[0]
        start_row, start_col, _ = next(iter(marker.cells))
        segment_length = taskvars["segment_length"]
        path_color = taskvars["path_color"]
        path_cells = set()
        for vertical, horizontal in ((-1, 1), (1, -1)):
            row, col = start_row, start_col
            while 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
                stopped = False
                for _ in range(segment_length):
                    row += vertical
                    if not (0 <= row < grid.shape[0]):
                        stopped = True
                        break
                    path_cells.add((row, col, path_color))
                if stopped:
                    break
                for _ in range(segment_length):
                    col += horizontal
                    if not (0 <= col < grid.shape[1]):
                        stopped = True
                        break
                    path_cells.add((row, col, path_color))
                if stopped:
                    break
        output = grid.copy()
        GridObject(path_cells).paste(output)
        return output
