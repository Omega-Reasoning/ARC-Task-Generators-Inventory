from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Taskd22278a0Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a rectangular {color('background_color')} grid with two to four isolated colored seeds on its boundary.",
            "2. Every seed has a distinct color and seed locations can occupy the same side or different sides.",
            "3. Grid dimensions, seed count, colors, and boundary layout vary between examples.",
            "4. All non-seed cells are {color('background_color')}.",
        ]
        transformation_reasoning_chain = [
            "1. For every cell, find the seed minimizing Chebyshev distance, using Manhattan distance only to break Chebyshev ties.",
            "2. If multiple seeds remain tied after both distances, leave the cell {color('background_color')}.",
            "3. Otherwise color the cell with its owner only when the Chebyshev distance is divisible by {vars['ring_period']}.",
            "4. Leave the alternating intervening square rings {color('background_color')} and preserve every seed.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"background_color": 0, "ring_period": 2}
        train_gridvars = [
            {"rows": 10, "cols": 12, "positions": [(0, 0), (0, 11)]},
            {"rows": 12, "cols": 12, "positions": [(0, 11), (11, 0)]},
            {"rows": 13, "cols": 11, "positions": [(0, 0), (12, 0)]},
            {"rows": 9, "cols": 13, "positions": [(0, 0), (0, 12), (8, 0)]},
        ]
        test_gridvars = {
            "rows": 15,
            "cols": 17,
            "positions": [(0, 0), (0, 16), (14, 0), (14, 16)],
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
        rows = int(gridvars["rows"])
        cols = int(gridvars["cols"])
        positions = [tuple(map(int, value)) for value in gridvars["positions"]]
        if (
            not 2 <= len(positions) <= 4
            or len(set(positions)) != len(positions)
            or any(
                not (0 <= row < rows and 0 <= col < cols)
                or (row not in {0, rows - 1} and col not in {0, cols - 1})
                for row, col in positions
            )
        ):
            raise ValueError("positions must be distinct boundary cells")
        grid = np.full(
            (rows, cols),
            taskvars["background_color"],
            dtype=int,
        )
        colors = [
            int(value)
            for value in gridvars.get(
                "seed_colors",
                random.sample(range(1, 10), len(positions)),
            )
        ]
        if (
            len(colors) != len(positions)
            or len(set(colors)) != len(colors)
            or any(not 1 <= color <= 9 for color in colors)
        ):
            raise ValueError("seed_colors must be distinct nonzero ARC colors")

        def sampled_marker_colors():
            values = []
            for color in colors:
                marker = create_object(
                    1,
                    1,
                    color,
                    contiguity=Contiguity.FOUR,
                    background=0,
                )
                values.append(int(marker[0, 0]))
            return values

        marker_colors = [
            int(value)
            for value in gridvars.get("marker_colors", sampled_marker_colors())
        ]
        if marker_colors != colors:
            raise ValueError("one-cell markers must retain their seed colors")
        for position, color in zip(positions, marker_colors):
            marker = np.array([[color]], dtype=int)
            GridObject.from_array(marker, offset=position).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        )
        seeds = []
        for obj in objects:
            if obj.size == 1:
                row, col, color = next(iter(obj.cells))
                seeds.append((row, col, int(color)))
        output = np.full(grid.shape, background, dtype=int)
        if not seeds:
            return output
        ring_period = taskvars["ring_period"]
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                distances = [
                    (
                        max(abs(row - seed_row), abs(col - seed_col)),
                        abs(row - seed_row) + abs(col - seed_col),
                    )
                    for seed_row, seed_col, _ in seeds
                ]
                best = min(distances)
                winners = [
                    index
                    for index, distance in enumerate(distances)
                    if distance == best
                ]
                if len(winners) == 1 and best[0] % ring_period == 0:
                    output[row, col] = seeds[winners[0]][2]
        return output
