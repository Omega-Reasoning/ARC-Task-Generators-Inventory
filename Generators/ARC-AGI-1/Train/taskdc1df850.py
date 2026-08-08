from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Taskdc1df850Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains one or more isolated {color('target_color')} target cells on an otherwise empty grid.",
            "2. Additional single cells of arbitrary nonzero colors may occur as distractors and must not be transformed.",
            "3. Targets may lie in the interior, on an edge, or in a corner, and two targets may have overlapping neighborhoods.",
            "4. Grid dimensions, target count, positions, and distractor colors vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Find every {color('target_color')} cell independently of all distractors.",
            "2. Color every in-bounds cell within {vars['radius']} row and column of a target {color('halo_color')}, combining overlapping neighborhoods.",
            "3. Restore each target center to {color('target_color')} and preserve every unrelated original colored cell.",
            "4. Keep the original grid dimensions and clip neighborhoods at the boundary.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        target_color, halo_color = random.sample(range(1, 10), 2)
        taskvars = {
            "target_color": target_color,
            "halo_color": halo_color,
            "radius": 1,
        }

        def make_pair(rows, cols, targets, distractor_count):
            input_grid = self.create_input(
                taskvars,
                {
                    "rows": rows,
                    "cols": cols,
                    "targets": targets,
                    "distractor_count": distractor_count,
                },
            )
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [
            make_pair(7, 7, [(3, 3)], 2),
            make_pair(8, 9, [(0, 0), (4, 6)], 2),
            make_pair(9, 9, [(4, 4), (4, 6)], 1),
            make_pair(8, 8, [(0, 5), (7, 2), (4, 0)], 2),
        ]
        test = [make_pair(10, 10, [(0, 0), (3, 4), (5, 5), (9, 8)], 3)]
        return taskvars, {"train": train, "test": test}

    def create_input(self, taskvars, gridvars):
        grid = np.zeros((gridvars["rows"], gridvars["cols"]), dtype=int)
        target = GridObject(
            {
                (row, col, taskvars["target_color"])
                for row, col in gridvars["targets"]
            }
        )
        target.paste(grid)
        distractor_colors = [
            color
            for color in range(1, 10)
            if color not in {taskvars["target_color"], taskvars["halo_color"]}
        ]
        for distractor_index in range(gridvars["distractor_count"]):
            try:
                sampled_row, sampled_col = retry(
                    lambda: (
                        random.randrange(grid.shape[0]),
                        random.randrange(grid.shape[1]),
                    ),
                    lambda cell: grid[cell] == 0,
                    max_attempts=50,
                )
            except ValueError:
                sampled_row, sampled_col = next(
                    (r, c)
                    for r in range(grid.shape[0])
                    for c in range(grid.shape[1])
                    if grid[r, c] == 0
                )
            row, col = gridvars.get(
                f"distractor_{distractor_index}_position",
                (sampled_row, sampled_col),
            )
            distractor_color = gridvars.get(
                f"distractor_{distractor_index}_color",
                random.choice(distractor_colors),
            )
            grid[int(row), int(col)] = int(distractor_color)
        return grid

    def transform_input(self, grid, taskvars):
        target_color = taskvars["target_color"]
        halo_color = taskvars["halo_color"]
        radius = taskvars["radius"]
        output = grid.copy()
        targets = np.argwhere(grid == target_color)
        for row, col in targets:
            for delta_row in range(-radius, radius + 1):
                for delta_col in range(-radius, radius + 1):
                    new_row = row + delta_row
                    new_col = col + delta_col
                    if (
                        0 <= new_row < grid.shape[0]
                        and 0 <= new_col < grid.shape[1]
                        and grid[new_row, new_col] == 0
                    ):
                        output[new_row, new_col] = halo_color
        output[grid == target_color] = target_color
        return output
