from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects


class Task0a938d79Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a strongly rectangular empty grid containing exactly two differently colored singleton seeds.",
            "2. In a wide grid the seeds identify two columns; in a tall grid they identify two rows.",
            "3. Reading along the long axis, the first seed precedes the second and their coordinate difference is the repetition spacing.",
            "4. The seed coordinates along the short axis do not affect the completed pattern.",
            "5. Every completed stripe has thickness {vars['line_thickness']} cell.",
        ]
        transformation_reasoning_chain = [
            "1. Choose columns for a wide grid or rows for a tall grid and sort the two seeds along that long axis.",
            "2. Expand each seed position into a complete {vars['line_thickness']}-cell-thick line across the short axis.",
            "3. Preserve the two seed colors in their sorted order and repeat the alternating pair forward using their spacing.",
            "4. Stop at the far boundary and leave all positions before the first seed empty.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {"line_thickness": 1}
        train_specs = [
            {"orientation": "wide", "spacing": 2},
            {"orientation": "wide", "spacing": 3},
            {"orientation": "tall", "spacing": 2},
            {"orientation": "tall", "spacing": 4},
        ]
        test_spec = {"orientation": "wide", "spacing": 5}

        def make_pair(gridvars: dict) -> GridPair:
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [make_pair(spec) for spec in train_specs]
        return taskvars, {"train": train, "test": [make_pair(test_spec)]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        spacing = gridvars["spacing"]
        orientation = gridvars["orientation"]

        def sample_dimensions() -> tuple:
            if orientation == "wide":
                return (
                    gridvars.get("rows", random.randint(7, 12)),
                    gridvars.get("cols", random.randint(20, 29)),
                    gridvars.get("first", random.randint(2, 7)),
                )
            return (
                gridvars.get("rows", random.randint(20, 29)),
                gridvars.get("cols", random.randint(7, 12)),
                gridvars.get("first", random.randint(2, 7)),
            )

        try:
            rows, cols, first = retry(
                sample_dimensions,
                lambda value: value[2] + 2 * spacing
                < (value[1] if orientation == "wide" else value[0]),
                max_attempts=60,
            )
        except ValueError:
            if orientation == "wide":
                rows, cols, first = 9, min(30, 4 + 4 * spacing), 2
            else:
                rows, cols, first = min(30, 4 + 4 * spacing), 9, 2

        colors = gridvars.get("colors", random.sample(range(1, 10), 2))
        grid = np.zeros((rows, cols), dtype=int)
        if orientation == "wide":
            first_seed = (
                gridvars.get("first_seed_row", random.randrange(rows)),
                first,
                colors[0],
            )
            second_seed = (
                gridvars.get("second_seed_row", random.randrange(rows)),
                first + spacing,
                colors[1],
            )
        else:
            first_seed = (
                first,
                gridvars.get("first_seed_col", random.randrange(cols)),
                colors[0],
            )
            second_seed = (
                first + spacing,
                gridvars.get("second_seed_col", random.randrange(cols)),
                colors[1],
            )
        GridObject({first_seed, second_seed}).paste(grid)
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=True,
            background=0,
            monochromatic=True,
        )
        seeds = [next(iter(obj.cells)) for obj in objects]
        output = np.zeros_like(grid)
        thickness = taskvars["line_thickness"]

        if grid.shape[1] > grid.shape[0]:
            ordered = sorted(seeds, key=lambda cell: cell[1])
            first = ordered[0][1]
            spacing = ordered[1][1] - first
            colors = [ordered[0][2], ordered[1][2]]
            index = 0
            for col in range(first, grid.shape[1], spacing):
                output[:, col : col + thickness] = colors[index % 2]
                index += 1
        else:
            ordered = sorted(seeds, key=lambda cell: cell[0])
            first = ordered[0][0]
            spacing = ordered[1][0] - first
            colors = [ordered[0][2], ordered[1][2]]
            index = 0
            for row in range(first, grid.shape[0], spacing):
                output[row : row + thickness, :] = colors[index % 2]
                index += 1
        return output
