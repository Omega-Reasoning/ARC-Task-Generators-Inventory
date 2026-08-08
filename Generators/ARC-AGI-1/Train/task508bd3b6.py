from __future__ import annotations

import random

import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator
from Framework.input_library import retry
from Framework.transformation_library import find_connected_objects


class Task508bd3b6Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Input grids use {color('background_color')} as empty space and contain a solid {color('wall_color')} band along exactly one grid edge.",
            "2. A short diagonal segment of two or three {color('seed_color')} cells lies away from the wall.",
            "3. Reading the segment in one direction points diagonally toward the wall, with no obstacle between the seed and wall.",
            "4. Wall edge, thickness, diagonal slope, and seed position vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Continue the {color('seed_color')} segment in the same diagonal direction using {color('ray_color')} cells.",
            "2. Stop immediately before the next diagonal step would enter the {color('wall_color')} band.",
            "3. Reflect the direction component perpendicular to the wall while retaining the component parallel to it.",
            "4. Continue the reflected diagonal in {color('ray_color')} until it exits the grid, preserving the seed and wall.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        wall_color, seed_color, ray_color = random.sample(range(1, 10), 3)
        taskvars = {
            "background_color": 0,
            "wall_color": wall_color,
            "seed_color": seed_color,
            "ray_color": ray_color,
        }
        schedules = [
            {"edge": "left", "slope": 1, "thickness": 1, "seed_length": 2, "approach": 3},
            {"edge": "right", "slope": -1, "thickness": 2, "seed_length": 3, "approach": 4},
            {"edge": "top", "slope": 1, "thickness": 3, "seed_length": 2, "approach": 3},
            {"edge": "bottom", "slope": -1, "thickness": 1, "seed_length": 3, "approach": 5},
        ]
        train = []
        for schedule in schedules:
            gridvars = {
                **schedule,
                "height": random.randint(14, 20),
                "width": random.randint(15, 22),
            }
            grid = self.create_input(taskvars, gridvars)
            train.append({"input": grid, "output": self.transform_input(grid, taskvars)})
        test_grid = self.create_input(
            taskvars,
            {
                "edge": random.choice(["left", "right", "top", "bottom"]),
                "slope": random.choice([-1, 1]),
                "thickness": 4,
                "seed_length": 3,
                "approach": 6,
                "height": random.randint(21, 25),
                "width": random.randint(22, 27),
            },
        )
        return taskvars, {
            "train": train,
            "test": [{"input": test_grid, "output": self.transform_input(test_grid, taskvars)}],
        }

    def create_input(self, taskvars, gridvars):
        rows = gridvars["height"]
        cols = gridvars["width"]
        edge = gridvars["edge"]
        thickness = gridvars["thickness"]
        seed_length = gridvars["seed_length"]
        approach = gridvars["approach"]
        slope = gridvars["slope"]

        def make_grid(parallel_coordinate):
            grid = np.full((rows, cols), taskvars["background_color"], dtype=int)
            wall = taskvars["wall_color"]
            if edge == "left":
                grid[:, :thickness] = wall
                turn = (parallel_coordinate, thickness)
                delta = (slope, -1)
            elif edge == "right":
                grid[:, cols - thickness :] = wall
                turn = (parallel_coordinate, cols - thickness - 1)
                delta = (slope, 1)
            elif edge == "top":
                grid[:thickness, :] = wall
                turn = (thickness, parallel_coordinate)
                delta = (-1, slope)
            else:
                grid[rows - thickness :, :] = wall
                turn = (rows - thickness - 1, parallel_coordinate)
                delta = (1, slope)
            endpoint = (turn[0] - approach * delta[0], turn[1] - approach * delta[1])
            for index in range(seed_length):
                row = endpoint[0] - index * delta[0]
                col = endpoint[1] - index * delta[1]
                if not (0 <= row < rows and 0 <= col < cols):
                    return None
                if grid[row, col] != taskvars["background_color"]:
                    return None
                grid[row, col] = taskvars["seed_color"]
            return grid

        upper = rows - 2 if edge in {"left", "right"} else cols - 2

        def sample_grid():
            parallel_coordinate = int(gridvars.get(
                "parallel_coordinate",
                random.randint(1, upper),
            ))
            return make_grid(parallel_coordinate)

        def valid_grid(grid):
            if grid is None:
                return False
            output = self.transform_input(grid, taskvars)
            ray_count = int(np.count_nonzero(output == taskvars["ray_color"]))
            return ray_count >= approach + 3

        try:
            return retry(sample_grid, valid_grid, max_attempts=60)
        except ValueError:
            for parallel_coordinate in range(1, upper + 1):
                grid = make_grid(parallel_coordinate)
                if valid_grid(grid):
                    return grid
        raise ValueError("could not place a diagonal seed with a visible reflected branch")

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        wall = taskvars["wall_color"]
        seed = taskvars["seed_color"]
        ray = taskvars["ray_color"]
        rows, cols = grid.shape
        wall_mask = grid == wall
        full_rows = [row for row in range(rows) if np.all(wall_mask[row, :])]
        full_cols = [col for col in range(cols) if np.all(wall_mask[:, col])]
        if full_cols:
            wall_is_vertical = True
            wall_on_low_side = min(full_cols) == 0
        else:
            wall_is_vertical = False
            wall_on_low_side = min(full_rows) == 0
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=True,
            background=background,
            monochromatic=True,
        )
        seed_objects = objects.with_color(seed)
        if len(seed_objects) != 1:
            return grid.copy()
        seed_cells = [(int(row), int(col)) for row, col, _ in seed_objects[0]]

        def distance_to_wall(cell):
            row, col = cell
            coordinate = col if wall_is_vertical else row
            boundary = min(full_cols) if wall_is_vertical else min(full_rows)
            if wall_on_low_side:
                boundary = max(full_cols) if wall_is_vertical else max(full_rows)
            return abs(coordinate - boundary)

        endpoint = min(seed_cells, key=distance_to_wall)
        neighbors = [
            cell
            for cell in seed_cells
            if cell != endpoint
            and abs(cell[0] - endpoint[0]) == 1
            and abs(cell[1] - endpoint[1]) == 1
        ]
        if not neighbors:
            return grid.copy()
        previous = neighbors[0]
        delta_row = endpoint[0] - previous[0]
        delta_col = endpoint[1] - previous[1]
        output = grid.copy()
        current_row, current_col = endpoint
        bounced = False
        while True:
            next_row = current_row + delta_row
            next_col = current_col + delta_col
            if not (0 <= next_row < rows and 0 <= next_col < cols):
                break
            if grid[next_row, next_col] == wall:
                if bounced:
                    break
                bounced = True
                if wall_is_vertical:
                    delta_col *= -1
                else:
                    delta_row *= -1
                next_row = current_row + delta_row
                next_col = current_col + delta_col
                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    break
            if output[next_row, next_col] == background:
                output[next_row, next_col] = ray
            current_row, current_col = next_row, next_col
        return output
