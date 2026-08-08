from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Taskd90796e8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains isolated cells and separated two-cell objects on {color('background_color')}.",
            "2. A target object is an orthogonally adjacent pair containing one {color('first_color')} cell and one {color('second_color')} cell.",
            "3. Target pairs can be horizontal or vertical and either color can occur first spatially.",
            "4. Isolated cells of either pair color and cells of {color('protected_color')} are distractors that must remain unchanged.",
            "5. Grid size, target-pair count, orientations, and distractor positions vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Find every separate two-cell component containing exactly one {color('first_color')} and one {color('second_color')} cell.",
            "2. Remove both cells of each target pair.",
            "3. Place one {color('merged_color')} cell at the former coordinate of the {color('first_color')} member.",
            "4. Preserve isolated pair-color cells, all {color('protected_color')} cells, and the {color('background_color')} field.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        first, second, merged, protected = random.sample(range(1, 10), 4)
        taskvars = {
            "background_color": 0,
            "first_color": first,
            "second_color": second,
            "merged_color": merged,
            "protected_color": protected,
        }
        train_gridvars = [
            {"rows": 12, "cols": 14, "orientations": ["horizontal", "horizontal"], "isolated": 2, "protected": 1},
            {"rows": 14, "cols": 15, "orientations": ["vertical", "horizontal", "vertical"], "isolated": 2, "protected": 2},
            {"rows": 15, "cols": 17, "orientations": ["horizontal", "vertical", "horizontal", "vertical"], "isolated": 3, "protected": 1},
            {"rows": 16, "cols": 18, "orientations": ["vertical", "vertical", "horizontal"], "isolated": 4, "protected": 3},
        ]
        test_gridvars = {
            "rows": 19,
            "cols": 21,
            "orientations": ["horizontal", "vertical", "horizontal", "vertical", "horizontal"],
            "isolated": 4,
            "protected": 3,
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
        first = taskvars["first_color"]
        second = taskvars["second_color"]
        background = taskvars["background_color"]
        objects = []
        for pair_index, orientation in enumerate(gridvars["orientations"]):
            def sample_pair():
                return np.asarray(
                    gridvars.get(
                        f"pair_{pair_index}",
                        create_object(
                            1,
                            2,
                            [first, second],
                            contiguity=Contiguity.FOUR,
                            background=background,
                        ),
                    ),
                    dtype=int,
                )

            def valid_pair(pair):
                return pair.shape == (1, 2) and set(map(int, pair[0])) == {first, second}

            try:
                pair = retry(sample_pair, valid_pair, max_attempts=60)
            except ValueError:
                pair = np.array([[first, second]], dtype=int)
            if gridvars.get(
                f"pair_{pair_index}_flip", random.choice([True, False])
            ):
                pair = np.fliplr(pair)
            if orientation == "vertical":
                pair = pair.T
            objects.append(pair)

        isolated_colors = [first, second] * ((gridvars["isolated"] + 1) // 2)
        for isolated_index, color in enumerate(
            isolated_colors[: gridvars["isolated"]]
        ):
            objects.append(
                np.asarray(
                    gridvars.get(
                        f"isolated_{isolated_index}",
                        create_object(
                            1,
                            1,
                            color,
                            contiguity=Contiguity.FOUR,
                            background=background,
                        ),
                    ),
                    dtype=int,
                )
            )
        for protected_index in range(gridvars["protected"]):
            objects.append(
                np.asarray(
                    gridvars.get(
                        f"protected_{protected_index}",
                        create_object(
                            1,
                            1,
                            taskvars["protected_color"],
                            contiguity=Contiguity.FOUR,
                            background=background,
                        ),
                    ),
                    dtype=int,
                )
            )

        rows = gridvars["rows"]
        cols = gridvars["cols"]

        def sample_positions():
            return [
                (
                    gridvars.get(
                        f"object_{object_index}_row",
                        random.randint(0, rows - obj.shape[0]),
                    ),
                    gridvars.get(
                        f"object_{object_index}_col",
                        random.randint(0, cols - obj.shape[1]),
                    ),
                )
                for object_index, obj in enumerate(objects)
            ]

        def valid_positions(positions):
            occupied_groups = []
            for obj, (top, left) in zip(objects, positions):
                coords = {
                    (top + row, left + col)
                    for row, col in zip(*np.where(obj != background))
                }
                for previous in occupied_groups:
                    if any(
                        max(abs(row - other_row), abs(col - other_col)) <= 1
                        for row, col in coords
                        for other_row, other_col in previous
                    ):
                        return False
                occupied_groups.append(coords)
            return True

        try:
            positions = retry(sample_positions, valid_positions, max_attempts=100)
        except ValueError:
            positions = [
                (1 + 3 * (index // 4), 1 + 4 * (index % 4))
                for index in range(len(objects))
            ]
        positions = gridvars.get("positions", positions)

        grid = np.full((rows, cols), background, dtype=int)
        for obj, position in zip(objects, positions):
            GridObject.from_array(obj, offset=position).paste(grid)
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        first = taskvars["first_color"]
        second = taskvars["second_color"]
        filtered = np.full(grid.shape, background, dtype=int)
        pair_mask = (grid == first) | (grid == second)
        filtered[pair_mask] = grid[pair_mask]
        objects = find_connected_objects(
            filtered,
            diagonal_connectivity=False,
            background=background,
            monochromatic=False,
        )
        output = grid.copy()
        for obj in objects:
            if obj.size == 2 and obj.colors == {first, second}:
                first_coords = [
                    (row, col)
                    for row, col, color in obj.cells
                    if int(color) == first
                ]
                obj.cut(output, background=background)
                output[first_coords[0]] = taskvars["merged_color"]
        return output
