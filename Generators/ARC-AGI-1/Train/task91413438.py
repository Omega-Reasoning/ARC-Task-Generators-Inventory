from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Task91413438Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Every input is a {vars['tile_size']}x{vars['tile_size']} grid on {color('background_color')}.",
            "2. Its foreground cells all share one non-{color('background_color')} color and need not be connected.",
            "3. The foreground-cell count and complementary background-cell count sum to {vars['tile_size']} squared.",
            "4. Foreground count varies between examples and controls how many copies will be drawn.",
            "5. Foreground color and cell arrangement also vary independently.",
        ]
        transformation_reasoning_chain = [
            "1. Count the non-{color('background_color')} cells to obtain the pattern-copy count.",
            "2. Count the {color('background_color')} cells to obtain the number of {vars['tile_size']}x{vars['tile_size']} tile slots along each output side.",
            "3. Create a square {color('background_color')} canvas with that many tile rows and tile columns.",
            "4. Visit tile slots in row-major order and paste the complete input pattern into the first copy-count slots.",
            "5. Leave every remaining tile slot entirely {color('background_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            "background_color": 0,
            "tile_size": 3,
        }
        train_specs = [
            {"foreground_count": 3},
            {"foreground_count": 4},
            {"foreground_count": 5},
            {"foreground_count": 6},
        ]
        test_spec = {"foreground_count": 2}

        def make_pair(gridvars: dict) -> GridPair:
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [make_pair(spec) for spec in train_specs]
        return taskvars, {"train": train, "test": [make_pair(test_spec)]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        background_color = taskvars["background_color"]
        tile_size = taskvars["tile_size"]
        foreground_count = gridvars["foreground_count"]
        foreground_color = gridvars.get(
            "foreground_color",
            random.randint(1, 9),
        )

        def sample_pattern() -> np.ndarray:
            return create_object(
                tile_size,
                tile_size,
                foreground_color,
                contiguity=Contiguity.NONE,
                background=background_color,
            )

        def has_requested_count(pattern: np.ndarray) -> bool:
            return int(np.sum(pattern != background_color)) == foreground_count

        try:
            sampled_pattern = retry(
                sample_pattern,
                has_requested_count,
                max_attempts=80,
            )
        except ValueError:
            sampled_pattern = np.full(
                (tile_size, tile_size),
                background_color,
                dtype=int,
            )
            for index in range(foreground_count):
                row, col = divmod(index, tile_size)
                sampled_pattern[row, col] = foreground_color

        pattern = np.asarray(
            gridvars.get("pattern", sampled_pattern),
            dtype=int,
        )
        if pattern.shape != (tile_size, tile_size):
            raise ValueError("pattern must match tile_size")
        if not np.all(
            np.logical_or(
                pattern == background_color,
                pattern == foreground_color,
            )
        ):
            raise ValueError("pattern may only use background and foreground colors")
        if not has_requested_count(pattern):
            raise ValueError("pattern must contain foreground_count occupied cells")
        return np.array(pattern, dtype=int, copy=True)

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background_color = taskvars["background_color"]
        tile_size = taskvars["tile_size"]
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=background_color,
            monochromatic=True,
        )
        foreground_count = sum(len(obj) for obj in objects)
        background_count = int(grid.size) - foreground_count
        output = np.full(
            (background_count * tile_size, background_count * tile_size),
            background_color,
            dtype=int,
        )
        pattern = GridObject.from_array(grid)
        for tile_index in range(foreground_count):
            tile_row = tile_index // background_count
            tile_col = tile_index % background_count
            pattern.copy().translate(
                tile_row * tile_size,
                tile_col * tile_size,
                grid_shape=output.shape,
            ).paste(
                output,
                overwrite=True,
                background=background_color,
            )
        return output
