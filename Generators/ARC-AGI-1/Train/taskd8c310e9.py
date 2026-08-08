from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import BorderBehavior, GridObject
import numpy as np
import random


class Taskd8c310e9Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['grid_rows']}-row grid containing a multicolor pattern in a left prefix on {color('background_color')}.",
            "2. The active prefix contains more than one copy, possibly partial, of one horizontal tile.",
            "3. The smallest valid tile width, tile colors, active height, and number of shown columns vary between examples.",
            "4. All columns to the right of the shown prefix are {color('background_color')}.",
            "5. Tile columns contain enough foreground evidence to determine the period uniquely.",
        ]
        transformation_reasoning_chain = [
            "1. End the active prefix at its last non-{color('background_color')} column.",
            "2. Find the smallest horizontal shift whose overlapping prefix columns agree exactly.",
            "3. Treat the first shift-width block as the repeating tile.",
            "4. Repeat the tile across the full grid width, clipping only the final copy at the right edge.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {
            "grid_rows": random.randint(5, 8),
            "background_color": 0,
        }
        train_gridvars = [
            {"cols": 17, "period": 3, "active_height": 2, "shown_repeats": 2, "partial": 1},
            {"cols": 19, "period": 4, "active_height": 3, "shown_repeats": 2, "partial": 0},
            {"cols": 21, "period": 5, "active_height": 4, "shown_repeats": 1, "partial": 4},
            {"cols": 23, "period": 6, "active_height": 3, "shown_repeats": 2, "partial": 2},
        ]
        test_gridvars = {
            "cols": 25,
            "period": 7,
            "active_height": 5,
            "shown_repeats": 1,
            "partial": 5,
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
        rows = taskvars["grid_rows"]
        background = taskvars["background_color"]
        period = gridvars["period"]
        active_height = min(gridvars["active_height"], rows)
        palette_size = gridvars.get(
            "palette_size",
            random.randint(2, 4),
        )
        palette = gridvars.get(
            "palette",
            random.sample(range(1, 10), palette_size),
        )

        def sample_tile():
            return create_object(
                active_height,
                period,
                palette,
                contiguity=Contiguity.NONE,
                background=background,
            )

        def valid_tile(tile):
            if not np.all(np.any(tile != background, axis=0)):
                return False
            if len(set(int(value) for value in np.unique(tile) if int(value) != background)) < 2:
                return False
            for shift in range(1, period):
                if np.array_equal(tile[:, shift:], tile[:, : period - shift]):
                    return False
            return True

        try:
            tile = gridvars.get(
                "tile",
                retry(sample_tile, valid_tile, max_attempts=80),
            )
        except ValueError:
            fallback_tile = np.full(
                (active_height, period),
                background,
                dtype=int,
            )
            for row in range(active_height):
                for col in range(period):
                    if (row + 2 * col) % 3 != 0:
                        fallback_tile[row, col] = palette[
                            (row + col) % len(palette)
                        ]
            fallback_tile[-1, :] = [
                palette[col % len(palette)]
                for col in range(period)
            ]
            tile = gridvars.get("tile", fallback_tile)
        tile = np.asarray(tile, dtype=int)

        unit = np.full((rows, period), background, dtype=int)
        unit[rows - active_height :, :] = tile
        shown = gridvars["shown_repeats"] * period + gridvars["partial"]
        grid = np.full((rows, gridvars["cols"]), background, dtype=int)
        for col in range(min(shown, grid.shape[1])):
            grid[:, col] = unit[:, col % period]
        return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        active_columns = np.where(np.any(grid != background, axis=0))[0]
        if len(active_columns) == 0:
            return grid.copy()
        active_stop = int(active_columns[-1]) + 1
        period = None
        for shift in range(1, active_stop):
            if np.array_equal(
                grid[:, shift:active_stop],
                grid[:, : active_stop - shift],
            ):
                period = shift
                break
        if period is None:
            return grid.copy()
        unit = GridObject.from_array(grid[:, :period])
        output = np.full(grid.shape, background, dtype=int)
        for left in range(0, grid.shape[1], period):
            unit.copy().translate(
                0,
                left,
                border_behavior=BorderBehavior.CLIP,
                grid_shape=grid.shape,
            ).paste(output)
        return output
