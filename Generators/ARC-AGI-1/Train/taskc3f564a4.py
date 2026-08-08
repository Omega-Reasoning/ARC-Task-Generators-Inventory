from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import Contiguity, create_object, retry


class Taskc3f564a4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['grid_size']} by {vars['grid_size']} grid containing a two-dimensional cyclic color pattern with some cells erased to {color('background_color')}.",
            "2. Moving one cell right or one cell down advances by one position in the same nonzero color cycle.",
            "3. The cycle length and the number, shapes, and positions of erased regions vary between examples.",
            "4. Every cycle color remains visible often enough to determine the palette and phase uniquely.",
            "5. Erased regions are proper subsets of the grid and include connected multi-cell holes.",
        ]
        transformation_reasoning_chain = [
            "1. Infer the ordered nonzero color cycle and its phase from the surviving cells.",
            "2. For every coordinate, advance through that cycle by the sum of its row and column offsets.",
            "3. Replace each {color('background_color')} erasure with the predicted cycle color.",
            "4. Preserve every already visible nonzero cell and return the completed periodic grid.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {
            "grid_size": random.choice([12, 14, 16, 18]),
            "background_color": 0,
        }
        train_gridvars = [
            {"cycle_length": 2, "mask_shapes": [(2, 3)]},
            {"cycle_length": 3, "mask_shapes": [(3, 3), (2, 2)]},
            {"cycle_length": 4, "mask_shapes": [(4, 3)]},
            {"cycle_length": 5, "mask_shapes": [(3, 5), (2, 3)]},
        ]
        test_gridvars = {
            "cycle_length": 6,
            "mask_shapes": [(5, 5), (3, 4)],
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
        size = taskvars["grid_size"]
        background = taskvars["background_color"]
        palette = list(
            gridvars.get(
                "palette",
                sorted(random.sample(range(1, 10), gridvars["cycle_length"])),
            )
        )
        if (
            len(palette) != gridvars["cycle_length"]
            or palette != sorted(palette)
            or len(set(palette)) != len(palette)
            or any(color not in range(1, 10) for color in palette)
        ):
            raise ValueError("palette must be sorted, unique, nonzero ARC colors")
        phase = int(gridvars.get("phase", random.randrange(len(palette))))
        if phase not in range(len(palette)):
            raise ValueError("phase must index the palette")
        completed = np.fromfunction(
            lambda row, col: np.take(
                palette,
                (row.astype(int) + col.astype(int) + phase) % len(palette),
            ),
            (size, size),
            dtype=int,
        ).astype(int)

        def four_connected(mask):
            coordinates = np.argwhere(mask != 0)
            if coordinates.size == 0:
                return False
            pending = [tuple(int(value) for value in coordinates[0])]
            reached = set()
            while pending:
                row, col = pending.pop()
                if (row, col) in reached or mask[row, col] == 0:
                    continue
                reached.add((row, col))
                for row_step, col_step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row, next_col = row + row_step, col + col_step
                    if (
                        0 <= next_row < mask.shape[0]
                        and 0 <= next_col < mask.shape[1]
                        and mask[next_row, next_col] != 0
                        and (next_row, next_col) not in reached
                    ):
                        pending.append((next_row, next_col))
            return len(reached) == int(np.count_nonzero(mask))

        def valid_mask(mask, height, width):
            occupied = mask != 0
            return bool(
                mask.shape == (height, width)
                and 3 <= np.count_nonzero(occupied) < height * width
                and np.count_nonzero(np.any(occupied, axis=0)) >= 2
                and np.count_nonzero(np.any(occupied, axis=1)) >= 2
                and four_connected(mask)
            )

        def make_mask(height, width):
            def sample_mask():
                return create_object(
                    height,
                    width,
                    1,
                    contiguity=Contiguity.FOUR,
                    background=0,
                )

            try:
                return retry(
                    sample_mask,
                    lambda mask: valid_mask(mask, height, width),
                    max_attempts=50,
                )
            except ValueError:
                mask = np.zeros((height, width), dtype=int)
                mask[: max(2, height - 1), : max(2, width - 1)] = 1
                return mask

        def sample_erased_grid():
            grid = completed.copy()
            for index, (height, width) in enumerate(gridvars["mask_shapes"]):
                mask = np.asarray(
                    gridvars.get(f"mask_{index}", make_mask(height, width)),
                    dtype=int,
                )
                if mask.shape != (height, width) or not set(
                    int(value) for value in np.unique(mask)
                ).issubset({0, 1}) or not valid_mask(mask, height, width):
                    raise ValueError("forced mask does not satisfy the connected-mask contract")
                top, left = (
                    int(value)
                    for value in gridvars.get(
                        f"position_{index}",
                        (
                            random.randint(0, size - height),
                            random.randint(0, size - width),
                        ),
                    )
                )
                if not (0 <= top <= size - height and 0 <= left <= size - width):
                    raise ValueError("forced mask position does not fit the grid")
                region = grid[top : top + height, left : left + width]
                region[mask != 0] = background
            return grid

        minimum_erased = max(5, 3 * len(gridvars["mask_shapes"]))

        def valid_erasure(grid):
            visible = set(int(value) for value in np.unique(grid))
            return bool(
                np.count_nonzero(grid == background) >= minimum_erased
                and np.count_nonzero(grid == background) < grid.size // 2
                and all(color in visible for color in palette)
            )

        try:
            return retry(sample_erased_grid, valid_erasure, max_attempts=60)
        except ValueError:
            grid = completed.copy()
            for index, (height, width) in enumerate(gridvars["mask_shapes"]):
                top = (2 + 4 * index) % (size - height + 1)
                left = (3 + 5 * index) % (size - width + 1)
                rectangle_height = max(2, height - 1)
                rectangle_width = max(2, width - 1)
                grid[
                    top : top + rectangle_height,
                    left : left + rectangle_width,
                ] = background
            return grid

    def transform_input(self, grid, taskvars):
        background = taskvars["background_color"]
        palette = sorted(
            int(value) for value in np.unique(grid) if int(value) != background
        )
        if not palette:
            return grid.copy()
        period = len(palette)
        best_phase = 0
        best_score = -1
        for phase in range(period):
            score = 0
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    value = int(grid[row, col])
                    if value != background and value == palette[(row + col + phase) % period]:
                        score += 1
            if score > best_score:
                best_score = score
                best_phase = phase
        output = np.zeros_like(grid)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                output[row, col] = palette[(row + col + best_phase) % period]
        return output
