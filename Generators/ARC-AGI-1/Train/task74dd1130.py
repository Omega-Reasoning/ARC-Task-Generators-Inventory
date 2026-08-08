from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random


class Task74dd1130Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a completely filled {vars['grid_size']}x{vars['grid_size']} square with no empty cells.",
            "2. Exactly {vars['palette_size']} nonempty colors occur, and their identities vary from example to example.",
            "3. Colored cells form an arrangement that is not symmetric across the main diagonal.",
            "4. Rows, columns, repeated colors, and the number of unequal mirrored cell pairs can vary across examples.",
        ]
        transformation_reasoning_chain = [
            "1. Use the main diagonal from the top-left corner to the bottom-right corner as the reflection axis.",
            "2. Move every cell at row r and column c to row c and column r, preserving its color.",
            "3. Leave the main-diagonal cells fixed and return the transposed {vars['grid_size']}x{vars['grid_size']} grid.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {"grid_size": 3, "palette_size": 3}
        train_specs = [
            {"mismatched_pairs": 1},
            {"mismatched_pairs": 1},
            {"mismatched_pairs": 2},
            {"mismatched_pairs": 2},
        ]
        random.shuffle(train_specs)

        train = []
        for gridvars in train_specs:
            input_grid = self.create_input(taskvars, gridvars)
            train.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )

        test_input = self.create_input(taskvars, {"mismatched_pairs": 3})
        return taskvars, {
            "train": train,
            "test": [
                {
                    "input": test_input,
                    "output": self.transform_input(test_input, taskvars),
                }
            ],
        }

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        grid_size = taskvars["grid_size"]
        palette = list(
            gridvars.get(
                "palette",
                random.sample(range(1, 10), taskvars["palette_size"]),
            )
        )
        if (
            len(palette) != taskvars["palette_size"]
            or len(set(palette)) != len(palette)
            or any(
                not isinstance(color, int) or color not in range(1, 10)
                for color in palette
            )
        ):
            raise ValueError("palette must contain distinct nonzero ARC colors")
        target_mismatches = int(gridvars["mismatched_pairs"])

        def sample_grid():
            candidate = np.zeros((grid_size, grid_size), dtype=int)
            return random_cell_coloring(
                candidate,
                palette,
                density=1.0,
                background=0,
                overwrite=False,
            )

        def mismatch_count(candidate):
            return sum(
                int(candidate[row, column] != candidate[column, row])
                for row in range(grid_size)
                for column in range(row + 1, grid_size)
            )

        def valid_grid(candidate):
            return bool(
                set(candidate.flatten()) == set(palette)
                and mismatch_count(candidate) == target_mismatches
            )

        def sample_valid_grid() -> np.ndarray:
            try:
                return retry(
                    sample_grid,
                    valid_grid,
                    max_attempts=80,
                )
            except ValueError:
                first, second, third = palette
                fallback = np.array(
                    [
                        [first, first, second],
                        [first, third, third],
                        [second, third, first],
                    ],
                    dtype=int,
                )
                mirrored_pairs = [(0, 1), (0, 2), (1, 2)]
                replacements = [second, third, first]
                for (row, column), color in zip(
                    mirrored_pairs[:target_mismatches],
                    replacements[:target_mismatches],
                ):
                    fallback[column, row] = color
                return fallback

        grid = np.asarray(
            gridvars.get("grid", sample_valid_grid()),
            dtype=int,
        )
        if grid.shape != (grid_size, grid_size):
            raise ValueError("grid must match grid_size")
        if not valid_grid(grid):
            raise ValueError(
                "grid must use exactly the declared palette and mismatch count"
            )
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        grid_size = taskvars["grid_size"]
        output = np.zeros((grid_size, grid_size), dtype=int)
        for row in range(grid_size):
            for column in range(grid_size):
                output[column, row] = grid[row, column]
        return output
