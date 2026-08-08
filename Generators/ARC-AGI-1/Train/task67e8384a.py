from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random


class Task67e8384aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a completely filled {vars['input_rows']}-row by {vars['input_cols']}-column grid.",
            "2. The cells form a multicolored pattern with no empty cells.",
            "3. The arrangement generally differs under both a left-to-right reflection and a top-to-bottom reflection.",
            "4. Different examples vary the number of colors and the geometry of the dense pattern.",
            "5. Every example in one task shares the same input dimensions."
        ]
        transformation_reasoning_chain = [
            "1. Copy the input into the upper-left of a {vars['input_rows'] * 2}-row by {vars['input_cols'] * 2}-column output grid.",
            "2. Reflect the input left-to-right and place that reflection directly to its right, with no separator.",
            "3. Reflect this entire upper strip top-to-bottom and place it directly below, with no separator.",
            "4. Preserve every source-cell color in all four reflected quadrants."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, object], TrainTestData]:
        taskvars = {
            "input_rows": random.randint(3, 10),
            "input_cols": random.randint(3, 10),
        }

        train_examples = []
        train_kinds = ["mosaic", "row_steps", "column_steps", "blocks"]
        for index, pattern_kind in enumerate(train_kinds):
            gridvars = {"pattern_kind": pattern_kind, "phase": index}
            input_grid = self.create_input(taskvars, gridvars)
            train_examples.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )

        test_gridvars = {
            "pattern_kind": "diagonal_wave",
            "phase": random.randint(0, 8),
        }
        test_input = self.create_input(taskvars, test_gridvars)
        test_example = {
            "input": test_input,
            "output": self.transform_input(test_input, taskvars),
        }
        return taskvars, {"train": train_examples, "test": [test_example]}

    def create_input(
        self, taskvars: dict[str, object], gridvars: dict[str, object]
    ) -> np.ndarray:
        rows = int(taskvars["input_rows"])
        cols = int(taskvars["input_cols"])
        pattern_kind = str(gridvars["pattern_kind"])
        phase = int(gridvars.get("phase", 0))

        def sample_pattern() -> np.ndarray:
            palette = [
                int(color)
                for color in gridvars.get(
                    "palette",
                    random.sample(range(1, 10), random.randint(3, 6)),
                )
            ]
            grid = np.zeros((rows, cols), dtype=int)
            if pattern_kind == "mosaic":
                return np.asarray(
                    gridvars.get(
                        "mosaic_candidate",
                        random_cell_coloring(
                            grid, palette, density=1.0, background=0
                        ),
                    ),
                    dtype=int,
                )

            for row in range(rows):
                for col in range(cols):
                    if pattern_kind == "row_steps":
                        index = 2 * row + col + row // 2 + phase
                    elif pattern_kind == "column_steps":
                        index = row + 2 * col + col // 2 + phase
                    elif pattern_kind == "blocks":
                        index = row // 2 + 2 * (col // 2) + row * col + phase
                    else:
                        index = row + col + (row * col) // 2 + phase
                    grid[row, col] = palette[index % len(palette)]
            return grid

        def valid(candidate: np.ndarray) -> bool:
            return bool(
                len(np.unique(candidate)) >= 2
                and not np.array_equal(candidate, np.fliplr(candidate))
                and not np.array_equal(candidate, np.flipud(candidate))
            )

        try:
            return retry(sample_pattern, valid, max_attempts=40)
        except ValueError:
            fallback = np.fromfunction(
                lambda row, col: 1 + ((2 * row + col + row * col) % 3),
                (rows, cols),
                dtype=int,
            ).astype(int)
            fallback[0, 0] = 4
            fallback[-1, -1] = 5
            return fallback

    def transform_input(
        self, grid: np.ndarray, taskvars: dict[str, object]
    ) -> np.ndarray:
        upper_half = np.concatenate((grid, np.fliplr(grid)), axis=1)
        return np.concatenate((upper_half, np.flipud(upper_half)), axis=0)
