from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random


class Task7fe24cddGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['input_size']}-by-{vars['input_size']} square grid.",
            "2. It contains a multicolored pattern on an empty background, with both occupied and empty cells allowed.",
            "3. The pattern is generally not invariant under a quarter-turn rotation.",
            "4. Different examples vary their palette, density, and spatial arrangement.",
            "5. Every example in one task shares the same source-grid side length."
        ]
        transformation_reasoning_chain = [
            "1. Create a {vars['input_size'] * 2}-by-{vars['input_size'] * 2} output divided into four equal quadrants with no separators.",
            "2. Place the unchanged input in the upper-left quadrant and its 90-degree clockwise rotation in the upper-right quadrant.",
            "3. Place the 90-degree counterclockwise rotation in the lower-left quadrant.",
            "4. Place the 180-degree rotation in the lower-right quadrant, preserving every color."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, object], TrainTestData]:
        taskvars = {"input_size": random.randint(3, 9)}
        train_examples = []
        for index, pattern_kind in enumerate(
            ["random_sparse", "bands", "diagonal", "blocks"]
        ):
            input_grid = self.create_input(
                taskvars, {"pattern_kind": pattern_kind, "phase": index}
            )
            train_examples.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )

        test_input = self.create_input(
            taskvars, {"pattern_kind": "nested_corner", "phase": 4}
        )
        test_example = {
            "input": test_input,
            "output": self.transform_input(test_input, taskvars),
        }
        return taskvars, {"train": train_examples, "test": [test_example]}

    def create_input(
        self, taskvars: dict[str, object], gridvars: dict[str, object]
    ) -> np.ndarray:
        size = int(taskvars["input_size"])
        pattern_kind = str(gridvars["pattern_kind"])
        phase = int(gridvars.get("phase", 0))

        def sample_pattern() -> np.ndarray:
            palette = random.sample(range(1, 10), random.randint(2, 5))
            grid = np.zeros((size, size), dtype=int)
            if pattern_kind == "random_sparse":
                return random_cell_coloring(
                    grid,
                    palette,
                    density=random.uniform(0.45, 0.8),
                    background=0,
                )
            if pattern_kind == "nested_corner":
                for layer in range((size + 1) // 2):
                    color = palette[(layer + phase) % len(palette)]
                    grid[layer, layer : size - layer] = color
                    grid[layer : size - layer, layer] = color
                if size > 3:
                    grid[-1, -1] = palette[-1]
                return grid

            for row in range(size):
                for col in range(size):
                    if pattern_kind == "bands":
                        index = row + col // 2 + phase
                    elif pattern_kind == "diagonal":
                        index = row + 2 * col + row * col + phase
                    else:
                        index = row // 2 + 2 * (col // 2) + row * col + phase
                    if (row + 2 * col + phase) % 5 != 0:
                        grid[row, col] = palette[index % len(palette)]
            return grid

        def valid(candidate: np.ndarray) -> bool:
            nonzero_colors = set(int(value) for value in np.unique(candidate)) - {0}
            return bool(
                len(nonzero_colors) >= 2
                and not np.array_equal(candidate, np.rot90(candidate, k=1))
            )

        def sample_valid_pattern() -> np.ndarray:
            try:
                return retry(sample_pattern, valid, max_attempts=50)
            except ValueError:
                fallback = np.fromfunction(
                    lambda row, col: ((2 * row + col + row * col) % 4),
                    (size, size),
                    dtype=int,
                ).astype(int)
                fallback[0, 0] = 5
                fallback[-1, 0] = 7
                return fallback

        pattern = np.asarray(
            gridvars.get("pattern", sample_valid_pattern()),
            dtype=int,
        )
        if pattern.shape != (size, size):
            raise ValueError("pattern must match the task input size")
        if np.any((pattern < 0) | (pattern > 9)):
            raise ValueError("pattern colors must be ARC palette values 0..9")
        if not valid(pattern):
            raise ValueError(
                "pattern must use at least two foreground colors and break "
                "quarter-turn symmetry"
            )
        return pattern

    def transform_input(
        self, grid: np.ndarray, taskvars: dict[str, object]
    ) -> np.ndarray:
        upper = np.concatenate((grid, np.rot90(grid, k=3)), axis=1)
        lower = np.concatenate(
            (np.rot90(grid, k=1), np.rot90(grid, k=2)), axis=1
        )
        return np.concatenate((upper, lower), axis=0)
