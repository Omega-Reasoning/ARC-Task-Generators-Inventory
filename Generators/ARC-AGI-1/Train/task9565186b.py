from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import random_cell_coloring, retry


class Task9565186bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a small rectangular grid with no {color('replacement_color')} cells initially.",
            "2. One keeper color is uniquely most frequent and visibly fills at least one complete row or column.",
            "3. Additional keeper-colored cells may occur outside that complete line, mixed with lower-frequency distractor colors.",
            "4. Grid dimensions, line orientation, keeper geometry, and distractor colors vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Count all input colors and identify the uniquely most frequent keeper color.",
            "2. Preserve every occurrence of that keeper color, including occurrences outside any complete line.",
            "3. Recolor every other cell, whether empty or distractor-colored, {color('replacement_color')}.",
            "4. Keep the input dimensions unchanged.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"replacement_color": random.randint(1, 9)}
        layouts = [
            {"rows": 3, "cols": 4, "orientation": "row", "multiple": False, "lower_frequency_line": False},
            {"rows": 4, "cols": 3, "orientation": "column", "multiple": False, "lower_frequency_line": False},
            {"rows": 5, "cols": 5, "orientation": "row", "multiple": False, "lower_frequency_line": True},
            {"rows": 5, "cols": 4, "orientation": "column", "multiple": True, "lower_frequency_line": False},
        ]
        test_layout = {
            "rows": 5,
            "cols": 6,
            "orientation": "column",
            "multiple": False,
            "lower_frequency_line": True,
        }

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {"input": input_grid, "output": self.transform_input(input_grid, taskvars)}

        return taskvars, {
            "train": [make_pair(layout) for layout in layouts],
            "test": [make_pair(test_layout)],
        }

    def create_input(self, taskvars, gridvars):
        replacement = taskvars["replacement_color"]
        palette = [color for color in range(1, 10) if color != replacement]
        keeper, *distractors = gridvars.get(
            "colors", random.sample(palette, 5)
        )
        rows, cols = gridvars["rows"], gridvars["cols"]

        def sample_grid():
            candidate = np.zeros((rows, cols), dtype=int)
            candidate = np.asarray(
                gridvars.get(
                    "base_grid",
                    random_cell_coloring(
                        candidate,
                        distractors,
                        density=1.0,
                        background=0,
                    ),
                ),
                dtype=int,
            )
            reserved = set()
            if gridvars["orientation"] == "row":
                line = gridvars.get("line", random.randrange(rows))
                keeper_lines = [line]
                if gridvars["multiple"]:
                    keeper_lines.append((line + 1) % rows)
                lower_line = None
                if gridvars["lower_frequency_line"]:
                    lower_line = gridvars.get(
                        "lower_line",
                        random.choice(
                            [index for index in range(rows) if index not in keeper_lines]
                        ),
                    )
                    candidate[lower_line, :] = distractors[0]
                    reserved.update((lower_line, col) for col in range(cols))
                for keeper_line in keeper_lines:
                    candidate[keeper_line, :] = keeper
                    reserved.update((keeper_line, col) for col in range(cols))
            else:
                line = gridvars.get("line", random.randrange(cols))
                keeper_lines = [line]
                if gridvars["multiple"]:
                    keeper_lines.append((line + 1) % cols)
                lower_line = None
                if gridvars["lower_frequency_line"]:
                    lower_line = gridvars.get(
                        "lower_line",
                        random.choice(
                            [index for index in range(cols) if index not in keeper_lines]
                        ),
                    )
                    candidate[:, lower_line] = distractors[0]
                    reserved.update((row, lower_line) for row in range(rows))
                for keeper_line in keeper_lines:
                    candidate[:, keeper_line] = keeper
                    reserved.update((row, keeper_line) for row in range(rows))
            available = [
                (row, col)
                for row in range(rows)
                for col in range(cols)
                if (row, col) not in reserved
            ]
            for row, col in gridvars.get(
                "extra_keeper_cells",
                random.sample(available, min(2, len(available))),
            ):
                candidate[row, col] = keeper
            return candidate

        def valid(candidate):
            colors, counts = np.unique(candidate, return_counts=True)
            frequency = {int(color): int(count) for color, count in zip(colors, counts)}
            keeper_is_unique_maximum = frequency.get(keeper, 0) > max(
                (count for color, count in frequency.items() if color != keeper),
                default=0,
            )
            if not keeper_is_unique_maximum:
                return False
            if not gridvars["lower_frequency_line"]:
                return True
            complete_distractor_colors = {
                int(candidate[row, 0])
                for row in range(rows)
                if np.all(candidate[row, :] == candidate[row, 0])
                and int(candidate[row, 0]) != keeper
            } | {
                int(candidate[0, col])
                for col in range(cols)
                if np.all(candidate[:, col] == candidate[0, col])
                and int(candidate[0, col]) != keeper
            }
            return any(
                frequency[color] < frequency[keeper]
                for color in complete_distractor_colors
            )

        try:
            return retry(sample_grid, valid, max_attempts=80)
        except ValueError:
            fallback = np.zeros((rows, cols), dtype=int)
            base_colors = distractors[1:] if gridvars["lower_frequency_line"] else distractors
            for index in range(rows * cols):
                row, col = divmod(index, cols)
                fallback[row, col] = base_colors[index % len(base_colors)]
            if gridvars["orientation"] == "row":
                fallback[0, :] = keeper
                keeper_lines = {0}
                if gridvars["multiple"]:
                    fallback[1, :] = keeper
                    keeper_lines.add(1)
                lower_line = rows - 1 if gridvars["lower_frequency_line"] else None
                if lower_line is not None:
                    fallback[lower_line, :] = distractors[0]
                available = [
                    (row, col)
                    for row in range(rows)
                    for col in range(cols)
                    if row not in keeper_lines and row != lower_line
                ]
            else:
                fallback[:, 0] = keeper
                keeper_lines = {0}
                if gridvars["multiple"]:
                    fallback[:, 1] = keeper
                    keeper_lines.add(1)
                lower_line = cols - 1 if gridvars["lower_frequency_line"] else None
                if lower_line is not None:
                    fallback[:, lower_line] = distractors[0]
                available = [
                    (row, col)
                    for row in range(rows)
                    for col in range(cols)
                    if col not in keeper_lines and col != lower_line
                ]
            for row, col in available[:2]:
                fallback[row, col] = keeper
            return fallback

    def transform_input(self, grid, taskvars):
        replacement = taskvars["replacement_color"]
        colors, counts = np.unique(grid, return_counts=True)
        eligible = [
            (int(count), int(color))
            for color, count in zip(colors, counts)
            if int(color) != replacement
        ]
        if not eligible:
            return grid.copy()
        keeper = max(eligible)[1]
        return np.where(grid == keeper, keeper, replacement)
