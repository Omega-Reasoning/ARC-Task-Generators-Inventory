from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import random_cell_coloring, retry


class Taskc3e719e8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['tile_size']} by {vars['tile_size']} square whose cells are all nonempty colors.",
            "2. Exactly one visible color occurs more often than every other color.",
            "3. The modal color and the positions where it occurs vary between examples.",
            "4. At least two nonmodal colors remain, so the complete input forms a nontrivial reusable tile.",
            "5. The output will have one tile-sized macro position for every input cell.",
        ]
        transformation_reasoning_chain = [
            "1. Count the input colors and identify the unique most frequent color.",
            "2. Create an empty square output with side length {vars['tile_size']} squared.",
            "3. For each input cell having the modal color, paste one exact copy of the entire input into the corresponding macro position.",
            "4. Leave every macro position associated with a nonmodal input cell empty.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        tile_size = random.choice([3, 4, 5])
        taskvars = {"tile_size": tile_size}
        first_modal_count = tile_size * tile_size // 4 + 1
        train_gridvars = [
            {"modal_count": first_modal_count, "arrangement": "scatter"},
            {"modal_count": first_modal_count + 1, "arrangement": "diagonal"},
            {"modal_count": first_modal_count + 2, "arrangement": "cluster"},
            {"modal_count": first_modal_count + 3, "arrangement": "edge"},
        ]
        test_gridvars = {
            "modal_count": first_modal_count + 4,
            "arrangement": "cross",
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
        size = taskvars["tile_size"]
        modal_count = gridvars["modal_count"]
        colors = list(
            gridvars.get("colors", random.sample(range(1, 10), 4))
        )
        if (
            len(colors) != 4
            or len(set(colors)) != 4
            or any(color not in range(1, 10) for color in colors)
        ):
            raise ValueError("colors must contain four distinct nonzero ARC colors")
        modal_color, *nonmodal_colors = colors
        all_positions = [(row, col) for row in range(size) for col in range(size)]

        def ordered_positions():
            arrangement = gridvars["arrangement"]
            positions = list(all_positions)
            if arrangement == "scatter":
                random.shuffle(positions)
                return positions
            if arrangement == "diagonal":
                preferred = [(index, index) for index in range(size)]
            elif arrangement == "cluster":
                center = (random.randrange(size), random.randrange(size))
                preferred = sorted(
                    positions,
                    key=lambda position: (
                        abs(position[0] - center[0]) + abs(position[1] - center[1]),
                        random.random(),
                    ),
                )
            elif arrangement == "edge":
                preferred = [
                    position
                    for position in positions
                    if position[0] in (0, size - 1)
                    or position[1] in (0, size - 1)
                ]
                random.shuffle(preferred)
            else:
                center = size // 2
                preferred = [
                    position
                    for position in positions
                    if position[0] == center or position[1] == center
                ]
                random.shuffle(preferred)
            remaining = [position for position in positions if position not in preferred]
            random.shuffle(remaining)
            return preferred + remaining

        def sample_grid():
            sampled_grid = np.zeros((size, size), dtype=int)
            random_cell_coloring(
                sampled_grid,
                nonmodal_colors,
                density=1.0,
                background=0,
            )
            grid = np.asarray(
                gridvars.get("base_grid", sampled_grid), dtype=int
            ).copy()
            if grid.shape != (size, size):
                raise ValueError("base_grid must match tile_size")
            modal_positions = [
                (int(row), int(col))
                for row, col in gridvars.get(
                    "modal_positions", ordered_positions()[:modal_count]
                )
            ]
            if (
                len(modal_positions) != modal_count
                or len(set(modal_positions)) != modal_count
                or any(
                    not 0 <= row < size or not 0 <= col < size
                    for row, col in modal_positions
                )
            ):
                raise ValueError(
                    "modal_positions must contain modal_count distinct in-bounds cells"
                )
            for row, col in modal_positions:
                grid[row, col] = modal_color
            return grid

        def has_unique_mode(grid):
            values, counts = np.unique(grid, return_counts=True)
            modal_matches = counts[values == modal_color]
            return bool(
                len(values) >= 3
                and len(modal_matches) == 1
                and int(modal_matches[0]) == modal_count
                and np.count_nonzero(counts == counts.max()) == 1
                and int(values[np.argmax(counts)]) == modal_color
            )

        try:
            return retry(sample_grid, has_unique_mode, max_attempts=80)
        except ValueError:
            grid = np.zeros((size, size), dtype=int)
            modal_positions = {
                (int(row), int(col))
                for row, col in gridvars.get(
                    "modal_positions", ordered_positions()[:modal_count]
                )
            }
            nonmodal_index = 0
            for position in all_positions:
                if position in modal_positions:
                    grid[position] = modal_color
                else:
                    grid[position] = nonmodal_colors[
                        nonmodal_index % len(nonmodal_colors)
                    ]
                    nonmodal_index += 1
            return grid

    def transform_input(self, grid, taskvars):
        values, counts = np.unique(grid, return_counts=True)
        modal_color = int(values[np.argmax(counts)])
        size = taskvars["tile_size"]
        output = np.zeros((size * size, size * size), dtype=int)
        for row in range(size):
            for col in range(size):
                if int(grid[row, col]) == modal_color:
                    output[
                        row * size : (row + 1) * size,
                        col * size : (col + 1) * size,
                    ] = grid
        return output
