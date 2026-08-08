from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random


class Task9dfd6313Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input is a square grid with {color('background_color')} background.",
            "2. Its main diagonal is completely filled with {color('diagonal_color')} cells.",
            "3. Additional colored cells occur strictly below the main diagonal.",
            "4. Their counts, colors, and lower-triangular positions vary by example.",
        ]
        transformation_reasoning_chain = [
            "1. Keep each {color('diagonal_color')} main-diagonal cell fixed.",
            "2. For every other cell, exchange its row and column coordinates.",
            "3. Preserve all colors and return the transposed square grid.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            'diagonal_color': random.randint(1, 9),
            'background_color': 0,
        }
        sizes = random.sample(range(3, 9), 5)
        train = []
        for size in sizes[:4]:
            input_grid = self.create_input(taskvars, {'size': size, 'density': random.uniform(0.3, 0.65)})
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        test_size = sizes[4]
        test_input = self.create_input(taskvars, {'size': test_size, 'density': 0.55})
        return taskvars, {'train': train, 'test': [{'input': test_input, 'output': self.transform_input(test_input, taskvars)}]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        size = gridvars['size']
        background = taskvars['background_color']
        diagonal_color = taskvars['diagonal_color']
        palette = list(
            gridvars.get(
                "palette",
                random.sample(
                    [color for color in range(1, 10) if color != diagonal_color],
                    random.randint(2, 4),
                ),
            )
        )

        def sample_grid():
            candidate = np.full((size, size), background, dtype=int)
            random_cell_coloring(
                candidate,
                palette,
                density=gridvars['density'],
                background=background,
            )
            candidate[np.triu_indices(size, 0)] = background
            np.fill_diagonal(candidate, diagonal_color)
            return candidate

        try:
            sampled_grid = retry(
                sample_grid,
                lambda candidate: int(np.count_nonzero(np.tril(candidate, -1))) >= 2,
                max_attempts=40,
            )
        except ValueError:
            sampled_grid = np.full((size, size), background, dtype=int)
            np.fill_diagonal(sampled_grid, diagonal_color)
            sampled_grid[1, 0] = palette[0]
            sampled_grid[size - 1, 0] = palette[1 if len(palette) > 1 else 0]
        input_grid = np.asarray(
            gridvars.get("input_grid", sampled_grid),
            dtype=int,
        )
        if input_grid.shape != (size, size):
            raise ValueError("input_grid shape does not match size")
        if not np.all(np.diag(input_grid) == diagonal_color):
            raise ValueError("input_grid must preserve the diagonal color")
        if np.any(input_grid[np.triu_indices(size, 1)] != background):
            raise ValueError("input_grid foreground must stay on or below the diagonal")
        if not set(int(value) for value in np.unique(input_grid)).issubset(
            {background, diagonal_color, *[int(color) for color in palette]}
        ):
            raise ValueError("input_grid contains a color outside the routed palette")
        return input_grid.copy()

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        return np.transpose(grid).copy()
