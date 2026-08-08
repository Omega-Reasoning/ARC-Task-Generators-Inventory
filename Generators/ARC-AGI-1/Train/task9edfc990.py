from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task9edfc990Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input is a dense multicolored grid containing {color('background_color')} cells.",
            "2. The {color('background_color')} cells form several 4-connected regions.",
            "3. Some regions are cardinally adjacent to {color('fill_color')} cells and others are not.",
            "4. Grid dimensions, foreground texture, and component geometry vary by example.",
        ]
        transformation_reasoning_chain = [
            "1. Identify all 4-connected regions of {color('background_color')} cells.",
            "2. For each region, test whether any cell has a cardinally adjacent {color('fill_color')} neighbor.",
            "3. Recolor the complete contacted region with {color('fill_color')} and leave every uncontacted region unchanged.",
            "4. Preserve all original foreground cells.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            'background_color': 0,
            'fill_color': random.randint(1, 9),
        }
        specs = [
            {'rows': 12, 'cols': 13, 'density': 0.55},
            {'rows': 14, 'cols': 15, 'density': 0.62},
            {'rows': 15, 'cols': 13, 'density': 0.68},
            {'rows': 16, 'cols': 16, 'density': 0.74},
        ]
        random.shuffle(specs)
        train = []
        for spec in specs:
            input_grid = self.create_input(taskvars, spec)
            train.append({'input': input_grid, 'output': self.transform_input(input_grid, taskvars)})
        test_input = self.create_input(taskvars, {'rows': 17, 'cols': 18, 'density': 0.59})
        return taskvars, {'train': train, 'test': [{'input': test_input, 'output': self.transform_input(test_input, taskvars)}]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        background = taskvars['background_color']
        fill_color = taskvars['fill_color']
        other_colors = list(
            gridvars.get(
                "other_colors",
                random.sample(
                    [color for color in range(1, 10) if color != fill_color],
                    5,
                ),
            )
        )
        palette = [fill_color] + other_colors

        def classifications(candidate):
            mask = (candidate == background).astype(int)
            regions = find_connected_objects(mask, diagonal_connectivity=False, background=0)
            contacted_sizes = []
            uncontacted_sizes = []
            for region in regions:
                contacted = any(
                    0 <= row + dr < candidate.shape[0]
                    and 0 <= col + dc < candidate.shape[1]
                    and candidate[row + dr, col + dc] == fill_color
                    for row, col in region.coords
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                )
                (contacted_sizes if contacted else uncontacted_sizes).append(len(region))
            return contacted_sizes, uncontacted_sizes

        def sample_grid():
            candidate = np.full((gridvars['rows'], gridvars['cols']), background, dtype=int)
            return random_cell_coloring(
                candidate,
                palette,
                density=gridvars['density'],
                background=background,
            )

        try:
            sampled_grid = retry(
                sample_grid,
                lambda candidate: (
                    bool(classifications(candidate)[1])
                    and any(size >= 2 for size in classifications(candidate)[0])
                ),
                max_attempts=100,
            )
        except ValueError:
            rows, cols = gridvars['rows'], gridvars['cols']
            sampled_grid = np.full((rows, cols), other_colors[0], dtype=int)
            sampled_grid[1:3, 1:3] = background
            sampled_grid[1, 3] = fill_color
            sampled_grid[rows - 2, cols - 2] = background
        input_grid = np.asarray(
            gridvars.get("input_grid", sampled_grid),
            dtype=int,
        )
        if input_grid.shape != (gridvars['rows'], gridvars['cols']):
            raise ValueError("input_grid shape does not match rows and cols")
        if not set(int(value) for value in np.unique(input_grid)).issubset(
            {background, *[int(color) for color in palette]}
        ):
            raise ValueError("input_grid contains a color outside the routed palette")
        contacted, uncontacted = classifications(input_grid)
        if not uncontacted or not any(size >= 2 for size in contacted):
            raise ValueError("input_grid must contain contacted and uncontacted regions")
        return input_grid.copy()

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background = taskvars['background_color']
        fill_color = taskvars['fill_color']
        output = grid.copy()
        mask = (grid == background).astype(int)
        regions = find_connected_objects(mask, diagonal_connectivity=False, background=0)
        for region in regions:
            touches_fill = any(
                0 <= row + dr < grid.shape[0]
                and 0 <= col + dc < grid.shape[1]
                and grid[row + dr, col + dc] == fill_color
                for row, col in region.coords
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
            if touches_fill:
                for row, col in region.coords:
                    output[row, col] = fill_color
        return output
