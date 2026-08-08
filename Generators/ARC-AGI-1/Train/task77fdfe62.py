from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task77fdfe62Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains an inset rectangular frame of {color('frame_color')} cells with an even-sized square interior.",
            "2. The frame interior contains a sparse pattern of {color('mask_color')} cells on {color('background_color')}.",
            "3. Four non-frame label colors appear outside the frame at its top-left, top-right, bottom-left, and bottom-right corners.",
            "4. Each corner label is associated with the corresponding quadrant of the framed interior.",
            "5. Interior size, mask geometry, active quadrants, and the four label colors vary between examples.",
        ]
        transformation_reasoning_chain = [
            "1. Extract the square region strictly inside the two full {color('frame_color')} frame rows and columns.",
            "2. Read the four outer corner labels and divide the extracted square at its row and column midpoints.",
            "3. For each {color('mask_color')} cell, use the label from its top-left, top-right, bottom-left, or bottom-right quadrant.",
            "4. Return only the recolored interior square, with every non-mask position set to {color('background_color')}.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        frame_color, mask_color = random.sample(range(1, 10), 2)
        taskvars = {
            "frame_color": frame_color,
            "mask_color": mask_color,
            "background_color": 0,
        }
        train_specs = [
            {"inner_size": 2, "active_quadrants": [(0, 0), (0, 1), (1, 0)]},
            {
                "inner_size": 4,
                "active_quadrants": [(0, 0), (0, 1), (1, 0), (1, 1)],
            },
            {"inner_size": 6, "active_quadrants": [(0, 0), (1, 1)]},
            {"inner_size": 8, "active_quadrants": [(0, 1), (1, 0), (1, 1)]},
        ]
        test_spec = {
            "inner_size": 10,
            "active_quadrants": [(0, 0), (0, 1), (1, 0), (1, 1)],
        }

        def make_pair(gridvars: dict) -> GridPair:
            input_grid = self.create_input(taskvars, gridvars)
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [make_pair(spec) for spec in train_specs]
        return taskvars, {"train": train, "test": [make_pair(test_spec)]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        frame_color = taskvars["frame_color"]
        mask_color = taskvars["mask_color"]
        background_color = taskvars["background_color"]
        inner_size = gridvars["inner_size"]
        active_quadrants = set(tuple(value) for value in gridvars["active_quadrants"])
        grid_size = inner_size + 4
        midpoint = inner_size // 2
        label_colors = list(
            gridvars.get(
                "label_colors",
                random.sample(
                    [
                        color
                        for color in range(1, 10)
                        if color not in (frame_color, mask_color)
                    ],
                    4,
                ),
            )
        )
        if (
            len(label_colors) != 4
            or len(set(label_colors)) != 4
            or any(
                not isinstance(color, int)
                or color not in range(1, 10)
                or color in (frame_color, mask_color)
                for color in label_colors
            )
        ):
            raise ValueError("label_colors must be four distinct non-frame colors")

        def sample_core() -> np.ndarray:
            core = np.full(
                (inner_size, inner_size),
                background_color,
                dtype=int,
            )
            minimum_density = (len(active_quadrants) + 1) / (inner_size * inner_size)
            random_cell_coloring(
                core,
                mask_color,
                density=min(0.75, max(0.40, minimum_density)),
                background=background_color,
            )
            for quadrant_row in range(2):
                for quadrant_col in range(2):
                    if (quadrant_row, quadrant_col) in active_quadrants:
                        continue
                    row_slice = slice(
                        quadrant_row * midpoint,
                        (quadrant_row + 1) * midpoint,
                    )
                    col_slice = slice(
                        quadrant_col * midpoint,
                        (quadrant_col + 1) * midpoint,
                    )
                    core[row_slice, col_slice] = background_color
            return core

        def has_requested_quadrants(core: np.ndarray) -> bool:
            observed_quadrants = set()
            for quadrant_row in range(2):
                for quadrant_col in range(2):
                    region = core[
                        quadrant_row * midpoint : (quadrant_row + 1) * midpoint,
                        quadrant_col * midpoint : (quadrant_col + 1) * midpoint,
                    ]
                    if np.any(region == mask_color):
                        observed_quadrants.add((quadrant_row, quadrant_col))
            return bool(
                observed_quadrants == active_quadrants
                and np.any(core == background_color)
            )

        def sample_valid_core() -> np.ndarray:
            try:
                return retry(
                    sample_core,
                    has_requested_quadrants,
                    max_attempts=50,
                )
            except ValueError:
                fallback = np.full(
                    (inner_size, inner_size),
                    background_color,
                    dtype=int,
                )
                for quadrant_row, quadrant_col in active_quadrants:
                    row = quadrant_row * midpoint + midpoint // 2
                    col = quadrant_col * midpoint + midpoint // 2
                    fallback[row, col] = mask_color
                    if midpoint > 1:
                        fallback[
                            quadrant_row * midpoint,
                            quadrant_col * midpoint,
                        ] = mask_color
                return fallback

        core = np.asarray(
            gridvars.get("core", sample_valid_core()),
            dtype=int,
        )
        if core.shape != (inner_size, inner_size):
            raise ValueError("core must match inner_size")
        if np.any((core != background_color) & (core != mask_color)):
            raise ValueError("core may contain only mask and background colors")
        if not has_requested_quadrants(core):
            raise ValueError(
                "core must occupy exactly the requested quadrants and retain "
                "background"
            )

        grid = np.full((grid_size, grid_size), background_color, dtype=int)
        grid[1, :] = frame_color
        grid[-2, :] = frame_color
        grid[:, 1] = frame_color
        grid[:, -2] = frame_color
        grid[0, 0] = label_colors[0]
        grid[0, -1] = label_colors[1]
        grid[-1, 0] = label_colors[2]
        grid[-1, -1] = label_colors[3]
        grid[2:-2, 2:-2] = core
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        frame_color = taskvars['frame_color']
        mask_color = taskvars['mask_color']
        background_color = taskvars['background_color']
        frame_rows = [
            row
            for row in range(grid.shape[0])
            if np.all(grid[row, :] == frame_color)
        ]
        frame_cols = [
            col
            for col in range(grid.shape[1])
            if np.all(grid[:, col] == frame_color)
        ]
        row_top, row_bottom = frame_rows[0], frame_rows[-1]
        col_left, col_right = frame_cols[0], frame_cols[-1]
        core = grid[row_top + 1 : row_bottom, col_left + 1 : col_right]
        labels = (
            (
                int(grid[row_top - 1, col_left - 1]),
                int(grid[row_top - 1, col_right + 1]),
            ),
            (
                int(grid[row_bottom + 1, col_left - 1]),
                int(grid[row_bottom + 1, col_right + 1]),
            ),
        )
        row_midpoint = core.shape[0] // 2
        col_midpoint = core.shape[1] // 2
        output = np.full_like(core, background_color)
        mask_objects = find_connected_objects(
            core,
            diagonal_connectivity=False,
            background=background_color,
            monochromatic=True,
        ).with_color(mask_color)
        for obj in mask_objects:
            for row, col, _ in obj.cells:
                quadrant_row = 0 if row < row_midpoint else 1
                quadrant_col = 0 if col < col_midpoint else 1
                output[row, col] = labels[quadrant_row][quadrant_col]
        return output
