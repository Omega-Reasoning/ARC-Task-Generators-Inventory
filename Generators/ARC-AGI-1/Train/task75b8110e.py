from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import GridObject
import numpy as np
import random


class Task75b8110eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is an {vars['panel_size']*2}x{vars['panel_size']*2} grid divided into four equal quadrants without separator lines.",
            "2. The top-left, top-right, bottom-left, and bottom-right quadrants contain sparse masks in {color('top_left_color')}, {color('top_right_color')}, {color('bottom_left_color')}, and {color('bottom_right_color')} respectively.",
            "3. Within each quadrant every non-mask cell is {color('background_color')}, and the four masks share aligned local row and column coordinates.",
            "4. Any subset of the four masks may occupy the same aligned position, including pairwise and higher-order overlaps.",
            "5. Mask shapes and overlap locations vary across examples while quadrant size, role colors, and overlay precedence remain fixed.",
        ]
        transformation_reasoning_chain = [
            "1. Align corresponding cells of all four {vars['panel_size']}x{vars['panel_size']} quadrants into one output position.",
            "2. Place the {color('top_right_color')} top-right mask first, so it wins every overlap in which it participates.",
            "3. Fill still-{color('background_color')} output cells from the {color('bottom_left_color')} bottom-left mask, then from the {color('bottom_right_color')} bottom-right mask.",
            "4. Fill any still-{color('background_color')} cells from the {color('top_left_color')} top-left mask.",
            "5. Leave an output cell {color('background_color')} only when all four aligned input cells are empty.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        colors = random.sample(range(1, 10), 4)
        taskvars = {
            "panel_size": 4,
            "background_color": 0,
            "top_left_color": colors[0],
            "top_right_color": colors[1],
            "bottom_left_color": colors[2],
            "bottom_right_color": colors[3],
        }
        train_specs = [
            {"density": 0.25, "include_four_way": False},
            {"density": 0.35, "include_four_way": False},
            {"density": 0.45, "include_four_way": False},
            {"density": 0.55, "include_four_way": False},
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

        test_input = self.create_input(
            taskvars,
            {"density": 0.4, "include_four_way": True},
        )
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
        panel_size = taskvars["panel_size"]
        background = taskvars["background_color"]
        role_colors = [
            taskvars["top_left_color"],
            taskvars["top_right_color"],
            taskvars["bottom_left_color"],
            taskvars["bottom_right_color"],
        ]
        pairwise_evidence = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 1, 0, 0),
            (1, 0, 1, 0),
            (1, 0, 0, 1),
            (0, 1, 1, 0),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        ]
        if gridvars.get("include_four_way", False):
            mandatory_patterns = [
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ]
        else:
            mandatory_patterns = pairwise_evidence

        def sample_scene():
            panels = []
            for role_color in role_colors:
                panel = np.full(
                    (panel_size, panel_size),
                    background,
                    dtype=int,
                )
                random_cell_coloring(
                    panel,
                    role_color,
                    density=float(gridvars["density"]),
                    background=background,
                    overwrite=False,
                )
                panels.append(panel)

            positions = random.sample(
                range(panel_size * panel_size),
                len(mandatory_patterns),
            )
            for flat_position, occupancy in zip(positions, mandatory_patterns):
                row, column = divmod(flat_position, panel_size)
                for panel_index, occupied in enumerate(occupancy):
                    panels[panel_index][row, column] = (
                        role_colors[panel_index] if occupied else background
                    )

            grid = np.full(
                (panel_size * 2, panel_size * 2),
                background,
                dtype=int,
            )
            grid[:panel_size, :panel_size] = panels[0]
            grid[:panel_size, panel_size:] = panels[1]
            grid[panel_size:, :panel_size] = panels[2]
            grid[panel_size:, panel_size:] = panels[3]
            return grid

        def four_way_overlap_count(grid):
            panels = [
                grid[:panel_size, :panel_size],
                grid[:panel_size, panel_size:],
                grid[panel_size:, :panel_size],
                grid[panel_size:, panel_size:],
            ]
            occupancy = np.stack([panel != background for panel in panels])
            return int(np.sum(np.all(occupancy, axis=0)))

        include_four_way = bool(gridvars.get("include_four_way", False))
        def sample_valid_scene() -> np.ndarray:
            try:
                return retry(
                    sample_scene,
                    lambda grid: (
                        four_way_overlap_count(grid) >= 3
                        if include_four_way
                        else four_way_overlap_count(grid) == 0
                    ),
                    max_attempts=50,
                )
            except ValueError:
                fallback_panels = [
                    np.full((panel_size, panel_size), background, dtype=int)
                    for _ in role_colors
                ]
                for flat_position, occupancy in enumerate(mandatory_patterns):
                    row, column = divmod(flat_position, panel_size)
                    for panel_index, occupied in enumerate(occupancy):
                        if occupied:
                            fallback_panels[panel_index][row, column] = (
                                role_colors[panel_index]
                            )
                fallback = np.full(
                    (panel_size * 2, panel_size * 2),
                    background,
                    dtype=int,
                )
                fallback[:panel_size, :panel_size] = fallback_panels[0]
                fallback[:panel_size, panel_size:] = fallback_panels[1]
                fallback[panel_size:, :panel_size] = fallback_panels[2]
                fallback[panel_size:, panel_size:] = fallback_panels[3]
                return fallback

        def split_panels(scene: np.ndarray) -> list[np.ndarray]:
            return [
                scene[:panel_size, :panel_size],
                scene[:panel_size, panel_size:],
                scene[panel_size:, :panel_size],
                scene[panel_size:, panel_size:],
            ]

        panels = [
            np.asarray(panel, dtype=int)
            for panel in gridvars.get(
                "panels",
                split_panels(sample_valid_scene()),
            )
        ]
        if len(panels) != 4 or any(
            panel.shape != (panel_size, panel_size)
            or np.any((panel != background) & (panel != role_colors[index]))
            for index, panel in enumerate(panels)
        ):
            raise ValueError("panels must be four aligned role-color masks")
        grid = np.full(
            (panel_size * 2, panel_size * 2),
            background,
            dtype=int,
        )
        grid[:panel_size, :panel_size] = panels[0]
        grid[:panel_size, panel_size:] = panels[1]
        grid[panel_size:, :panel_size] = panels[2]
        grid[panel_size:, panel_size:] = panels[3]
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        panel_size = taskvars["panel_size"]
        background = taskvars["background_color"]
        output = np.full((panel_size, panel_size), background, dtype=int)

        priority_panels = [
            (0, panel_size, taskvars["top_right_color"]),
            (panel_size, 0, taskvars["bottom_left_color"]),
            (panel_size, panel_size, taskvars["bottom_right_color"]),
            (0, 0, taskvars["top_left_color"]),
        ]
        for row_start, column_start, role_color in priority_panels:
            panel = grid[
                row_start : row_start + panel_size,
                column_start : column_start + panel_size,
            ]
            role_mask = np.where(panel == role_color, role_color, background)
            GridObject.from_array(role_mask).paste(
                output,
                overwrite=False,
                background=background,
            )

        return output
