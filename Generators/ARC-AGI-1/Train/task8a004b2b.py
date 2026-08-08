from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import GridObject, find_connected_objects
import numpy as np
import random


class Task8a004b2bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. The input contains four {color('marker_color')} cells at the corners of a rectangular output panel on a {color('background_color')} field.",
            "2. Inside that panel are at least two solid colored square anchors sharing one side length between {vars['minimum_block_size']} and {vars['maximum_block_size']}.",
            "3. Outside the marker-bounded panel is a much smaller multicolored blueprint whose non-{color('background_color')} cells define a pattern at unit-cell resolution.",
            "4. Anchor colors and relative positions occur in the blueprint and determine both its alignment in the panel and the side length used to enlarge each blueprint cell.",
            "5. Panel position, panel dimensions, block size, blueprint geometry, colors, and the visible anchor subset vary across examples.",
        ]
        transformation_reasoning_chain = [
            "1. Use the four {color('marker_color')} corner cells to extract their inclusive rectangular panel.",
            "2. Isolate the colored blueprint outside the panel and identify the solid square anchor blocks already visible inside it.",
            "3. Infer the common anchor side length and align blueprint coordinates with panel coordinates by matching anchor colors and positions.",
            "4. Expand every non-{color('background_color')} blueprint cell into a solid square block of that side length at the aligned panel location.",
            "5. Return the completed marker-bounded panel, preserving its four {color('marker_color')} corners and all original anchors.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        taskvars = {
            "background_color": 0,
            "marker_color": random.randint(1, 9),
            "minimum_block_size": 2,
            "maximum_block_size": 4,
        }
        train_specs = [
            {
                "blueprint_shape": (2, 3),
                "occupied_count": 4,
                "block_size": 2,
                "seed_count": 3,
            },
            {
                "blueprint_shape": (3, 3),
                "occupied_count": 5,
                "block_size": 3,
                "seed_count": 2,
            },
            {
                "blueprint_shape": (2, 4),
                "occupied_count": 6,
                "block_size": 2,
                "seed_count": 3,
            },
            {
                "blueprint_shape": (3, 4),
                "occupied_count": 7,
                "block_size": 4,
                "seed_count": 2,
            },
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
            {
                "blueprint_shape": (4, 4),
                "occupied_count": 8,
                "block_size": 3,
                "seed_count": 3,
            },
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
        background = taskvars["background_color"]
        marker_color = taskvars["marker_color"]
        blueprint_height, blueprint_width = gridvars["blueprint_shape"]
        occupied_count = gridvars["occupied_count"]
        block_size = gridvars["block_size"]

        def sample_occupancy():
            return create_object(
                blueprint_height,
                blueprint_width,
                1,
                contiguity=Contiguity.EIGHT,
                background=background,
            ) != background

        def occupancy_is_valid(mask):
            return (
                int(np.sum(mask)) == occupied_count
                and np.any(mask[0, :])
                and np.any(mask[-1, :])
                and np.any(mask[:, 0])
                and np.any(mask[:, -1])
            )

        def sample_valid_occupancy():
            try:
                return retry(
                    sample_occupancy,
                    occupancy_is_valid,
                    max_attempts=160,
                )
            except ValueError:
                fallback = np.zeros(
                    (blueprint_height, blueprint_width),
                    dtype=bool,
                )
                fallback_cells = {
                    (0, column) for column in range(blueprint_width)
                } | {
                    (row, blueprint_width - 1)
                    for row in range(blueprint_height)
                }
                remaining_cells = [
                    (row, column)
                    for row in range(1, blueprint_height)
                    for column in range(blueprint_width - 2, -1, -1)
                ]
                for coordinate in remaining_cells:
                    if len(fallback_cells) >= occupied_count:
                        break
                    fallback_cells.add(coordinate)
                for row, column in fallback_cells:
                    fallback[row, column] = True
                return fallback

        occupancy = np.asarray(
            gridvars.get("occupancy", sample_valid_occupancy()), dtype=bool
        )

        available_colors = [
            color
            for color in range(1, 10)
            if color != marker_color
        ]
        fill_color, first_anchor_color, second_anchor_color = gridvars.get(
            "colors",
            random.sample(
                available_colors,
                3,
            )
        )
        occupied_cells = list(map(tuple, np.argwhere(occupancy)))
        anchor_pair = tuple(
            tuple(coordinate)
            for coordinate in gridvars.get(
                "anchor_pair",
                random.sample(occupied_cells, 2),
            )
        )
        blueprint = np.full(
            (blueprint_height, blueprint_width),
            background,
            dtype=int,
        )
        blueprint[occupancy] = fill_color
        blueprint[anchor_pair[0]] = first_anchor_color
        blueprint[anchor_pair[1]] = second_anchor_color

        top_margin = gridvars.get("top_margin", random.randint(1, 3))
        bottom_margin = gridvars.get("bottom_margin", random.randint(1, 3))
        left_margin = gridvars.get("left_margin", random.randint(1, 3))
        right_margin = gridvars.get("right_margin", random.randint(1, 3))
        panel_height = (
            top_margin
            + blueprint_height * block_size
            + bottom_margin
        )
        panel_width = (
            left_margin
            + blueprint_width * block_size
            + right_margin
        )
        panel_top = gridvars.get("panel_top", random.randint(0, 2))
        panel_left = gridvars.get("panel_left", random.randint(0, 2))
        blueprint_top = gridvars.get(
            "blueprint_top",
            panel_top + panel_height + random.randint(2, 3),
        )
        blueprint_left = gridvars.get(
            "blueprint_left",
            random.randint(
                0,
                max(0, panel_left + panel_width - blueprint_width),
            ),
        )
        input_height = gridvars.get(
            "input_height", blueprint_top + blueprint_height + 1
        )
        input_width = gridvars.get(
            "input_width",
            max(
                panel_left
                + panel_width
                + gridvars.get("input_right_padding", random.randint(0, 2)),
                blueprint_left + blueprint_width + 1,
            ),
        )
        grid = np.full(
            (input_height, input_width),
            background,
            dtype=int,
        )

        marker_cells = {
            (panel_top, panel_left, marker_color),
            (panel_top, panel_left + panel_width - 1, marker_color),
            (panel_top + panel_height - 1, panel_left, marker_color),
            (
                panel_top + panel_height - 1,
                panel_left + panel_width - 1,
                marker_color,
            ),
        }
        GridObject(marker_cells).paste(
            grid,
            overwrite=False,
            background=background,
        )

        seed_cells = [anchor_pair[0], anchor_pair[1]]
        fill_cells = [
            coordinate
            for coordinate in occupied_cells
            if coordinate not in seed_cells
        ]
        if gridvars["seed_count"] > 2 and fill_cells:
            seed_cells.append(
                tuple(
                    gridvars.get(
                        "extra_seed_cell",
                        random.choice(fill_cells),
                    )
                )
            )
        for blueprint_row, blueprint_column in seed_cells:
            block = np.full(
                (block_size, block_size),
                blueprint[blueprint_row, blueprint_column],
                dtype=int,
            )
            GridObject.from_array(
                block,
                offset=(
                    panel_top + top_margin + blueprint_row * block_size,
                    panel_left + left_margin + blueprint_column * block_size,
                ),
            ).paste(grid, overwrite=False, background=background)

        GridObject.from_array(
            blueprint,
            offset=(blueprint_top, blueprint_left),
        ).paste(grid, overwrite=False, background=background)
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        background = taskvars["background_color"]
        marker_color = taskvars["marker_color"]
        marker_positions = list(map(tuple, np.argwhere(grid == marker_color)))
        if len(marker_positions) != 4:
            return grid.copy()
        panel_top = min(row for row, _ in marker_positions)
        panel_bottom = max(row for row, _ in marker_positions)
        panel_left = min(column for _, column in marker_positions)
        panel_right = max(column for _, column in marker_positions)
        panel = grid[
            panel_top : panel_bottom + 1,
            panel_left : panel_right + 1,
        ].copy()

        outside_cells = [
            (row, column)
            for row in range(grid.shape[0])
            for column in range(grid.shape[1])
            if (
                grid[row, column] not in (background, marker_color)
                and not (
                    panel_top <= row <= panel_bottom
                    and panel_left <= column <= panel_right
                )
            )
        ]
        if not outside_cells:
            return panel
        blueprint_top = min(row for row, _ in outside_cells)
        blueprint_bottom = max(row for row, _ in outside_cells)
        blueprint_left = min(column for _, column in outside_cells)
        blueprint_right = max(column for _, column in outside_cells)
        blueprint = grid[
            blueprint_top : blueprint_bottom + 1,
            blueprint_left : blueprint_right + 1,
        ].copy()

        anchor_grid = panel.copy()
        anchor_grid[anchor_grid == marker_color] = background
        anchor_objects = find_connected_objects(
            anchor_grid,
            diagonal_connectivity=False,
            background=background,
            monochromatic=True,
        ).sort_by_size(reverse=True)
        if len(anchor_objects) == 0:
            return panel
        object_dimensions = []
        for object_ in anchor_objects:
            box = object_.bounding_box
            object_dimensions.append(
                min(
                    box[0].stop - box[0].start,
                    box[1].stop - box[1].start,
                )
            )
        block_size = min(object_dimensions)

        candidates = set()
        blueprint_cells = [
            (row, column, int(blueprint[row, column]))
            for row, column in map(tuple, np.argwhere(blueprint != background))
        ]
        for object_ in anchor_objects:
            box = object_.bounding_box
            object_color = next(iter(object_.colors))
            for blueprint_row, blueprint_column, color in blueprint_cells:
                if color == object_color:
                    candidates.add(
                        (
                            box[0].start - blueprint_row * block_size,
                            box[1].start - blueprint_column * block_size,
                        )
                    )

        valid_candidates = []
        for offset_row, offset_column in candidates:
            if not (
                0 <= offset_row
                and 0 <= offset_column
                and offset_row + blueprint.shape[0] * block_size
                <= panel.shape[0]
                and offset_column + blueprint.shape[1] * block_size
                <= panel.shape[1]
            ):
                continue
            compatible = True
            for row, column in map(
                tuple,
                np.argwhere(
                    (panel != background) & (panel != marker_color)
                ),
            ):
                relative_row = row - offset_row
                relative_column = column - offset_column
                if not (
                    0 <= relative_row < blueprint.shape[0] * block_size
                    and 0 <= relative_column
                    < blueprint.shape[1] * block_size
                ):
                    compatible = False
                    break
                expected_color = blueprint[
                    relative_row // block_size,
                    relative_column // block_size,
                ]
                if expected_color != panel[row, column]:
                    compatible = False
                    break
            if compatible:
                score = 0
                for object_ in anchor_objects:
                    box = object_.bounding_box
                    object_color = next(iter(object_.colors))
                    for blueprint_row, blueprint_column, color in blueprint_cells:
                        if (
                            color == object_color
                            and box[0].start
                            == offset_row + blueprint_row * block_size
                            and box[1].start
                            == offset_column + blueprint_column * block_size
                        ):
                            score += 1
                            break
                valid_candidates.append((score, offset_row, offset_column))
        if not valid_candidates:
            return panel
        _, offset_row, offset_column = max(valid_candidates)

        output = panel.copy()
        for blueprint_row, blueprint_column, color in blueprint_cells:
            block = np.full(
                (block_size, block_size),
                color,
                dtype=int,
            )
            GridObject.from_array(
                block,
                offset=(
                    offset_row + blueprint_row * block_size,
                    offset_column + blueprint_column * block_size,
                ),
            ).paste(output, overwrite=True, background=background)
        return output
