from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task90c28cc7Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['canvas_size']}-by-{vars['canvas_size']} {color('background_color')} canvas containing one translated solid rectangular mosaic.",
            "2. The mosaic encodes a coarse grid with at least {vars['minimum_band_count']} row bands and at least {vars['minimum_band_count']} column bands.",
            "3. Each coarse cell is expanded to a constant-color rectangle, with row-band heights and column-band widths allowed to differ.",
            "4. Adjacent coarse cells may share a color and merge visually, but every true row and column boundary is exposed by a color change somewhere else along that boundary.",
            "5. Coarse dimensions, band sizes, palette, color arrangement, and mosaic offset vary between examples."
        ]
        transformation_reasoning_chain = [
            "1. Crop the solid mosaic away from the surrounding {color('background_color')} canvas.",
            "2. Start a new row band whenever any column changes color between consecutive mosaic rows.",
            "3. Start a new column band whenever any row changes color between consecutive mosaic columns.",
            "4. At every recovered row-band and column-band intersection, take the constant cell color as one output cell.",
            "5. Return the reconstructed coarse grid, preserving its color layout and orientation."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, object], TrainTestData]:
        taskvars = {
            "background_color": 0,
            "canvas_size": random.randint(21, 29),
            "minimum_band_count": 2,
        }
        train_examples = []
        train_shapes = [(2, 2), (2, 3), (3, 2), (3, 3)]
        for phase, (row_count, column_count) in enumerate(train_shapes):
            input_grid = self.create_input(
                taskvars,
                {
                    "row_count": row_count,
                    "column_count": column_count,
                    "phase": phase,
                },
            )
            train_examples.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )

        test_input = self.create_input(
            taskvars,
            {"row_count": 4, "column_count": 4, "phase": 4},
        )
        test_example = {
            "input": test_input,
            "output": self.transform_input(test_input, taskvars),
        }
        return taskvars, {"train": train_examples, "test": [test_example]}

    def create_input(
        self, taskvars: dict[str, object], gridvars: dict[str, object]
    ) -> np.ndarray:
        background_color = int(taskvars["background_color"])
        canvas_size = int(taskvars["canvas_size"])
        row_count = int(gridvars["row_count"])
        column_count = int(gridvars["column_count"])
        phase = int(gridvars["phase"])
        palette_size = min(6, max(3, row_count + column_count - 1))
        sampled_palette = random.sample(range(1, 10), palette_size)
        random.shuffle(sampled_palette)
        palette = list(gridvars.get("palette", sampled_palette))

        def sample_coarse_grid() -> np.ndarray:
            candidate = random_cell_coloring(
                np.zeros((row_count, column_count), dtype=int),
                palette,
                density=1.0,
                background=background_color,
            )
            if phase % 2 == 1:
                candidate[0, 0] = palette[phase % len(palette)]
            return candidate

        def valid_coarse_grid(candidate: np.ndarray) -> bool:
            row_boundaries_visible = all(
                np.any(candidate[row, :] != candidate[row - 1, :])
                for row in range(1, row_count)
            )
            column_boundaries_visible = all(
                np.any(candidate[:, column] != candidate[:, column - 1])
                for column in range(1, column_count)
            )
            has_hidden_boundary_segment = bool(
                np.any(candidate[1:, :] == candidate[:-1, :])
                or np.any(candidate[:, 1:] == candidate[:, :-1])
            )
            return bool(
                len(np.unique(candidate)) >= 2
                and row_boundaries_visible
                and column_boundaries_visible
                and has_hidden_boundary_segment
            )

        coarse_grid = retry(
            sample_coarse_grid, valid_coarse_grid, max_attempts=100
        )
        coarse_grid = np.asarray(
            gridvars.get("coarse_grid", coarse_grid),
            dtype=int,
        )
        if coarse_grid.shape != (row_count, column_count):
            raise ValueError("coarse_grid shape must match the requested band counts")

        def sample_row_heights() -> list[int]:
            return [random.randint(2, 6) for _ in range(row_count)]

        def sample_column_widths() -> list[int]:
            return [random.randint(2, 6) for _ in range(column_count)]

        row_heights = retry(
            sample_row_heights,
            lambda values: sum(values) <= canvas_size - 2
            and len(set(values)) >= 2,
            max_attempts=100,
        )
        row_heights = list(gridvars.get("row_heights", row_heights))
        column_widths = retry(
            sample_column_widths,
            lambda values: sum(values) <= canvas_size - 2
            and len(set(values)) >= 2,
            max_attempts=100,
        )
        column_widths = list(gridvars.get("column_widths", column_widths))
        mosaic = np.repeat(coarse_grid, row_heights, axis=0)
        mosaic = np.repeat(mosaic, column_widths, axis=1)
        row_offset = gridvars.get(
            "row_offset",
            random.randint(0, canvas_size - mosaic.shape[0]),
        )
        column_offset = gridvars.get(
            "column_offset",
            random.randint(0, canvas_size - mosaic.shape[1]),
        )
        grid = np.full(
            (canvas_size, canvas_size), background_color, dtype=int
        )
        grid[
            row_offset : row_offset + mosaic.shape[0],
            column_offset : column_offset + mosaic.shape[1],
        ] = mosaic
        return grid

    def transform_input(
        self, grid: np.ndarray, taskvars: dict[str, object]
    ) -> np.ndarray:
        background_color = int(taskvars["background_color"])
        canvas_size = int(taskvars["canvas_size"])
        minimum_band_count = int(taskvars["minimum_band_count"])
        if grid.shape != (canvas_size, canvas_size):
            return np.array(grid, dtype=int, copy=True)

        objects = list(
            find_connected_objects(
                grid,
                diagonal_connectivity=False,
                background=background_color,
                monochromatic=False,
            )
        )
        if len(objects) == 0:
            return np.array(grid, dtype=int, copy=True)
        mosaic_cells = list(max(objects, key=lambda obj: len(list(obj))))
        rows = [row for row, _, _ in mosaic_cells]
        columns = [column for _, column, _ in mosaic_cells]
        crop = np.array(
            grid[
                min(rows) : max(rows) + 1,
                min(columns) : max(columns) + 1,
            ],
            dtype=int,
            copy=True,
        )

        row_starts = [0]
        for row in range(1, crop.shape[0]):
            if np.any(crop[row, :] != crop[row - 1, :]):
                row_starts.append(row)
        column_starts = [0]
        for column in range(1, crop.shape[1]):
            if np.any(crop[:, column] != crop[:, column - 1]):
                column_starts.append(column)
        if (
            len(row_starts) < minimum_band_count
            or len(column_starts) < minimum_band_count
        ):
            return crop

        output = np.zeros(
            (len(row_starts), len(column_starts)), dtype=int
        )
        for output_row, source_row in enumerate(row_starts):
            for output_column, source_column in enumerate(column_starts):
                output[output_row, output_column] = crop[
                    source_row, source_column
                ]
        return output
