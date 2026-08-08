from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import Contiguity, create_object, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random


class Task80af3007Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains a raster of equal, completely filled {vars['block_size']}-by-{vars['block_size']} {color('object_color')} blocks on an empty background.",
            "2. The possible block locations form a {vars['motif_size']}-by-{vars['motif_size']} coarse square lattice.",
            "3. Some lattice locations contain a block and the others are empty, thereby encoding a binary coarse motif.",
            "4. The occupied coarse motif touches every row and column of its lattice, while its detailed occupancy varies by example.",
            "5. The block raster can be translated within a larger nuisance canvas."
        ]
        transformation_reasoning_chain = [
            "1. Crop the bounding raster of {color('object_color')} cells and divide it into {vars['block_size']}-by-{vars['block_size']} lattice cells.",
            "2. Reduce each full block to one occupied cell to recover the {vars['motif_size']}-by-{vars['motif_size']} coarse motif.",
            "3. Replace every occupied coarse cell with a complete copy of that motif and every empty coarse cell with an empty motif-sized patch.",
            "4. Render the resulting {vars['motif_size'] * vars['motif_size']}-square self-similar output in {color('object_color')}."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict[str, object], TrainTestData]:
        taskvars = {
            "motif_size": random.choice([3, 4]),
            "block_size": random.randint(2, 4),
            "object_color": random.randint(1, 9),
        }
        train_examples = []
        for pattern_kind in ["sparse", "diagonal", "branching", "dense"]:
            input_grid = self.create_input(
                taskvars, {"pattern_kind": pattern_kind}
            )
            train_examples.append(
                {
                    "input": input_grid,
                    "output": self.transform_input(input_grid, taskvars),
                }
            )

        test_input = self.create_input(
            taskvars, {"pattern_kind": "hollow_corner"}
        )
        test_example = {
            "input": test_input,
            "output": self.transform_input(test_input, taskvars),
        }
        return taskvars, {"train": train_examples, "test": [test_example]}

    def create_input(
        self, taskvars: dict[str, object], gridvars: dict[str, object]
    ) -> np.ndarray:
        motif_size = int(taskvars["motif_size"])
        block_size = int(taskvars["block_size"])
        object_color = int(taskvars["object_color"])
        pattern_kind = str(gridvars["pattern_kind"])

        def sample_motif() -> np.ndarray:
            return create_object(
                motif_size,
                motif_size,
                color_palette=object_color,
                contiguity=Contiguity.EIGHT,
                background=0,
            )

        def valid(mask: np.ndarray) -> bool:
            occupied = int(np.count_nonzero(mask))
            spans = bool(
                np.all(np.any(mask != 0, axis=0))
                and np.all(np.any(mask != 0, axis=1))
            )
            if pattern_kind == "sparse":
                count_ok = motif_size <= occupied <= 2 * motif_size
            elif pattern_kind == "dense":
                count_ok = motif_size * motif_size - 4 <= occupied
                count_ok = count_ok and occupied <= motif_size * motif_size - 2
            else:
                count_ok = motif_size + 1 <= occupied <= motif_size * motif_size - 2
            return bool(spans and count_ok)

        def sample_pattern_motif():
            if pattern_kind == "hollow_corner":
                candidate = np.zeros((motif_size, motif_size), dtype=int)
                candidate[:, 0] = object_color
                for row in range(motif_size):
                    candidate[
                        row,
                        min(motif_size - 1, row + 1),
                    ] = object_color
                candidate[-1, :] = object_color
                return candidate
            try:
                return retry(sample_motif, valid, max_attempts=100)
            except ValueError:
                candidate = np.zeros(
                    (motif_size, motif_size),
                    dtype=int,
                )
                if pattern_kind == "sparse":
                    np.fill_diagonal(candidate, object_color)
                    candidate[0, -1] = object_color
                    candidate[-1, 0] = object_color
                elif pattern_kind == "diagonal":
                    np.fill_diagonal(candidate, object_color)
                    candidate[:, -1] = object_color
                elif pattern_kind == "branching":
                    candidate[motif_size // 2, :] = object_color
                    candidate[:, motif_size // 2] = object_color
                    candidate[0, 0] = object_color
                    candidate[-1, -1] = object_color
                else:
                    candidate[:, :] = object_color
                    candidate[0, -1] = 0
                    candidate[-1, 0] = 0
                return candidate

        motif = np.asarray(
            gridvars.get("motif", sample_pattern_motif()),
            dtype=int,
        )

        raster_size = motif_size * block_size
        rows = gridvars.get(
            "rows",
            random.randint(raster_size + 2, min(30, raster_size + 10)),
        )
        cols = gridvars.get(
            "cols",
            random.randint(raster_size + 2, min(30, raster_size + 10)),
        )
        row_offset = gridvars.get(
            "row_offset",
            random.randint(0, rows - raster_size),
        )
        col_offset = gridvars.get(
            "col_offset",
            random.randint(0, cols - raster_size),
        )
        grid = np.zeros((rows, cols), dtype=int)
        for motif_row in range(motif_size):
            for motif_col in range(motif_size):
                if motif[motif_row, motif_col] != 0:
                    row_start = row_offset + motif_row * block_size
                    col_start = col_offset + motif_col * block_size
                    grid[
                        row_start : row_start + block_size,
                        col_start : col_start + block_size,
                    ] = object_color
        return grid

    def transform_input(
        self, grid: np.ndarray, taskvars: dict[str, object]
    ) -> np.ndarray:
        motif_size = taskvars["motif_size"]
        block_size = taskvars["block_size"]
        object_color = taskvars["object_color"]
        objects = find_connected_objects(
            grid, diagonal_connectivity=False, background=0, monochromatic=True
        ).with_color(object_color)
        cells = [cell for obj in objects for cell in obj]
        rows = [row for row, _, _ in cells]
        cols = [col for _, col, _ in cells]
        crop = grid[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1]

        motif = np.zeros((motif_size, motif_size), dtype=int)
        for motif_row in range(motif_size):
            for motif_col in range(motif_size):
                row_start = motif_row * block_size
                col_start = motif_col * block_size
                region = crop[
                    row_start : row_start + block_size,
                    col_start : col_start + block_size,
                ]
                if np.all(region == object_color):
                    motif[motif_row, motif_col] = 1
        return np.kron(motif, motif) * object_color
