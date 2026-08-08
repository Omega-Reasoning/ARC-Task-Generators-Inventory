from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import retry
from Framework.transformation_library import GridObject, find_connected_objects


class Task150deff5Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input contains one connected figure made entirely of {color('source_color')} on a {color('background_color')} background.",
            "2. The figure is the nonoverlapping union of {vars['square_size']} by {vars['square_size']} square blocks and straight bars of length {vars['bar_length']}.",
            "3. Primitive pieces may touch, creating accidental overlapping candidate shapes in the combined silhouette.",
            "4. Exactly one selection of candidate squares and bars covers every source-colored cell once.",
        ]
        transformation_reasoning_chain = [
            "1. Enumerate every all-{color('source_color')} {vars['square_size']} by {vars['square_size']} square and every horizontal or vertical length-{vars['bar_length']} bar.",
            "2. Solve the exact-cover constraint so each {color('source_color')} cell belongs to one selected primitive.",
            "3. Recolor cells of selected square primitives {color('square_color')}.",
            "4. Recolor cells of selected bar primitives {color('bar_color')} while preserving the original canvas.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        source_color, square_color, bar_color = random.sample(range(1, 10), 3)
        taskvars = {
            "source_color": source_color,
            "square_color": square_color,
            "bar_color": bar_color,
            "background_color": 0,
            "square_size": 2,
            "bar_length": 3,
        }

        def make_pair(square_count: int, horizontal: int, vertical: int) -> GridPair:
            input_grid = self.create_input(
                taskvars,
                {
                    "square_count": square_count,
                    "horizontal_bars": horizontal,
                    "vertical_bars": vertical,
                },
            )
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [
            make_pair(2, 2, 0),
            make_pair(3, 0, 3),
            make_pair(2, 2, 2),
            make_pair(4, 1, 2),
        ]
        return taskvars, {"train": train, "test": [make_pair(3, 2, 3)]}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        source_color = taskvars["source_color"]
        background = taskvars["background_color"]
        square_size = taskvars["square_size"]
        bar_length = taskvars["bar_length"]
        piece_kinds = (
            ["square"] * gridvars["square_count"]
            + ["horizontal"] * gridvars["horizontal_bars"]
            + ["vertical"] * gridvars["vertical_bars"]
        )
        height = gridvars.get("height", random.randint(13, 18))
        width = gridvars.get("width", random.randint(14, 20))

        def cells_for(kind: str, top: int, left: int) -> set[tuple[int, int]]:
            if kind == "square":
                return {
                    (top + delta_row, left + delta_col)
                    for delta_row in range(square_size)
                    for delta_col in range(square_size)
                }
            if kind == "horizontal":
                return {(top, left + delta_col) for delta_col in range(bar_length)}
            return {(top + delta_row, left) for delta_row in range(bar_length)}

        def sample_figure() -> np.ndarray:
            def shuffled_piece_kinds() -> list[str]:
                kinds = piece_kinds.copy()
                random.shuffle(kinds)
                return kinds

            kinds = list(
                gridvars.get("piece_kinds", shuffled_piece_kinds())
            )
            occupied: set[tuple[int, int]] = set()
            placed = []
            for piece_index, kind in enumerate(kinds):
                accepted = None
                for _ in range(300):
                    if piece_index == 0:
                        top = gridvars.get(
                            f"piece_{piece_index}_top",
                            random.randint(2, height - 5),
                        )
                        left = gridvars.get(
                            f"piece_{piece_index}_left",
                            random.randint(2, width - 5),
                        )
                    else:
                        near_row, near_col = gridvars.get(
                            f"piece_{piece_index}_near_cell",
                            random.choice(sorted(occupied)),
                        )
                        top = gridvars.get(
                            f"piece_{piece_index}_top",
                            near_row + random.randint(-3, 1),
                        )
                        left = gridvars.get(
                            f"piece_{piece_index}_left",
                            near_col + random.randint(-3, 1),
                        )
                    cells = cells_for(kind, top, left)
                    in_bounds = all(
                        0 <= row < height and 0 <= col < width
                        for row, col in cells
                    )
                    touching = not occupied or any(
                        max(abs(row - other_row), abs(col - other_col)) == 1
                        for row, col in cells
                        for other_row, other_col in occupied
                    )
                    if in_bounds and not (cells & occupied) and touching:
                        accepted = cells
                        break
                if accepted is None:
                    return np.full((height, width), background, dtype=int)
                occupied.update(accepted)
                placed.append(accepted)

            grid = np.full((height, width), background, dtype=int)
            for cells in placed:
                GridObject(
                    {(row, col, source_color) for row, col in cells}
                ).paste(grid, background=background)
            return grid

        def cover_count(grid: np.ndarray) -> int:
            foreground = {
                (int(row), int(col))
                for row, col in np.argwhere(grid == source_color)
            }
            candidates = []
            for row in range(grid.shape[0]):
                for col in range(grid.shape[1]):
                    for kind in ("square", "horizontal", "vertical"):
                        cells = cells_for(kind, row, col)
                        if cells <= foreground:
                            candidates.append(cells)
            count = 0

            def search(remaining: set) -> None:
                nonlocal count
                if count >= 2:
                    return
                if not remaining:
                    count += 1
                    return
                anchor = min(
                    remaining,
                    key=lambda cell: sum(
                        1
                        for cells in candidates
                        if cell in cells and cells <= remaining
                    ),
                )
                for cells in candidates:
                    if anchor in cells and cells <= remaining:
                        search(remaining - cells)

            search(foreground)
            return count

        def valid_figure(grid: np.ndarray) -> bool:
            objects = find_connected_objects(
                grid,
                diagonal_connectivity=True,
                background=background,
                monochromatic=False,
            )
            return len(objects) == 1 and cover_count(grid) == 1

        try:
            return retry(sample_figure, valid_figure, max_attempts=240)
        except ValueError:
            grid = np.full((height, width), background, dtype=int)
            cursor_row, cursor_col = 1, 1
            for piece_index, kind in enumerate(piece_kinds):
                cells = cells_for(kind, cursor_row, cursor_col)
                GridObject(
                    {(row, col, source_color) for row, col in cells}
                ).paste(grid, background=background)
                cursor_col += 4
                if cursor_col + 3 >= width:
                    cursor_col = 1 + (piece_index % 2)
                    cursor_row += 4
            return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        source_color = taskvars["source_color"]
        square_color = taskvars["square_color"]
        bar_color = taskvars["bar_color"]
        square_size = taskvars["square_size"]
        bar_length = taskvars["bar_length"]
        foreground = {
            (int(row), int(col))
            for row, col in np.argwhere(grid == source_color)
        }
        candidates = []
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                square = {
                    (row + delta_row, col + delta_col)
                    for delta_row in range(square_size)
                    for delta_col in range(square_size)
                }
                if square and square <= foreground:
                    candidates.append(("square", square))
                horizontal = {
                    (row, col + delta_col) for delta_col in range(bar_length)
                }
                if horizontal and horizontal <= foreground:
                    candidates.append(("bar", horizontal))
                vertical = {
                    (row + delta_row, col) for delta_row in range(bar_length)
                }
                if vertical and vertical <= foreground:
                    candidates.append(("bar", vertical))

        solutions = []

        def search(remaining: set, chosen: list) -> None:
            if len(solutions) >= 2:
                return
            if not remaining:
                solutions.append(chosen.copy())
                return
            anchor = min(
                remaining,
                key=lambda cell: sum(
                    1
                    for _, cells in candidates
                    if cell in cells and cells <= remaining
                ),
            )
            for kind, cells in candidates:
                if anchor in cells and cells <= remaining:
                    search(remaining - cells, chosen + [(kind, cells)])

        search(foreground, [])
        if not solutions:
            return grid.copy()
        output = grid.copy()
        for kind, cells in solutions[0]:
            replacement = square_color if kind == "square" else bar_color
            for row, col in cells:
                output[row, col] = replacement
        return output
