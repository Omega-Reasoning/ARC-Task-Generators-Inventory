from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import Contiguity, create_object, retry


class Task017c7c7bGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input is a {vars['input_rows']}-row by {vars['grid_cols']}-column grid containing only empty cells and {color('input_color')} cells.",
            "2. Reading from top to bottom, the rows form a short repeating sequence.",
            "3. The first occurrence of that row sequence is the motif that controls all later rows.",
            "4. Different examples use different motif periods and different arrangements of {color('input_color')} cells within their motif rows.",
            "5. All examples in the task share the same grid width, heights, and foreground color roles.",
        ]
        transformation_reasoning_chain = [
            "1. Find the shortest vertical row sequence whose repetition explains every row of the input.",
            "2. Continue that sequence cyclically until the grid has {vars['output_rows']} rows.",
            "3. Recolor every repeated {color('input_color')} cell to {color('output_color')}.",
            "4. Keep every empty cell empty and preserve the {vars['grid_cols']}-column width.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> tuple[dict, TrainTestData]:
        input_color, output_color = random.sample(range(1, 10), 2)
        input_rows = random.randint(6, 10)
        output_rows = random.randint(input_rows + 3, min(18, input_rows + 7))
        taskvars = {
            "input_color": input_color,
            "output_color": output_color,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "grid_cols": random.randint(3, 7),
        }

        def make_pair(period: int) -> GridPair:
            input_grid = self.create_input(taskvars, {"period": period})
            return {
                "input": input_grid,
                "output": self.transform_input(input_grid, taskvars),
            }

        train = [make_pair(period) for period in (2, 3, 2, 3)]
        test = [make_pair(4)]
        return taskvars, {"train": train, "test": test}

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        period = gridvars["period"]
        rows = taskvars["input_rows"]
        cols = taskvars["grid_cols"]
        color = taskvars["input_color"]

        def detected_period(candidate: np.ndarray) -> int:
            for offset in range(1, candidate.shape[0]):
                if np.array_equal(candidate[offset:], candidate[:-offset]):
                    return offset
            return candidate.shape[0]

        def sample_grid() -> np.ndarray:
            motif = np.asarray(
                gridvars.get(
                    "motif",
                    create_object(
                        period,
                        cols,
                        color,
                        contiguity=Contiguity.NONE,
                    ),
                ),
                dtype=int,
            )
            return np.vstack([motif[row % period] for row in range(rows)])

        try:
            return retry(
                sample_grid,
                lambda candidate: (
                    detected_period(candidate) == period
                    and np.all(np.any(candidate != 0, axis=1))
                    and np.count_nonzero(candidate) >= rows
                ),
                max_attempts=80,
            )
        except ValueError:
            motif = np.zeros((period, cols), dtype=int)
            for row in range(period):
                code = row + 1
                for col in range(cols):
                    if code & (1 << col):
                        motif[row, col] = color
            return np.vstack([motif[row % period] for row in range(rows)])

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        input_rows = grid.shape[0]
        period = input_rows
        for candidate in range(1, input_rows):
            if np.array_equal(grid[candidate:], grid[:-candidate]):
                period = candidate
                break

        output_rows = taskvars["output_rows"]
        completed = np.vstack([grid[row % period] for row in range(output_rows)])
        return np.where(
            completed == taskvars["input_color"],
            taskvars["output_color"],
            completed,
        )
