from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from Framework.input_library import Contiguity, create_object, retry


class Task963e52fcGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "1. Each input has {vars['input_rows']} rows and a variable width, with empty rows surrounding one or more patterned rows.",
            "2. The nonempty rows contain a multicolored motif whose internal structure varies between examples.",
            "3. Input width, occupied-row count and location, colors, and motif periodicity vary per example.",
            "4. Each row has a shortest horizontal motif that controls its continuation.",
        ]
        transformation_reasoning_chain = [
            "1. Infer the shortest horizontally repeating motif of each input row independently.",
            "2. Continue every row motif cyclically until its width is multiplied by {vars['repeat_count']}.",
            "3. Stack the extended rows so output height stays {vars['input_rows']}.",
            "4. Preserve every motif color and empty cell without rotation or recoloring.",
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        taskvars = {"input_rows": 5, "repeat_count": 2}
        variant = random.randint(0, 1)
        layouts = [
            {"width": 6 + variant, "row_periods": {1: 2, 2: 3}},
            {"width": 7, "row_periods": {2: 3}},
            {"width": 8 + variant, "row_periods": {1: 3, 2: 4, 3: 2}},
            {"width": 10, "row_periods": {0: 4, 3: 3}},
        ]
        test_layout = {"width": 9 + variant, "row_periods": {1: 4, 2: 5}}

        def make_pair(gridvars):
            input_grid = self.create_input(taskvars, gridvars)
            return {"input": input_grid, "output": self.transform_input(input_grid, taskvars)}

        return taskvars, {
            "train": [make_pair(layout) for layout in layouts],
            "test": [make_pair(test_layout)],
        }

    def create_input(self, taskvars, gridvars):
        width = gridvars["width"]
        grid = np.zeros((taskvars["input_rows"], width), dtype=int)
        palette = list(
            gridvars.get(
                "palette",
                random.sample(range(1, 10), 3),
            )
        )
        routed_motifs = gridvars.get("motifs", {})

        def shortest_period(row):
            for candidate in range(1, len(row) + 1):
                if all(row[col] == row[col % candidate] for col in range(len(row))):
                    return candidate
            return len(row)

        for routed_row, routed_period in gridvars["row_periods"].items():
            row = int(routed_row)
            period = int(routed_period)
            def sample_motif():
                return create_object(
                    1,
                    period,
                    palette,
                    contiguity=Contiguity.NONE,
                    background=0,
                )[0]

            try:
                sampled_motif = retry(
                    sample_motif,
                    lambda candidate: (
                        np.count_nonzero(candidate) >= 2
                        and shortest_period(candidate) == period
                    ),
                    max_attempts=80,
                )
            except ValueError:
                sampled_motif = np.zeros(period, dtype=int)
                sampled_motif[0] = palette[0]
                sampled_motif[-1] = palette[1]
            motif = np.asarray(
                routed_motifs.get(str(row), sampled_motif),
                dtype=int,
            )
            repeated_row = np.asarray(
                [motif[col % period] for col in range(width)],
                dtype=int,
            )
            if (
                motif.shape != (period,)
                or np.count_nonzero(motif) < 2
                or shortest_period(repeated_row) != period
                or not set(int(value) for value in np.unique(motif)).issubset(
                    {0, *[int(color) for color in palette]}
                )
            ):
                raise ValueError("motif does not match its routed row period")
            grid[row, :] = repeated_row
        return grid

    def transform_input(self, grid, taskvars):
        output_width = grid.shape[1] * taskvars["repeat_count"]
        output = np.zeros((grid.shape[0], output_width), dtype=int)
        for row in range(grid.shape[0]):
            period = grid.shape[1]
            for candidate in range(1, grid.shape[1] + 1):
                if all(
                    grid[row, col] == grid[row, col % candidate]
                    for col in range(grid.shape[1])
                ):
                    period = candidate
                    break
            output[row, :] = [grid[row, col % period] for col in range(output_width)]
        return output
