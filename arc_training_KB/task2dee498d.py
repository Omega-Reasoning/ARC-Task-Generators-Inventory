from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task2dee498dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes. A single square block (the base pattern) is repeated three times either horizontally or vertically to form the full input grid; the chosen orientation is consistent across all examples in the task.",
            "They contain multi-colored (1-9) and sometimes empty (0) cells.",
            "Each example uses **three different colors**, which vary across examples, except for **{color('cell_color')}**, which remains fixed across all examples.",
            "To construct the input grid, create a random square pattern (the base block) and repeat it three times in the chosen orientation."
        ]

        transformation_reasoning_chain = [
            "The output grids are constructed by copying the first repeated block from the input grid (i.e. the first one-third of the grid along the repetition axis) and using that as the output.",
            "When the input repeats horizontally this is the first one-third of the columns; when it repeats vertically this is the first one-third of the rows."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        cell_color = random.randint(1, 9)
        orientation = random.choice(["horizontal", "vertical"])  # consistent for this task instance
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1
        train_data = []
        test_data = []

        for i in range(nr_train_examples):
            input_grid = self._create_repeated_pattern(cell_color, ensure_zero=(i == 0), orientation=orientation)
            output_grid = self.transform_input(input_grid, {"cell_color": cell_color, "orientation": orientation})
            train_data.append({"input": input_grid, "output": output_grid})

        input_grid = self._create_repeated_pattern(cell_color, ensure_zero=False, orientation=orientation)
        output_grid = self.transform_input(input_grid, {"cell_color": cell_color, "orientation": orientation})
        test_data.append({"input": input_grid, "output": output_grid})
        
        taskvars = {"cell_color": cell_color, "orientation": orientation}
        train_test_data = {"train": train_data, "test": test_data}
        
        return taskvars, train_test_data

    def _create_repeated_pattern(self, cell_color: int, ensure_zero: bool, orientation: str = "horizontal") -> np.ndarray:
        N = random.randint(5, 10)
        possible_colors = [c for c in range(1, 10) if c != cell_color]
        color_a, color_b = random.sample(possible_colors, 2)
        used_colors = [cell_color, color_a, color_b]

        def generate_pattern():
            pattern = np.random.choice(used_colors + ([0] if ensure_zero else []),
                                       size=(N, N),
                                       p=[0.3, 0.3, 0.3, 0.1] if ensure_zero else [1/3]*3)
            return pattern

        while True:
            pattern = generate_pattern()
            color_set = set(pattern.flatten()) - {0}
            if color_set == set(used_colors):
                if not ensure_zero or (ensure_zero and 0 in pattern):
                    break

        if orientation == "horizontal":
            return np.hstack([pattern, pattern, pattern])
        else:
            return np.vstack([pattern, pattern, pattern])

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        return self._create_repeated_pattern(taskvars["cell_color"], ensure_zero=False, orientation=taskvars.get("orientation", "horizontal"))

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        orientation = taskvars.get("orientation", "horizontal")
        if orientation == "horizontal":
            # copy first one-third of the columns
            return grid[:, : grid.shape[1] // 3]
        else:
            # copy first one-third of the rows
            return grid[: grid.shape[0] // 3, :]

