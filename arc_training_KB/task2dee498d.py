from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task2dee498dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes, where the number of columns is always a multiple of three, and the number of rows is equal to one-third of the number of columns.",
            "They contain multi-colored (1-9) and sometimes empty (0) cells.",
            "Each example uses **three different colors**, which vary across examples, except for **{color('cell_color')}**, which remains fixed across all examples.",
            "To construct the input grid, create a random pattern using the multi-colored cells in the first one-third of the columns, then repeat this pattern three times across the entire grid."
        ]

        transformation_reasoning_chain = [
            "The output grids have the same number of rows, but the number of columns is one-third of the input grid.",
            "They are constructed by copying the first one-third of the columns from the input grid and pasting them into the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        cell_color = random.randint(1, 9)
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1
        train_data = []
        test_data = []

        for i in range(nr_train_examples):
            input_grid = self._create_repeated_pattern(cell_color, ensure_zero=(i == 0))
            output_grid = self.transform_input(input_grid, {"cell_color": cell_color})
            train_data.append({"input": input_grid, "output": output_grid})

        input_grid = self._create_repeated_pattern(cell_color, ensure_zero=False)
        output_grid = self.transform_input(input_grid, {"cell_color": cell_color})
        test_data.append({"input": input_grid, "output": output_grid})
        
        taskvars = {"cell_color": cell_color}
        train_test_data = {"train": train_data, "test": test_data}
        
        return taskvars, train_test_data

    def _create_repeated_pattern(self, cell_color: int, ensure_zero: bool) -> np.ndarray:
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

        return np.hstack([pattern, pattern, pattern])

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        return self._create_repeated_pattern(taskvars["cell_color"], ensure_zero=False)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        return grid[:, :grid.shape[1] // 3]

