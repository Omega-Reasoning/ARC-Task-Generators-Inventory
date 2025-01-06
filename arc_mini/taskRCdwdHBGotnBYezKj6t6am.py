from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import retry
import numpy as np
import random


class TasktaskRCdwdHBGotnBYezKj6t6amGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "All input grids are of size nxn.",
            "Each input grid contains a square frame of size (n-2)x(n-2), which is one cell wide and contains empty (0) cells both inside and outside the frame.",
            "It is placed centrally in the grid, such that the first and the last, rows and columns are empty (0).",
            "The square frame is of {color('object_color1')} color or {color('object_color2')} color."
        ]

        reasoning_chain = [
            "To construct the output grid, copy the input grid and apply the following transformation.",
            "The left half of the frame retains its original color.",
            "The right half of the frame is transformed to a new color, based on the color of the input grid; {color('object_color1')} → {color('object_color3')} and {color('object_color2')} → {color('object_color4')}"
        ]

        super().__init__(observation_chain, reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        n = taskvars["n"]
        obj_col1 = taskvars["object_color1"]
        obj_col2 = taskvars["object_color2"]

        chosen_color = random.choice([obj_col1, obj_col2])
        grid = np.zeros((n, n), dtype=int)

        for c in range(1, n - 1):
            grid[1, c] = chosen_color
            grid[n - 2, c] = chosen_color
        for r in range(1, n - 1):
            grid[r, 1] = chosen_color
            grid[r, n - 2] = chosen_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output = grid.copy()
        n = taskvars["n"]
        col1 = taskvars["object_color1"]
        col2 = taskvars["object_color2"]
        col3 = taskvars["object_color3"]
        col4 = taskvars["object_color4"]

        mid_col = n // 2
        border_coords = []
        border_coords += [(1, c) for c in range(1, n - 1)]
        border_coords += [(n - 2, c) for c in range(1, n - 1)]
        border_coords += [(r, 1) for r in range(1, n - 1)]
        border_coords += [(r, n - 2) for r in range(1, n - 1)]

        for (r, c) in border_coords:
            if c >= mid_col:
                if output[r, c] == col1:
                    output[r, c] = col3
                elif output[r, c] == col2:
                    output[r, c] = col4

        return output

    def create_grids(self):
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        obj_col1, obj_col2, obj_col3, obj_col4 = all_colors[:4]

        taskvars = {
            "object_color1": obj_col1,
            "object_color2": obj_col2,
            "object_color3": obj_col3,
            "object_color4": obj_col4
        }

        nr_train = random.randint(3, 5)
        even_candidates = [x for x in range(6, 19) if x % 2 == 0]
        random.shuffle(even_candidates)
        train_ns = even_candidates[:nr_train]
        remainder = even_candidates[nr_train:]
        test_n = remainder[0]

        train_pairs = []
        for n_val in train_ns:
            local_taskvars = dict(taskvars)
            local_taskvars["n"] = n_val
            input_grid = self.create_input(local_taskvars, {})
            output_grid = self.transform_input(input_grid, local_taskvars)
            train_pairs.append({
                "input": input_grid,
                "output": output_grid
            })

        local_taskvars = dict(taskvars)
        local_taskvars["n"] = test_n
        test_input = self.create_input(local_taskvars, {})
        test_output = self.transform_input(test_input, local_taskvars)

        taskvars["n"] = test_n

        data = {
            "train": train_pairs,
            "test": [{
                "input": test_input,
                "output": test_output
            }]
        }

        return (taskvars, data)



