from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from input_library import Contiguity, create_object

class TaskFCUY2yh6ka8QS6WWHYZ7UnGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains a single column, that is completely filled, with cells of the same color.",
            "This filled column can only be {color('object_color1')}, {color('object_color2')} or {color('object_color3')}.",
            "The remaining cells in the grid are empty (0)."
        ]
        
        reasoning_chain = [
            "To construct the output grid, start with an empty (0) grid and completely fill the last row with a single color.",
            "The color choice is based on the color of the input grid : {color('object_color1')} → {color('object_color4')}, {color('object_color2')}→ {color('object_color5')}, {color('object_color3')}→{color('object_color6')}"
        ]
        
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        all_colors = random.sample(range(1, 10), 6)
        taskvars = {
            "object_color1": all_colors[0],
            "object_color2": all_colors[1],
            "object_color3": all_colors[2],
            "object_color4": all_colors[3],
            "object_color5": all_colors[4],
            "object_color6": all_colors[5],
        }

        nr_train_examples = random.randint(3, 5)
        base_colors = [
            taskvars["object_color1"],
            taskvars["object_color2"],
            taskvars["object_color3"]
        ]
        random.shuffle(base_colors)
        train_colors = base_colors[:3]
        while len(train_colors) < nr_train_examples:
            train_colors.append(random.choice(base_colors))

        train_grids = []
        used_cols = []

        for col_color in train_colors:
            gridvars = {"filled_column_color": col_color}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_grids.append({
                "input": input_grid,
                "output": output_grid
            })
            used_cols.append(self._detect_filled_column(input_grid))

        test_color = random.choice(base_colors)
        test_gridvars = {"filled_column_color": test_color, "avoid_columns": used_cols}
        test_input_grid = self.create_input(taskvars, test_gridvars)
        test_output_grid = self.transform_input(test_input_grid, taskvars)
        
        test_grids = [{
            "input": test_input_grid,
            "output": test_output_grid
        }]

        train_test_data = {
            "train": train_grids,
            "test": test_grids
        }
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        filled_color = gridvars["filled_column_color"]
        rows = random.randint(6, 20)
        cols = random.randint(6, 20)
        
        avoid_columns = gridvars.get("avoid_columns", [])
        valid_cols = [c for c in range(cols) if c not in avoid_columns]
        if not valid_cols:
            for _ in range(50):
                rows = random.randint(6, 20)
                cols = random.randint(6, 20)
                valid_cols = [c for c in range(cols) if c not in avoid_columns]
                if valid_cols:
                    break
            if not valid_cols:
                valid_cols = list(range(cols))

        filled_col = random.choice(valid_cols)
        
        grid = np.zeros((rows, cols), dtype=int)
        grid[:, filled_col] = filled_color
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        rows, cols = grid.shape
        output_grid = np.zeros((rows, cols), dtype=int)

        nonzero_vals = grid[grid != 0]
        if len(nonzero_vals) == 0:
            return output_grid
        
        input_color = nonzero_vals[0]
        mapping = {
            taskvars["object_color1"]: taskvars["object_color4"],
            taskvars["object_color2"]: taskvars["object_color5"],
            taskvars["object_color3"]: taskvars["object_color6"]
        }
        output_color = mapping.get(input_color, 0)

        output_grid[rows - 1, :] = output_color
        return output_grid

    def _detect_filled_column(self, grid: np.ndarray) -> int:
        rows, cols = grid.shape
        for c in range(cols):
            col_slice = grid[:, c]
            if np.all(col_slice != 0):
                return c
        return -1