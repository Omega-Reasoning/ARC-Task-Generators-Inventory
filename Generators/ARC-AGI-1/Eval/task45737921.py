from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random


class Task45737921Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of different sizes.",
            "Each input grid contains many sub-grids of size {vars['rows']} × {vars['columns']}.",
            "Each sub-grid contains a small pattern using one color, and the empty space inside the sub-grid is filled with another color.",
            "Different sub-grids use different colors."
        ]

        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "Within each sub-grid, the pattern color and background color are swapped."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # ---------------------------------------------------------
    # INPUT CREATION
    # ---------------------------------------------------------

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        sub_rows = taskvars['rows']
        sub_cols = taskvars['columns']
        grid_size = gridvars['grid_size']

        grid = np.zeros((grid_size, grid_size), dtype=int)

        num_sub_grids = random.randint(2, 5)

        for _ in range(num_sub_grids):
            if grid_size <= sub_rows or grid_size <= sub_cols:
                continue

            start_row = random.randint(0, grid_size - sub_rows)
            start_col = random.randint(0, grid_size - sub_cols)

            pattern_color = random.randint(1, 9)
            background_color = random.randint(1, 9)
            while background_color == pattern_color:
                background_color = random.randint(1, 9)

            sub_region = np.full((sub_rows, sub_cols), background_color)

            # simple random binary pattern
            pattern_mask = np.random.randint(0, 2, size=(sub_rows, sub_cols))
            sub_region[pattern_mask == 1] = pattern_color

            grid[start_row:start_row+sub_rows,
                 start_col:start_col+sub_cols] = sub_region

        return grid

    # ---------------------------------------------------------
    # TRANSFORMATION (Dummy-safe)
    # ---------------------------------------------------------

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        rows, cols = grid.shape
        sub_rows = taskvars['rows']
        sub_cols = taskvars['columns']

        output_grid = grid.copy()
        visited = np.zeros_like(grid, dtype=bool)

        for r in range(rows - sub_rows + 1):
            for c in range(cols - sub_cols + 1):

                if np.any(visited[r:r+sub_rows, c:c+sub_cols]):
                    continue

                sub_region = grid[r:r+sub_rows, c:c+sub_cols]

                if np.all(sub_region == 0):
                    continue

                unique_colors = np.unique(sub_region)
                non_zero_colors = unique_colors[unique_colors != 0]

                if len(non_zero_colors) != 2:
                    continue

                color1, color2 = non_zero_colors

                region = output_grid[r:r+sub_rows, c:c+sub_cols]

                mask1 = region == color1
                mask2 = region == color2

                region[mask1] = color2
                region[mask2] = color1

                output_grid[r:r+sub_rows, c:c+sub_cols] = region
                visited[r:r+sub_rows, c:c+sub_cols] = True

        return output_grid

    # ---------------------------------------------------------
    # GRID CREATION
    # ---------------------------------------------------------

    def create_grids(self) -> tuple[dict, TrainTestData]:

        sub_size = random.choice([2, 3])

        taskvars = {
            'rows': sub_size,
            'columns': sub_size,
        }

        num_train = random.randint(3, 5)

        min_size = max(8, sub_size * 3)
        max_size = 20

        sizes = [random.randint(min_size, max_size)
                 for _ in range(num_train + 1)]

        train_examples = []

        for i in range(num_train):
            gridvars = {'grid_size': sizes[i]}

            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)

            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })

        test_gridvars = {'grid_size': sizes[-1]}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)

        test_examples = [{
            'input': test_input,
            'output': test_output
        }]

        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }