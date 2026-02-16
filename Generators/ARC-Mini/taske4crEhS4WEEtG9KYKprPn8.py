# generator_example.py

import random
import numpy as np

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
# You can import from Framework.transformation_library if needed; for this example, we do straightforward array operations
# from Framework.transformation_library import ...
# We do not need input_library here beyond basic random generation, so no import required

class Taske4crEhS4WEEtG9KYKprPn8Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains four same-colored cells located in the four corners of the grid.",
            "These cells can only be {color('cell_color1')} or {color('cell_color2')}.",
            "All other cells are empty (0)."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling all the empty (0) cells, based on the color of the corner cells.",
            "If the corner cells are of {color('cell_color1')} color, fill all the empty (0) cells with {color('fill_color1')} color.",
            "If the corner cells are of {color('cell_color2')} color, fill all the empty (0) cells with {color('fill_color2')} color."
        ]
        
        # 3) Call parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid of random size (5..30 rows/cols) where
        the four corners have the same color, which is either cell_color1 or cell_color2.
        All other cells are 0.
        """
        corner_color = gridvars["corner_color"]
        used_sizes = gridvars["used_sizes"]

        # Randomly pick a grid size that has not been used yet
        while True:
            rows = random.randint(5, 30)
            cols = random.randint(5, 30)
            if (rows, cols) not in used_sizes:
                used_sizes.add((rows, cols))
                break

        grid = np.zeros((rows, cols), dtype=int)
        # Fill corners
        grid[0, 0] = corner_color
        grid[0, cols - 1] = corner_color
        grid[rows - 1, 0] = corner_color
        grid[rows - 1, cols - 1] = corner_color

        return grid

    def transform_input(self,
                        grid: np.ndarray,
                        taskvars: dict) -> np.ndarray:
        """
        Transform the input grid by checking the corner color
        and filling all empty cells with the appropriate fill color.
        """
        cell_color1 = taskvars["cell_color1"]
        cell_color2 = taskvars["cell_color2"]
        fill_color1 = taskvars["fill_color1"]
        fill_color2 = taskvars["fill_color2"]

        output_grid = grid.copy()
        # All corners have the same color, so just read the top-left corner
        corner_color = output_grid[0, 0]

        if corner_color == cell_color1:
            # Fill all empty cells (0) with fill_color1
            output_grid[output_grid == 0] = fill_color1
        else:
            # Fill all empty cells (0) with fill_color2
            output_grid[output_grid == 0] = fill_color2

        return output_grid

    def create_grids(self):
        """
        Randomly initializes task variables (colors) and constructs
        a set of train/test grids ensuring:
        1) Distinct sizes for each grid.
        2) At least one training example uses cell_color1 corners and at least
           one training example uses cell_color2 corners.
        3) Colors are all distinct in {1..9}.
        """
        # Randomly pick four distinct colors from 1..9
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        cell_color1 = available_colors[0]
        cell_color2 = available_colors[1]
        fill_color1 = available_colors[2]
        fill_color2 = available_colors[3]

        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2,
            "fill_color1": fill_color1,
            "fill_color2": fill_color2
        }

        # Number of train examples between 3 and 6
        nr_train = random.randint(3, 6)

        train_data = []
        used_sizes = set()

        # Force at least one example with cell_color1 corners
        gridvars = {"corner_color": cell_color1, "used_sizes": used_sizes}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_data.append({"input": input_grid, "output": output_grid})

        # Force at least one example with cell_color2 corners
        gridvars = {"corner_color": cell_color2, "used_sizes": used_sizes}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        train_data.append({"input": input_grid, "output": output_grid})

        # Create remaining train grids (if any) with random corner color
        for _ in range(nr_train - 2):
            random_corner = random.choice([cell_color1, cell_color2])
            gridvars = {"corner_color": random_corner, "used_sizes": used_sizes}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({"input": input_grid, "output": output_grid})

        # Create exactly one test grid (can be either corner color)
        test_corner = random.choice([cell_color1, cell_color2])
        gridvars = {"corner_color": test_corner, "used_sizes": used_sizes}
        test_input_grid = self.create_input(taskvars, gridvars)
        test_output_grid = self.transform_input(test_input_grid, taskvars)

        test_data = [
            {
                "input": test_input_grid,
                "output": test_output_grid
            }
        ]

        train_test_data = {
            "train": train_data,
            "test": test_data
        }

        # Return the variables and all train/test data
        return taskvars, train_test_data



