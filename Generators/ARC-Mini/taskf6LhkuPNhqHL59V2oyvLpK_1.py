from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskf6LhkuPNhqHL59V2oyvLpK_1Task(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain a completely filled grid border of {color('object_color1')} color, with the interior arranged in a checkerboard pattern made of {color('object_color2')} and {color('object_color3')} cells.",
            "The checkerboard pattern alternates {color('object_color2')} and {color('object_color3')} cells across both rows and columns."
        ]
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and removing (setting to 0) any {color('object_color2')} and {color('object_color3')} cells that are 4-way connected to a {color('object_color1')} cell."
        ]
        # 3) Call super constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create a square grid with:
          - A border of object_color1
          - Interior in a checkerboard pattern of object_color2 and object_color3
        """
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]
        color3 = taskvars["object_color3"]
        n = gridvars["size"]  # The chosen size for this grid

        grid = np.zeros((n, n), dtype=int)

        # Fill the border with color1
        grid[0, :] = color1
        grid[-1, :] = color1
        grid[:, 0] = color1
        grid[:, -1] = color1

        # Fill interior in checkerboard pattern
        for r in range(1, n - 1):
            for c in range(1, n - 1):
                # Decide which color to use based on parity of (r + c)
                if (r + c) % 2 == 0:
                    grid[r, c] = color2
                else:
                    grid[r, c] = color3

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transformation:
          - Copy the input grid
          - Remove (set to 0) any color2 or color3 cells that are directly adjacent (4-way connected)
            to any color1 cell.
        """
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]
        color3 = taskvars["object_color3"]

        out_grid = grid.copy()
        nrows, ncols = out_grid.shape

        # List to collect positions that need to be set to 0
        to_remove = []

        # Check all cells for adjacency to color1
        directions = [(0,1), (1,0), (0,-1), (-1,0)]  # 4-way connectivity
        for r in range(nrows):
            for c in range(ncols):
                if out_grid[r, c] in (color2, color3):
                    # Check if any adjacent cell is color1
                    for dr, dc in directions:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < nrows and 0 <= cc < ncols:
                            if out_grid[rr, cc] == color1:
                                to_remove.append((r, c))
                                break  # No need to check further neighbors

        # Apply removals
        for r, c in to_remove:
            out_grid[r, c] = 0

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Randomly initialize object_color1, object_color2, object_color3,
        pick 3-6 training sizes plus 1 test size (all distinct).
        Build the corresponding input grids, transform them, 
        and return them along with the color task variables.
        """
        # 1) Random distinct colors
        color1, color2, color3 = random.sample(range(1, 10), 3)

        # 2) Number of training examples
        nr_train = random.randint(3, 6)  # 3 to 6 training grids
        nr_total = nr_train + 1  # +1 for test

        # 3) Distinct square sizes between 5 and 30
        possible_sizes = list(range(5, 31))
        chosen_sizes = random.sample(possible_sizes, nr_total)

        # Prepare train/test results
        train_data = []
        for i in range(nr_train):
            size = chosen_sizes[i]
            gridvars = {"size": size}
            inp = self.create_input(
                {"object_color1": color1, "object_color2": color2, "object_color3": color3},
                gridvars
            )
            out = self.transform_input(
                inp, 
                {"object_color1": color1, "object_color2": color2, "object_color3": color3}
            )
            train_data.append({"input": inp, "output": out})

        # 4) Create test data
        test_size = chosen_sizes[-1]
        test_gridvars = {"size": test_size}
        inp_test = self.create_input(
            {"object_color1": color1, "object_color2": color2, "object_color3": color3},
            test_gridvars
        )
        out_test = self.transform_input(
            inp_test,
            {"object_color1": color1, "object_color2": color2, "object_color3": color3}
        )

        # 5) Return the task variables plus the train/test data
        taskvars = {
            "object_color1": color1,
            "object_color2": color2,
            "object_color3": color3
        }
        train_test_data: TrainTestData = {
            "train": train_data,
            "test": [{"input": inp_test, "output": out_test}]
        }
        return taskvars, train_test_data



