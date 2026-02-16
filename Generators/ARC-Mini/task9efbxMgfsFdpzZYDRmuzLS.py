from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random


class Task9efbxMgfsFdpzZYDRmuzLSGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid has a completely filled border with {color('border_color1')} or {color('border_color2')} color and a completely filled interior with a different color."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created, by copying the input grid and changing the color of all interior cells, based on the border color.",
            "If the border is of {color('border_color1')} color, the interior cells are changed to {color('change_color1')}.",
            "If the border is of {color('border_color2')} color, the interior cells are changed to {color('change_color2')}."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Create 3 training examples and 2 test examples, each with distinct grid sizes.
        We also ensure that border_color1, border_color2, change_color1, and change_color2
        are distinct and between 1 and 9.
        """
        # 1) Choose 4 distinct color values from 1..9 for the four variables
        colors = random.sample(range(1, 10), 4)
        border_color1, border_color2, change_color1, change_color2 = colors

        # 2) We'll generate 5 distinct grid sizes (for 3 train + 2 test)
        #    each dimension (rows, cols) is between 5 and 30, and we want them all different.
        #    A simple approach: choose 5 distinct integers from [5..30] for square grids 
        #    or we can choose random pairs. We'll just pick square grids for simplicity,
        #    each with a different side length.
        sizes = random.sample(range(5, 30), 7)

        # Store our task variables in a dictionary
        taskvars = {
            "border_color1": border_color1,
            "border_color2": border_color2,
            "change_color1": change_color1,
            "change_color2": change_color2,
        }

        # 3) Create train/test data
        train_pairs = []
        for i in range(5):
            # create an input grid
            grid_size = sizes[i]
            input_grid = self.create_input(
                taskvars,
                gridvars={"n": grid_size}  # pass chosen size to create_input
            )
            # transform it
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        # create test pairs
        test_pairs = []
        for i in range(2):
            test_size = sizes[5 + i]
            test_input_grid = self.create_input(
                taskvars,
                gridvars={"n": test_size}
            )
            test_output_grid = self.transform_input(test_input_grid, taskvars)
            test_pairs.append(GridPair(input=test_input_grid, output=test_output_grid))

        train_test_data = TrainTestData(
            train=train_pairs,
            test=test_pairs
        )

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid of size gridvars["n"] x gridvars["n"].
        The border is either border_color1 or border_color2 (randomly chosen),
        and the interior is filled with a single color (different from the chosen border color
        and different from the associated change_color).
        """
        n = gridvars["n"]
        border_color1 = taskvars["border_color1"]
        border_color2 = taskvars["border_color2"]
        change_color1 = taskvars["change_color1"]
        change_color2 = taskvars["change_color2"]

        # Randomly pick which border color to use
        chosen_border_color, chosen_change_color = random.choice(
            [
                (border_color1, change_color1),
                (border_color2, change_color2)
            ]
        )

        # The interior color should be different from chosen_border_color 
        # and different from chosen_change_color, to avoid degenerate cases.
        # We'll pick any from [0..9], but not those two.
        possible_interior_colors = [c for c in range(1 ,10) 
                                    if c not in [chosen_border_color, chosen_change_color]]
        if not possible_interior_colors:
            # fallback if somehow the range is too small, but should not happen with 0..9
            possible_interior_colors = [0]
        interior_color = random.choice(possible_interior_colors)

        # Create the grid
        grid = np.full((n, n), interior_color, dtype=int)
        # Fill the border
        grid[0, :] = chosen_border_color
        grid[-1, :] = chosen_border_color
        grid[:, 0] = chosen_border_color
        grid[:, -1] = chosen_border_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        According to the transformation reasoning chain:
        - Copy the grid
        - If the border is border_color1, change all interior cells to change_color1
        - If the border is border_color2, change all interior cells to change_color2
        """
        border_color1 = taskvars["border_color1"]
        border_color2 = taskvars["border_color2"]
        change_color1 = taskvars["change_color1"]
        change_color2 = taskvars["change_color2"]

        output_grid = grid.copy()

        # Check what the border color in the top-left corner is
        # (all border cells have the same color by construction).
        border_color_in_grid = output_grid[0, 0]

        if border_color_in_grid == border_color1:
            new_interior_color = change_color1
        elif border_color_in_grid == border_color2:
            new_interior_color = change_color2
        else:
            # If something unexpected, just return the grid as-is 
            # (ideally should not happen with correct input grids).
            return output_grid

        # Change interior cells (from row 1..n-2 and col 1..n-2)
        # to the new interior color
        n = output_grid.shape[0]
        if n > 2:  # only if there's an interior
            output_grid[1:-1, 1:-1] = new_interior_color

        return output_grid