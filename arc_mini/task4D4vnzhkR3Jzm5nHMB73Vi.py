from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry
import numpy as np
import random

class Task4D4vnzhkR3Jzm5nHMB73ViGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "All input grids are squares.",
            "Let n be the height and width of an input grid, where n is an odd number, between 5 and 20.",
            "Each grid has an outer border filled with either {color('border_color1')} or {color('border_color2')} color and an empty (0) interior."
        ]

        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling the empty (0) interior cells with a checkerboard pattern that alternates two colors consistently across rows and columns.",
            "The two colors for the pattern are determined by the color of the border cells.",
            "If the border is {color('border_color1')}, the checkerboard pattern uses {color('pattern_color1')} and {color('pattern_color2')} colors.",
            "If the border is {color('border_color2')}, the checkerboard pattern uses {color('pattern_color3')} and {color('pattern_color4')} colors.",
            "The checkerboard pattern starts with {color('pattern_color1')} or {color('pattern_color3')} color."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        n = gridvars["n"]
        grid = np.zeros((n, n), dtype=int)
        border_color = gridvars["border_color"]

        grid[0, :] = border_color
        grid[n-1, :] = border_color
        grid[:, 0] = border_color
        grid[:, n-1] = border_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        border_color1 = taskvars["border_color1"]
        border_color2 = taskvars["border_color2"]
        pattern_color1 = taskvars["pattern_color1"]
        pattern_color2 = taskvars["pattern_color2"]
        pattern_color3 = taskvars["pattern_color3"]
        pattern_color4 = taskvars["pattern_color4"]

        out_grid = np.copy(grid)
        actual_border_color = out_grid[0, 0]

        if actual_border_color == border_color1:
            color_a, color_b = pattern_color1, pattern_color2
        else:
            color_a, color_b = pattern_color3, pattern_color4

        n = out_grid.shape[0]
        for r in range(1, n-1):
            for c in range(1, n-1):
                if (r + c) % 2 == 0:
                    out_grid[r, c] = color_a
                else:
                    out_grid[r, c] = color_b

        return out_grid

    def create_grids(self) -> tuple:
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        border_color1 = all_colors[0]
        border_color2 = all_colors[1]
        pattern_color1 = all_colors[2]
        pattern_color2 = all_colors[3]
        pattern_color3 = all_colors[4]
        pattern_color4 = all_colors[5]

        taskvars = {
            "border_color1": border_color1,
            "border_color2": border_color2,
            "pattern_color1": pattern_color1,
            "pattern_color2": pattern_color2,
            "pattern_color3": pattern_color3,
            "pattern_color4": pattern_color4
        }

        possible_ns = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        random.shuffle(possible_ns)
        chosen_ns = possible_ns[:6]

        train = []
        gridvars_1 = {"n": chosen_ns[0], "border_color": border_color1}
        input_grid_1 = self.create_input(taskvars, gridvars_1)
        output_grid_1 = self.transform_input(input_grid_1, taskvars)
        train.append(GridPair(input=input_grid_1, output=output_grid_1))

        gridvars_2 = {"n": chosen_ns[1], "border_color": border_color2}
        input_grid_2 = self.create_input(taskvars, gridvars_2)
        output_grid_2 = self.transform_input(input_grid_2, taskvars)
        train.append(GridPair(input=input_grid_2, output=output_grid_2))

        gridvars_3 = {"n": chosen_ns[2], "border_color": random.choice([border_color1, border_color2])}
        input_grid_3 = self.create_input(taskvars, gridvars_3)
        output_grid_3 = self.transform_input(input_grid_3, taskvars)
        train.append(GridPair(input=input_grid_3, output=output_grid_3))

        gridvars_4 = {"n": chosen_ns[3], "border_color": random.choice([border_color1, border_color2])}
        input_grid_4 = self.create_input(taskvars, gridvars_4)
        output_grid_4 = self.transform_input(input_grid_4, taskvars)
        train.append(GridPair(input=input_grid_4, output=output_grid_4))

        test = []

        gridvars_test1 = {"n": chosen_ns[4], "border_color": border_color1}
        input_test_1 = self.create_input(taskvars, gridvars_test1)
        output_test_1 = self.transform_input(input_test_1, taskvars)
        test.append(GridPair(input=input_test_1, output=output_test_1))

        gridvars_test2 = {"n": chosen_ns[5], "border_color": border_color2}
        input_test_2 = self.create_input(taskvars, gridvars_test2)
        output_test_2 = self.transform_input(input_test_2, taskvars)
        test.append(GridPair(input=input_test_2, output=output_test_2))

        return taskvars, TrainTestData(train=train, test=test)

