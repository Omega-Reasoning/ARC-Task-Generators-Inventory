from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Task8xwytGZPr5TeZw3QpX25mPGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input matrix contains a 2x2 block which is made of {color('cell_color1')}, {color('cell_color2')}, {color('cell_color3')} and {color('object_color')} cells."
        ]
        reasoning_chain = [
            "To construct the output matrix, create four 2x2 blocks, each block made of one color and positioned in one of the four corners of the matrix.",
            "Each block is colored to match the corresponding cell in the 2x2 block from the input matrix.",
            "All other cells of the output matrix are set to empty (0)."
        ]
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        color_choices = random.sample(range(1, 10), 4)
        taskvars = {
            "cell_color1": color_choices[0],
            "cell_color2": color_choices[1],
            "cell_color3": color_choices[2],
            "object_color": color_choices[3]
        }
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        grid = np.zeros((height, width), dtype=int)
        r = random.randint(0, height - 2)
        c = random.randint(0, width - 2)
        grid[r, c] = taskvars["cell_color1"]
        grid[r, c + 1] = taskvars["cell_color2"]
        grid[r + 1, c] = taskvars["cell_color3"]
        grid[r + 1, c + 1] = taskvars["object_color"]
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        height, width = grid.shape
        output_grid = np.zeros_like(grid)
        cc1 = taskvars["cell_color1"]
        cc2 = taskvars["cell_color2"]
        cc3 = taskvars["cell_color3"]
        oc = taskvars["object_color"]
        block_found = False
        for r in range(height - 1):
            for c in range(width - 1):
                if (grid[r, c] == cc1 and
                    grid[r, c+1] == cc2 and
                    grid[r+1, c] == cc3 and
                    grid[r+1, c+1] == oc):
                    top_left_color = cc1
                    top_right_color = cc2
                    bottom_left_color = cc3
                    bottom_right_color = oc
                    block_found = True
                    break
            if block_found:
                break
        output_grid[0, 0] = top_left_color
        output_grid[0, 1] = top_left_color
        output_grid[1, 0] = top_left_color
        output_grid[1, 1] = top_left_color
        output_grid[0, width-2] = top_right_color
        output_grid[0, width-1] = top_right_color
        output_grid[1, width-2] = top_right_color
        output_grid[1, width-1] = top_right_color
        output_grid[height-2, 0] = bottom_left_color
        output_grid[height-2, 1] = bottom_left_color
        output_grid[height-1, 0] = bottom_left_color
        output_grid[height-1, 1] = bottom_left_color
        output_grid[height-2, width-2] = bottom_right_color
        output_grid[height-2, width-1] = bottom_right_color
        output_grid[height-1, width-2] = bottom_right_color
        output_grid[height-1, width-1] = bottom_right_color
        return output_grid

