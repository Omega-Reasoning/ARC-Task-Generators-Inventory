from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, retry
from Framework.transformation_library import find_connected_objects
import numpy as np
import random

class Task1b2d62fbGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Two rectangular subgrids each contain {color('color_main')} and empty cells.",
            "A vertical bar of {color('color_vertical_separator')} cells separates these subgrids",
            "The size of each subgrid is {vars['subgrid_rows']} rows by {vars['subgrid_cols']} columns."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same dimension as each of the subgrids.",
            "Cells that are not {color('color_main')} in both rectangular subgrids (logical NOR) are painted {color('color_output')}.",
            "All other cells remain empty(0) in the output matrix."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        subgrid_rows = taskvars['subgrid_rows']
        subgrid_cols = taskvars['subgrid_cols']
        color_main = taskvars['color_main']
        color_vertical_separator = taskvars['color_vertical_separator']

        height = subgrid_rows
        width = 2 * subgrid_cols + 1
        grid = np.zeros((height, width), dtype=int)

        for subgrid_start_col in [0, subgrid_cols + 1]:
            subgrid = create_object(
                height=subgrid_rows,
                width=subgrid_cols,
                color_palette=color_main
            )
            grid[:, subgrid_start_col:subgrid_start_col + subgrid_cols] = subgrid

        grid[:, subgrid_cols] = color_vertical_separator
        return grid

    def transform_input(self, grid, taskvars):
        subgrid_cols = taskvars['subgrid_cols']
        color_main = taskvars['color_main']
        color_output = taskvars['color_output']

        subgrid_1 = grid[:, :subgrid_cols]
        subgrid_2 = grid[:, subgrid_cols + 1:]

        output = np.zeros_like(subgrid_1)
        output[(subgrid_1 != color_main) & (subgrid_2 != color_main)] = color_output
        return output

    def create_grids(self):
        subgrid_rows = random.randint(5, 15)
        subgrid_cols = random.randint(5, 15)

        colors = random.sample(range(1, 10), 3)
        color_main, color_vertical_separator, color_output = colors

        taskvars = {
            'subgrid_rows': subgrid_rows,
            'subgrid_cols': subgrid_cols,
            'color_main': color_main,
            'color_vertical_separator': color_vertical_separator,
            'color_output': color_output
        }

        train_examples = random.randint(3, 4)
        train_test_data = self.create_grids_default(train_examples, 1, taskvars)

        return taskvars, train_test_data


