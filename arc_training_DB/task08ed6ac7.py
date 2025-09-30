from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, retry
from typing import Tuple
import numpy as np
import random

class ARCTask08ed6ac7Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size n x n.",
            "floor({vars['n']/2}) columns are filled with varying number cells per column which are stacked in a column-wise manner from bottom to top of color {vars['input_color']}.",
            "The heights of the colored cell stacks in the columns vary.",
            "Alternate columns are filled with cells of above color starting from the 2nd column(column indexing starts from 0) and ending at the last but one column."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First copy the input grid.",
            "The columns cells should be colored according to the stack height of the cells, column with least number of cells is filled with {vars['color_1']}, second least with {vars['color_2']}, third least with {vars['color_3']} and the fourth with {vars['color_4']}."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        n = taskvars['n']
        input_color = taskvars['input_color']

        grid = np.zeros((n, n), dtype=int)
        
        # Generate unique heights for each column
        available_heights = list(range(1, n + 1))
        num_columns = (n + 1) // 2  # Number of columns to fill (every other column)
        selected_heights = random.sample(available_heights, num_columns)
        
        # Alternate columns filling logic with unique heights
        for i, col in enumerate(range(1, n-1, 2)):
            stack_height = selected_heights[i]
            grid[n - stack_height:, col] = input_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        n = taskvars['n']
        color_1, color_2, color_3, color_4 = taskvars['color_1'], taskvars['color_2'], taskvars['color_3'], taskvars['color_4']

        output_grid = grid.copy()

        # Count non-zero cells in each column
        column_heights = []
        for col in range(n):
            height = np.sum(grid[:, col] != 0)
            if height > 0:  # Only include columns that have cells
                column_heights.append((col, height))
        
        # Sort by height (ascending)
        column_heights.sort(key=lambda x: x[1])

        # Assign colors based on sorted heights
        colors = [color_1, color_2, color_3, color_4]
        for i, (col, _) in enumerate(column_heights):
            mask = grid[:, col] != 0
            output_grid[mask, col] = colors[i]

        return output_grid

    def create_grids(self) -> Tuple[dict, TrainTestData]:
        n = 9
        colors = random.sample(range(1, 10), 5)

        taskvars = {
            'n': n,
            'input_color': colors[0],
            'color_1': colors[1],
            'color_2': colors[2],
            'color_3': colors[3],
            'color_4': colors[4]
        }

        train_test_data = self.create_grids_default(nr_train_examples=4, nr_test_examples=1, taskvars=taskvars)
        return taskvars, train_test_data


