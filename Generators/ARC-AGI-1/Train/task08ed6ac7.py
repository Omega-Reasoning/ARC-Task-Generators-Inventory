from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import create_object, retry
from typing import Tuple
import numpy as np
import random

class Task08ed6ac7Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "{(vars['grid_size'] - 1)//2} columns are filled with varying number cells per column which are stacked in a column-wise manner from bottom to top of color {vars['input_color']}.",
            "The heights of the colored cell stacks in the columns vary.",
            "Alternate columns are filled with cells of above color starting from the 2nd column(column indexing starts from 0) and ending at the last but one column."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First copy the input grid.",
            "The columns cells should be colored according to the stack height of the cells: smallest -> {color('color_1')}, second -> {color('color_2')}, third -> {color('color_3')}, fourth -> {color('color_4')}, fifth -> {color('color_5')}, sixth -> {color('color_6')}, seventh -> {color('color_7')}."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        grid_size = taskvars['grid_size']
        input_color = taskvars['input_color']

        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        available_heights = list(range(1, grid_size + 1))
        num_columns = (grid_size - 1) // 2
        selected_heights = random.sample(available_heights, num_columns)
        
        for i, col in enumerate(range(1, grid_size-1, 2)):
            stack_height = selected_heights[i]
            grid[grid_size - stack_height:, col] = input_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        grid_size = taskvars['grid_size']

        colors = [
            taskvars['color_1'],
            taskvars['color_2'],
            taskvars['color_3'],
            taskvars['color_4'],
            taskvars['color_5'],
            taskvars['color_6'],
            taskvars['color_7'],
        ]

        output_grid = grid.copy()

        column_heights = []
        for col in range(grid_size):
            height = np.sum(grid[:, col] != 0)
            if height > 0:
                column_heights.append((col, height))
        
        column_heights.sort(key=lambda x: x[1])

        for i, (col, _) in enumerate(column_heights[:len(colors)]):
            mask = grid[:, col] != 0
            output_grid[mask, col] = colors[i]

        return output_grid

    def create_grids(self) -> Tuple[dict, TrainTestData]:
        grid_size = random.choice(list(range(5, 16, 2)))
        colors = random.sample(range(1, 10), 8)

        taskvars = {
            'grid_size': grid_size,
            'n': grid_size,
            'input_color': colors[0],
            'color_1': colors[1],
            'color_2': colors[2],
            'color_3': colors[3],
            'color_4': colors[4],
            'color_5': colors[5],
            'color_6': colors[6],
            'color_7': colors[7]
        }

        train_test_data = self.create_grids_default(nr_train_examples=4, nr_test_examples=1, taskvars=taskvars)
        return taskvars, train_test_data