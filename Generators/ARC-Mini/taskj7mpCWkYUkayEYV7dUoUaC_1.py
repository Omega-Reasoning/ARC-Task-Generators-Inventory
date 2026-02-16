from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optionally use these if you need them within create_input() (not transform_input()):
from Framework.input_library import retry
from Framework.input_library import Contiguity
from Framework.transformation_library import find_connected_objects

class Taskj7mpCWkYUkayEYV7dUoUaC_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input Reasoning Chain
        input_reasoning_chain = [
            "All input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid contains exactly three colored cells, positioned so that they are separated by at least two rows or two columns from each other.",
            "The three colored cells are {color('cell_color1')}, {color('cell_color2')}, and {color('cell_color3')}."
        ]

        # 2) Transformation Reasoning Chain
        transformation_reasoning_chain = [
    "The output grid is created by copying the input grid, and for each colored cell, all adjacent (up, down, left, right) empty (0) cells are filled.",
    "The fill color depends on the color of the original cell: If the original cell is {color('cell_color1')}, the fill color is {color('fill_color1')}; if {color('cell_color2')}, the fill color is {color('fill_color2')}; if {color('cell_color3')}, the fill color is {color('fill_color3')}."
]


        # 3) Call super().__init__ with these arguments
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates the random task variables and then calls create_grids_default()
        to produce the train/test pairs.
        """

        # 1) Randomly choose number of train grids (3 or 4) and 1 test grid.
        nr_train_examples = random.choice([3, 4])
        nr_test_examples = 1

        # 2) Randomly choose a grid size (rows and cols between 7 and 30).
        rows = random.randint(7, 30)
        cols = random.randint(7, 30)

        # 3) Pick 6 distinct colors from 1..9 for the cell colors and fill colors.
        #    We do not allow duplication among cell_color1, cell_color2, cell_color3, fill_color1, fill_color2, fill_color3.
        chosen_colors = random.sample(range(1, 10), 6)
        cell_color1, cell_color2, cell_color3 = chosen_colors[0:3]
        fill_color1, fill_color2, fill_color3 = chosen_colors[3:6]

        # 4) Store in taskvars
        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2,
            'cell_color3': cell_color3,
            'fill_color1': fill_color1,
            'fill_color2': fill_color2,
            'fill_color3': fill_color3
        }

        # 5) Create train/test data using the default pattern of simply calling create_input() and transform_input().
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
        1) Grid of size rows x cols.
        2) Exactly three colored cells (cell_color1, cell_color2, cell_color3),
           each placed at least 3 rows or 3 columns away from each other,
           and none on the border (i.e. not in row=0, row=rows-1, col=0, or col=cols-1).
        """

        rows = taskvars['rows']
        cols = taskvars['cols']
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']
        color3 = taskvars['cell_color3']

        grid = np.zeros((rows, cols), dtype=int)

        # Valid positions (not on the edges)
        valid_positions = [
            (r, c)
            for r in range(1, rows-1)
            for c in range(1, cols-1)
        ]

        # We must place three cells so that each pair is separated
        # by at least 3 rows OR 3 columns.
        # We'll try random positions until we find a valid arrangement.
        def is_valid_arrangement(positions):
            # positions is a list of (r1,c1), (r2,c2), (r3,c3)
            # check pairwise separation
            for i in range(3):
                for j in range(i+1, 3):
                    rdiff = abs(positions[i][0] - positions[j][0])
                    cdiff = abs(positions[i][1] - positions[j][1])
                    # Must be at least 3 in either rows or columns
                    if rdiff < 3 and cdiff < 3:
                        return False
            return True

        def attempt_placement():
            # pick 3 distinct positions from valid_positions
            chosen = random.sample(valid_positions, 3)
            if is_valid_arrangement(chosen):
                return chosen
            return None

        chosen_positions = retry(
            generator=lambda: attempt_placement(),
            predicate=lambda x: x is not None,
            max_attempts=200
        )

        # Assign colors
        grid[chosen_positions[0][0], chosen_positions[0][1]] = color1
        grid[chosen_positions[1][0], chosen_positions[1][1]] = color2
        grid[chosen_positions[2][0], chosen_positions[2][1]] = color3

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        1) Copy the input grid.
        2) For each cell that is one of the three cell_colors, fill each of its
           (up, down, left, right) neighbors (if in-bounds and empty) with the
           corresponding fill color.
        """

        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']
        cell_color3 = taskvars['cell_color3']
        fill_color1 = taskvars['fill_color1']
        fill_color2 = taskvars['fill_color2']
        fill_color3 = taskvars['fill_color3']

        rows, cols = grid.shape
        output_grid = np.copy(grid)

        # We'll scan through the original grid so we don't "chain fill" newly-filled cells.
        for r in range(rows):
            for c in range(cols):
                original_color = grid[r, c]
                if original_color == cell_color1:
                    fc = fill_color1
                elif original_color == cell_color2:
                    fc = fill_color2
                elif original_color == cell_color3:
                    fc = fill_color3
                else:
                    continue  # not one of the special colors, skip

                # Now fill the 4 neighbors in output_grid if they are 0
                for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if output_grid[rr, cc] == 0:
                            output_grid[rr, cc] = fc

        return output_grid



