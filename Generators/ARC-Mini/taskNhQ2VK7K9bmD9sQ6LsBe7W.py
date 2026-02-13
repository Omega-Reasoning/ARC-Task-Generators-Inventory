from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry
import numpy as np
import random

class TaskNhQ2VK7K9bmD9sQ6LsBe7WGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain one {color('cell_color1')}, one {color('cell_color2')}, and empty (0) cells.",
            "Each colored cell is completely separated by empty (0) cells."
        ]
        # 2) The transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and expanding each colored cell vertically (up and down) and horizontally (left and right) until it meets another colored cell or the grid edge.",
            "The expansion begins with the {color('cell_color1')} cell, followed by {color('cell_color2')} cell."
        ]
        # 3) Call super().__init__ exactly once
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Randomly pick rows, cols between 5 and 30, pick two distinct colors from [1..9],
        then create 3-4 training examples and 1 test example.  All examples share
        the same task variables (rows, cols, cell_color1, cell_color2).
        """
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        
        cell_color1 = random.randint(1, 9)
        # Ensure cell_color2 != cell_color1
        while True:
            cell_color2 = random.randint(1, 9)
            if cell_color2 != cell_color1:
                break
        
        taskvars = {
            'rows': rows,
            'cols': cols,
            'cell_color1': cell_color1,
            'cell_color2': cell_color2
        }
        
        nr_train_examples = random.randint(3, 4)
        nr_test_examples = 1
        
        # Use the ARCTaskGenerator built-in helper which automatically calls
        # create_input() and transform_input() for the requested number of examples.
        train_test_data = self.create_grids_default(nr_train_examples, nr_test_examples, taskvars)
        
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an empty grid of size rows x cols with exactly 2 colored cells:
        - One cell of color 'cell_color1'
        - One cell of color 'cell_color2'
        Ensuring they are 8-way separated (i.e., no adjacency including diagonals).
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # We can simply keep retrying random positions until we find two that are 8-way separated
        # i.e. chebyshev distance >= 2
        def generate_2_positions():
            r1 = random.randint(0, rows - 1)
            c1 = random.randint(0, cols - 1)
            r2 = random.randint(0, rows - 1)
            c2 = random.randint(0, cols - 1)
            return (r1, c1), (r2, c2)
        
        def valid_8way_separation(positions):
            (r1, c1), (r2, c2) = positions
            if (r1, c1) == (r2, c2):
                return False
            # Chebyshev distance >= 2 ensures no 8-way adjacency
            if abs(r1 - r2) < 2 and abs(c1 - c2) < 2:
                return False
            return True
        
        (r1, c1), (r2, c2) = retry(generate_2_positions, valid_8way_separation)
        
        grid[r1, c1] = color1
        grid[r2, c2] = color2
        
        return grid

    def transform_input(self, grid, taskvars):
        """
        Transform the input grid by expanding each colored cell (in order):
        
        1) First expand the cell of color1 in all four directions (left/right/up/down)
           until encountering grid boundary or another non-zero cell.
        2) Then do the same expansion for color2.

        The expansions overwrite only empty (0) cells.
        """
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']
        
        output_grid = grid.copy()
        
        # Expand color1 first
        self._expand_color(output_grid, color1)
        # Then expand color2
        self._expand_color(output_grid, color2)
        
        return output_grid

    def _expand_color(self, grid, color):
        """
        For the single cell with 'color', expand horizontally (left and right) and 
        vertically (up and down) until hitting a non-empty cell or the grid boundary.
        """
        positions = np.argwhere(grid == color)
        if len(positions) == 0:
            return  # No cell of this color found
        
        # We assume there's exactly one cell of the color in the input
        r, c = positions[0]
        
        # Expand left
        cc = c - 1
        while cc >= 0 and grid[r, cc] == 0:
            grid[r, cc] = color
            cc -= 1
        
        # Expand right
        cc = c + 1
        while cc < grid.shape[1] and grid[r, cc] == 0:
            grid[r, cc] = color
            cc += 1
        
        # Expand up
        rr = r - 1
        while rr >= 0 and grid[rr, c] == 0:
            grid[rr, c] = color
            rr -= 1
        
        # Expand down
        rr = r + 1
        while rr < grid.shape[0] and grid[rr, c] == 0:
            grid[rr, c] = color
            rr += 1



