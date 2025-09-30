from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, random_cell_coloring
import numpy as np
import random

class ARCTask1c786137Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}",
            "Each cell in the input grid can either have a color between 1 and 5 or be empty(0).",
            "The input grid has a subgrid of size {vars['subrows']} X {vars['subcols']}, where only the perimeter of this subgrid is of color {color('subgrid_color')}"
        ]

        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "First identify the subgrid whose perimeter has color {color('subgrid_color')}.",
            "The subgrid is the output grid."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        subrows, subcols = taskvars['subrows'], taskvars['subcols']
        subgrid_color = taskvars['subgrid_color']

        grid = np.zeros((rows, cols), dtype=int)
        cell_color = [1, 2, 3, 4, 5]
        # Randomly color cells
        grid = random_cell_coloring(grid, cell_color, density=0.7)

        # Create subgrid with colored perimeter
        start_row = random.randint(0, rows - subrows)
        start_col = random.randint(0, cols - subcols)
        
        # Draw subgrid perimeter
        grid[start_row, start_col:start_col + subcols] = subgrid_color
        grid[start_row + subrows - 1, start_col:start_col + subcols] = subgrid_color
        grid[start_row:start_row + subrows, start_col] = subgrid_color
        grid[start_row:start_row + subrows, start_col + subcols - 1] = subgrid_color

        return grid

    def transform_input(self, grid, taskvars):
        subrows, subcols = taskvars['subrows'], taskvars['subcols']
        subgrid_color = taskvars['subgrid_color']
        
        # Find the subgrid by looking for its perimeter
        rows, cols = grid.shape
        for i in range(rows - subrows + 1):
            for j in range(cols - subcols + 1):
                # Check if this position has a valid subgrid perimeter
                if (all(grid[i, j:j+subcols] == subgrid_color) and  # top edge
                    all(grid[i+subrows-1, j:j+subcols] == subgrid_color) and  # bottom edge
                    all(grid[i:i+subrows, j] == subgrid_color) and  # left edge
                    all(grid[i:i+subrows, j+subcols-1] == subgrid_color)):  # right edge
                    # Found the subgrid, extract it without the perimeter
                    return grid[i+1:i+subrows-1, j+1:j+subcols-1]
        
        return None  # Return None if no valid subgrid is found

    def create_grids(self):
        train_data = []
        for _ in range(random.randint(3, 4)):
            rows = random.randint(8, 25)
            cols = random.randint(8, 25)
            subrows = rows // 2
            subcols = cols // 2
            subgrid_color = random.randint(6, 9)
            taskvars = {'rows': rows, 'cols': cols, 'subrows': subrows, 'subcols': subcols, 'subgrid_color': subgrid_color}
            
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        rows = random.randint(8, 25)
        cols = random.randint(8, 25)
        subrows = rows // 2
        subcols = cols // 2
        subgrid_color = random.randint(6, 9)
        taskvars = {'rows': rows, 'cols': cols, 'subrows': subrows, 'subcols': subcols, 'subgrid_color': subgrid_color}
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]

        
        return taskvars, {'train': train_data, 'test': test_data}

