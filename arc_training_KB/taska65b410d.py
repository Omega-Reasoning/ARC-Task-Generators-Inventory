from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry

class Taska65b410dGenerator(ARCTaskGenerator):
    def __init__(self):
        # Initialize the reasoning chains
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain a single horizontal line of {color('object_color')} color and empty (0) cells.",
            "The horizontal line always starts from column 1 and can be positioned anywhere in the top half of the grid, except in the first row.",
            "The number of cells in the {color('object_color')} horizontal line are n, where n must be between 2 and 5.",
            "The number of cells in the {color('object_color')} horizontal line are decided according to the number of rows below it, which should be at least n-1."]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid, identifying the length n of the {color('object_color')} horizontal line and finally forming a right-angled triangle using {color('object_color')}, {color('fill_color1')} and {color('fill_color2')} cells.",
            "Once the {color('object_color')} horizontal line is identified, the rows above it are filled with {color('fill_color1')} color to form the top part of the triangle.",
            "The number of cells filled in each row above the {color('object_color')} horizontal line depends on the row.",
            "For each row above the horizontal line, start from the first column and if the row is directly above the {color('object_color')} horizontal line, add n+1 {color('fill_color1')} cells to it.",
            "Similarly, if the row is two rows above the {color('object_color')} horizontal line, n+2 {color('fill_color1')} cells are added.",
            "Continue like this until the first row is reached, which will have the widest {color('fill_color1')} line.",
            "Below the {color('object_color')} horizontal line, cells are filled with {color('fill_color2')} to form the bottom part of the triangle, with each row becoming narrower as it moves away from the horizontal line.",
            "For the row directly below the horizontal line, start from the first column and add n-1 {color('fill_color2')} cells.",
            "Similarly, if the row is two rows below the {color('object_color')} horizontal line, n-2 {color('fill_color2')} cells are added.",
            "Continue like this until the row where exactly one {color('fill_color2')} cell is added; this will be the shortest line."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Generate unique colors
        colors = random.sample(range(1, 10), 3)
        taskvars = {
            'object_color': colors[0],
            'fill_color1': colors[1],
            'fill_color2': colors[2]
        }
        
        # Create 3 training examples and 1 test example
        train_examples = []
        for _ in range(3):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        test_examples = [{
            'input': (input_grid := self.create_input(taskvars, {})),
            'output': self.transform_input(input_grid, taskvars)
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars, gridvars):
        # Generate random grid dimensions
        rows = random.randint(8, 20)
        cols = random.randint(8, 20)
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate the horizontal line width (2-5 cells)
        line_width = random.randint(2, 5)
        
        # Ensure we have enough rows below for the triangle
        min_row_position = 1  # Not in first row
        max_row_position = rows // 2 - 1  # In top half, with room for triangle below
        
        # Calculate position constraints
        valid_position = False
        while not valid_position:
            # Position the horizontal line
            line_row = random.randint(min_row_position, max(min_row_position, max_row_position))
            
            # Ensure we have enough rows below for the triangle (at least line_width-1 rows)
            if rows - line_row - 1 >= line_width - 1:
                valid_position = True
        
        # Place the horizontal line starting from column 1
        for col in range(line_width):
            grid[line_row, col] = taskvars['object_color']
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find the horizontal line
        line_row = None
        line_width = 0
        
        for row in range(grid.shape[0]):
            count = 0
            for col in range(grid.shape[1]):
                if grid[row, col] == taskvars['object_color']:
                    count += 1
            
            if count > 0:  # Found a row with the object color
                line_row = row
                line_width = count
                break
        
        # Fill the top part of the triangle (above the line)
        for r in range(line_row):
            # Calculate width of this row: n + (line_row - r)
            width = line_width + (line_row - r)
            width = min(width, grid.shape[1])  # Ensure we don't exceed grid width
            
            for c in range(width):
                output_grid[r, c] = taskvars['fill_color1']
        
        # Fill the bottom part of the triangle (below the line)
        for r in range(line_row + 1, grid.shape[0]):
            # Calculate width of this row: n - (r - line_row)
            width = line_width - (r - line_row)
            
            if width <= 0:  # Stop when we reach a row with 0 or negative width
                break
                
            for c in range(width):
                output_grid[r, c] = taskvars['fill_color2']
        
        return output_grid

