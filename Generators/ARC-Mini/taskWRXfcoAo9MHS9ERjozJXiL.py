from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskWRXfcoAo9MHS9ERjozJXiLGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain three multi-colored (1-9) cells in the first row, while all other cells remain empty (0).",
            "The three colored cells are positioned at (0,0), (0,1) and (0,5)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and completely filling the third, fourth, and fifth columns, leaving the first-row cells unchanged.",
            "The three columns are filled with the three colors used in the first row, maintaining the same order."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Initialize an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Randomly assign three different colors (1-9)
        colors = random.sample(range(1, 10), 3)
        grid[0, 0], grid[0, 1], grid[0, 5] = colors
        
        return grid
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Extract colors from the first row
        color1, color2, color3 = grid[0, 0], grid[0, 1], grid[0, 5]
        
        # Fill the third, fourth, and fifth columns
        output_grid[1:, 2] = color1
        output_grid[1:, 3] = color2
        output_grid[1:, 4] = color3
        
        return output_grid
    
    def create_grids(self):
        taskvars = {
            'rows': random.randint(3, 30),
            'cols': random.randint(7, 30)
        }
        
        train_data = []
        for _ in range(random.randint(3, 4)):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}


