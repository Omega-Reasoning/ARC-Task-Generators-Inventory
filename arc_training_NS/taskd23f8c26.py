from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import random_cell_coloring

class Taskd23f8c26(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is of size n x n.",
            "n is a positive odd integer greater than 1 and varies for each input grid.",
            "m random cells in the grid are colored using k random colors, where m and k are integers greater than 1. The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First copy the output grid.",
            "Any cell aside from the cells on the middle column is converted to empty(0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        taskvars = {}
        
        # Generate 3-6 training examples
        num_train_examples = random.randint(3, 6)
        
        train_examples = []
        for _ in range(num_train_examples):
            # Generate random grid size (odd number between 5 and 13)
            n = random.choice([i for i in range(3, 14, 2)])
            
            # For each training example, generate random parameters
            gridvars = {
                'size': n, 
                'num_colors': random.randint(2, 5),  # k random colors
                'density': random.uniform(0.3, 0.7)  # controls m (number of colored cells)
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate one test example
        n = random.choice([i for i in range(3, 14, 2)])
        gridvars = {
            'size': n,
            'num_colors': random.randint(2, 5),
            'density': random.uniform(0.3, 0.7)
        }
        
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        
        test_examples = [{'input': input_grid, 'output': output_grid}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars, gridvars):
        # Create an empty grid of size n x n
        n = gridvars['size']
        grid = np.zeros((n, n), dtype=int)
        
        # Generate a list of k random colors (1-9)
        colors = random.sample(range(1, 10), gridvars['num_colors'])
        
        # Randomly color cells in the grid
        return random_cell_coloring(
            grid=grid,
            color_palette=colors,
            density=gridvars['density'],
            background=0
        )
    
    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Calculate the middle column index
        middle_col = grid.shape[1] // 2
        
        # Set all cells except those in the middle column to empty
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if col != middle_col:
                    output_grid[row, col] = 0
        
        return output_grid
