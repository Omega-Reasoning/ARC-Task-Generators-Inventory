from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects, GridObject
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random

class Task0c786b71Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "They contain a completely filled grid with colored objects and single-colored cells.",
            "Each grid has exactly 3 different colors, which vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {2 * vars['rows']} x {2 * vars['cols']}.",
            "They are constructed by copying the input grid and pasting it into the bottom-right quadrant of the output grid.",
            "The bottom-right quadrant is then reflected horizontally to create the bottom-left quadrant, and reflected vertically to create the top-right quadrant.",
            "Finally, the top-right quadrant is reflected horizontally to form the top-left quadrant."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        colors = gridvars['colors']
        
        # Create a grid and fill it completely with the 3 colors
        def generate_grid():
            grid = np.zeros((rows, cols), dtype=int)
            # Fill the entire grid with random colors from our palette
            for r in range(rows):
                for c in range(cols):
                    grid[r, c] = random.choice(colors)
            return grid
        
        # Ensure all 3 colors are used
        def has_all_colors(grid):
            unique_colors = set(np.unique(grid))
            return len(unique_colors) == 3 and unique_colors == set(colors)
        
        return retry(generate_grid, has_all_colors)
    
    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        output_rows = 2 * rows
        output_cols = 2 * cols
        
        # Create output grid
        output = np.zeros((output_rows, output_cols), dtype=int)
        
        # Step 1: Place input in bottom-right quadrant
        output[rows:, cols:] = grid
        
        # Step 2: Reflect bottom-right horizontally to create bottom-left
        output[rows:, :cols] = np.fliplr(grid)
        
        # Step 3: Reflect bottom-right vertically to create top-right
        output[:rows, cols:] = np.flipud(grid)
        
        # Step 4: Reflect top-right horizontally to create top-left
        output[:rows, :cols] = np.fliplr(np.flipud(grid))
        
        return output
    
    def create_grids(self):
        # Random task variables
        rows = random.randint(3, 15)
        cols = random.randint(3, 15)
        
        # Ensure output doesn't exceed 30x30
        while 2 * rows > 30 or 2 * cols > 30:
            rows = random.randint(3, 15)
            cols = random.randint(3, 15)
        
        taskvars = {
            'rows': rows,
            'cols': cols
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Choose 3 different colors for this example (excluding background 0)
            available_colors = list(range(1, 10))
            colors = random.sample(available_colors, 3)
            gridvars = {'colors': colors}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        colors = random.sample(available_colors, 3)
        gridvars = {'colors': colors}
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

