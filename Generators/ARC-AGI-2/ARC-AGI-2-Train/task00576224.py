from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import random_cell_coloring, retry

class Task00576224Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a completely filled grid, with no empty (0) cells.",
            "The grids are filled with multi-colored (1–9) cells.",
            "The cells are placed in such a way that the first and last columns of the input grid are different."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {3*vars['grid_size']}x{3*vars['grid_size']}.",
            "They are constructed by copying the input grid and pasting it 9 times in the output grid, starting from (0,0) and going until ({3*vars['grid_size'] - 1}, {3*vars['grid_size'] - 1}).",
            "In the top and bottom {vars['grid_size']} rows, the input grid is pasted directly.",
            "In all middle rows, the input grid is slightly changed.",
            "The slight change involves swapping the first and last columns of the input grid.",
            "These modified grids are then pasted into the middle rows."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> (dict, TrainTestData):
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(2, 10)
        }
        
        # Create 3 train examples and 1 test example
        train_examples = []
        for _ in range(3):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        test_examples = []
        input_grid = self.create_input(taskvars, {})
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples.append({'input': input_grid, 'output': output_grid})
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        
        # Initialize grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Predicate function to check no same-colored cells are 4-way connected
        def no_same_color_adjacency(g):
            # Check if first and last columns are different
            if np.array_equal(g[:, 0], g[:, -1]):
                return False
                
            # Check if there are any same-colored adjacent cells
            for r in range(grid_size):
                for c in range(grid_size):
                    color = g[r, c]
                    # Check 4-way neighbors
                    neighbors = []
                    if r > 0: 
                        neighbors.append((r-1, c))
                    if r < grid_size-1: 
                        neighbors.append((r+1, c))
                    if c > 0: 
                        neighbors.append((r, c-1))
                    if c < grid_size-1: 
                        neighbors.append((r, c+1))
                    
                    for nr, nc in neighbors:
                        if g[nr, nc] == color:
                            return False
            return True
        
        # Generate a grid with random colors, ensuring no adjacent same colors
        def generate_grid():
            # Start with all cells as color 1
            g = np.ones((grid_size, grid_size), dtype=int)
            
            # For each cell, assign a random color ensuring no adjacency
            for r in range(grid_size):
                for c in range(grid_size):
                    # Get colors of adjacent cells
                    adjacent_colors = []
                    if r > 0: 
                        adjacent_colors.append(g[r-1, c])
                    if r < grid_size-1: 
                        adjacent_colors.append(g[r+1, c])
                    if c > 0: 
                        adjacent_colors.append(g[r, c-1])
                    if c < grid_size-1: 
                        adjacent_colors.append(g[r, c+1])
                    
                    # Choose a random color different from adjacent colors
                    available_colors = [i for i in range(1, 10) if i not in adjacent_colors]
                    g[r, c] = random.choice(available_colors)
            
            # If first and last columns happen to be the same, try to fix it
            if np.array_equal(g[:, 0], g[:, -1]):
                # Choose a random row and change the last column color
                row = random.randint(0, grid_size-1)
                # Find a color different from adjacent colors and the first column
                adjacent_colors = [g[row, -2]]  # Left neighbor
                if row > 0: 
                    adjacent_colors.append(g[row-1, -1])  # Top neighbor
                if row < grid_size-1: 
                    adjacent_colors.append(g[row+1, -1])  # Bottom neighbor
                adjacent_colors.append(g[row, 0])  # First column color
                
                available_colors = [i for i in range(1, 10) if i not in adjacent_colors]
                if available_colors:
                    g[row, -1] = random.choice(available_colors)
            
            return g
        
        # Use retry to generate a valid grid
        grid = retry(generate_grid, no_same_color_adjacency, max_attempts=100)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        output_size = 3 * grid_size
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Create modified grid with swapped first and last columns
        modified_grid = grid.copy()
        modified_grid[:, 0], modified_grid[:, -1] = grid[:, -1].copy(), grid[:, 0].copy()
        
        # Paste grids to create output
        for row in range(3):
            for col in range(3):
                start_r = row * grid_size
                start_c = col * grid_size
                
                # Middle row gets modified grid, others get original
                if row == 1:
                    source_grid = modified_grid
                else:
                    source_grid = grid
                
                # Copy the source grid to the appropriate location
                for r in range(grid_size):
                    for c in range(grid_size):
                        output_grid[start_r + r, start_c + c] = source_grid[r, c]
        
        return output_grid

