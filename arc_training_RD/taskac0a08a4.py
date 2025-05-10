from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class ColorExpansionTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are 3x3 fixed size..",
            "The grid consists of 1 or more colored cells (between 1-9).Each color is unique."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is a larger grid whose size is determined by output_size = 3 * number_of_colored_cells.",
            "So, if there are n colored cells, the output grid will be of size 3n x 3n.",
            "Identify all colored cells in the 3x3 input grid. Each represents a unique colored object.",
            "For each such cell at position (i, j): Map its position to the output grid by placing a 3×3 block of that color at position: output_row = 3 * object_index + i, output_col = 3 * object_index + j (where object_index is the order in which colored cells are found — top to bottom, left to right).",
            "Alternatively, maintain relative positions and just scale them up.",
            "Ensure that the color and position scaling (3×3 expansion) is preserved for each object."
        ]
        
        taskvars_definitions = {}
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self):
        # Create a 3x3 grid with n colored cells where n is between 1 and 9
        n_objects = random.randint(1, 9)
        colors = random.sample(range(1, 10), n_objects)  # Select unique colors
        
        # Create empty 3x3 grid
        grid = np.zeros((3, 3), dtype=int)
        
        # Randomly place n colored cells
        positions = []
        for r in range(3):
            for c in range(3):
                positions.append((r, c))
        
        selected_positions = random.sample(positions, n_objects)
        
        for (r, c), color in zip(selected_positions, colors):
            grid[r, c] = color
            
        return grid
    
    def transform_input(self, input_grid):
        # Identify all colored cells in the input grid
        colored_cells = []
        for r in range(3):
            for c in range(3):
                if input_grid[r, c] != 0:
                    colored_cells.append((r, c, input_grid[r, c]))
        
        # Calculate object_index for each cell (top to bottom, left to right)
        # Sort by row first, then by column
        colored_cells.sort()
        
        # Calculate output grid size based on number of colored cells
        n = len(colored_cells)
        output_size = 3 * n
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # For each colored cell, place a 3×3 block in the output grid
        for object_index, (i, j, color) in enumerate(colored_cells):
            # Calculate the position for the 3×3 block
            output_row = 3 * object_index + i
            output_col = 3 * object_index + j
            
            # Place a 3×3 block of that color at position (output_row, output_col)
            for dr in range(-1, 2):  # -1, 0, 1
                for dc in range(-1, 2):  # -1, 0, 1
                    nr = output_row + dr
                    nc = output_col + dc
                    
                    # Ensure we stay within grid bounds
                    if 0 <= nr < output_size and 0 <= nc < output_size:
                        output_grid[nr, nc] = color
        
        return output_grid
    
    def create_grids(self):
        # Determine number of train examples (between 3 and 5)
        num_train = random.randint(3, 5)
        
        # Generate train examples
        train_pairs = []
        for _ in range(num_train):
            input_grid = self.create_input()    
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input()
        test_output = self.transform_input(test_input)
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        taskvars = {}  # No task variables needed
        
        return taskvars, TrainTestData(train=train_pairs, test=test_examples)

