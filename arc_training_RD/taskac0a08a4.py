from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class ColorExpansionTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are {task_var('input_grid_size')} fixed size.",
            "Each grid contains 1 to 4 colored cells, and each colored cell has a unique color.",
            "The positions of the colored cells within the {task_var('input_grid_size')} grid determine their placement in the output grid."
        ]
        
        transformation_reasoning_chain = [
            "The transformation follows these rules:",
            "- If there is only 1 colored cell in the bottom-right corner, the output grid is identical to the input grid.",
            "- If there is only 1 colored cell in the top row, it expands to fill the entire column.",
            "- If there are exactly 2 colored cells, the output grid is identical to the input grid.",
            "- If there are 3 or more colored cells, each colored cell expands to a 3×3 block of the same color.",
            "- For 3+ colored cells, the output grid is 12×12, and the 3×3 blocks are positioned based on the original cell positions.",
            "- Top row cells expand to blocks at rows 0-2, middle row at rows 5-7, and bottom row at rows 8-10.",
            "- Left column cells expand to blocks at columns 0-2, middle at 5-7, and right at 8-10."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars):
        # Get input grid size from taskvars
        input_size = taskvars["input_size"]
        
        # Create a grid with n colored cells where n is between 1 and 4
        n_objects = random.randint(1, 4)
        colors = random.sample(range(1, 10), n_objects)  # Select unique colors
        
        # Create empty input_size x input_size grid
        grid = np.zeros((input_size, input_size), dtype=int)
        
        # Randomly place n colored cells
        positions = []
        for r in range(input_size):
            for c in range(input_size):
                positions.append((r, c))
        
        selected_positions = random.sample(positions, n_objects)
        
        for (r, c), color in zip(selected_positions, colors):
            grid[r, c] = color
            
        return grid
    
    def transform_input(self, input_grid):
        # Get input grid size
        input_size = self.taskvars["input_size"]
        
        # Find all colored cells
        colored_cells = []
        for r in range(input_size):
            for c in range(input_size):
                if input_grid[r, c] > 0:
                    colored_cells.append((r, c, input_grid[r, c]))
        
        num_colored_cells = len(colored_cells)
        
        # Handle case with no colored cells
        if num_colored_cells == 0:
            return np.zeros((input_size, input_size), dtype=int)
        
        # Handle single cell cases
        if num_colored_cells == 1:
            r, c, color = colored_cells[0]
            
            # If it's in the top row, fill the entire column
            if r == 0:
                output_grid = np.zeros((input_size, input_size), dtype=int)
                for row in range(input_size):
                    output_grid[row, c] = color
                return output_grid
            
            # Otherwise, no change
            return input_grid.copy()
        
        # For exactly 2 colored cells, return input unchanged
        if num_colored_cells == 2:
            return input_grid.copy()
        
        # For 3+ colored cells, create a 12×12 output grid
        output_grid = np.zeros((12, 12), dtype=int)
        
        # Place each colored cell as a 3×3 block at a specific position
        for r, c, color in colored_cells:
            # Determine row and column placement in output grid
            if r == 0:  # Top row
                row_start = 0
            elif r == 1:  # Middle row
                row_start = 5
            else:  # Bottom row (r == 2)
                row_start = 8
            
            if c == 0:  # Left column
                col_start = 0
            elif c == 1:  # Middle column
                col_start = 5
            else:  # Right column (c == 2)
                col_start = 8
            
            # Fill a 3×3 block with the color
            for dr in range(3):
                for dc in range(3):
                    output_grid[row_start + dr, col_start + dc] = color
        
        return output_grid
    
    def create_grids(self):
        # Set input grid size
        input_size = 3  # Fixed at 3x3 as per the reasoning
        
        # Create task variables
        taskvars = {
            "input_size": input_size,
            "input_grid_size": f"{input_size}x{input_size}"
        }
        
        # Store taskvars as instance variable for access in transform_input
        self.taskvars = taskvars
        
        # Update reasoning chains with task variables
        self.input_reasoning_chain = [
            chain.replace("{task_var('input_grid_size')}", taskvars["input_grid_size"])
            for chain in self.input_reasoning_chain
        ]
        
        self.transformation_reasoning_chain = [
            chain.replace("{task_var('input_grid_size')}", taskvars["input_grid_size"])
                 .replace("{task_var('input_size')}", str(taskvars["input_size"]))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate train examples
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input)
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_examples)