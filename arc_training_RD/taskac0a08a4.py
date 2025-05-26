from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class ColorExpansionTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are {task_var('input_grid_size')} fixed size.",
            "Each grid contains between 1 and 6 colored cells, and each colored cell has a unique color.",
            "The positions of the colored cells within the {task_var('input_grid_size')} grid determine their placement in the output grid."
        ]
        
        transformation_reasoning_chain = [
            "The transformation follows these rules:",
            "For n colored cells, the output grid is of size 9x9, and each colored cell expands to a 3x3 block.",
            "Each block is placed in the output grid at the magnified position corresponding to its original cell position in the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars):
        input_size = taskvars["input_size"]
        n_objects = taskvars["num_colors"]
        colors = random.sample(range(1, 10), n_objects)  # Select unique colors
        
        # Create empty input_size x input_size grid
        grid = np.zeros((input_size, input_size), dtype=int)
        
        # Randomly place n colored cells
        positions = [(r, c) for r in range(input_size) for c in range(input_size)]
        selected_positions = random.sample(positions, n_objects)
        
        for (r, c), color in zip(selected_positions, colors):
            grid[r, c] = color
            
        return grid
    
    def transform_input(self, input_grid):
        input_size = self.taskvars["input_size"]  # Always 3
        block_size = input_size  # Always 3
        output_size = input_size * input_size  # 9

        output_grid = np.zeros((output_size, output_size), dtype=int)

        for r in range(input_size):
            for c in range(input_size):
                color = input_grid[r, c]
                if color > 0:
                    row_start = r * block_size
                    col_start = c * block_size
                    output_grid[row_start:row_start+block_size, col_start:col_start+block_size] = color

        return output_grid
    
    def create_grids(self):
        # Set input grid size
        input_size = 3  # Always 3x3
        num_colors = random.randint(1, 6)  # 1 to 6 colors
        
        # Create task variables
        taskvars = {
            "input_size": input_size,
            "input_grid_size": f"{input_size}x{input_size}",
            "num_colors": num_colors
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