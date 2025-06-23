from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class Taskac0a08a4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are 3 x 3 fixed size.",
            "Each grid contains between 2 and 6 colored cells, and each colored cell has a unique color.",
            "The positions of the colored cells within the input grid determine their placement in the output grid."
        ]
        
        transformation_reasoning_chain = [
            "The transformation follows these rules:",
            "For n colored cells, the output grid size is determined by n × input_size (where n is the number of colors).",
            "Each colored cell expands to an n×n block in the output grid.",
            "Each block is placed in the output grid at the magnified position corresponding to its original cell position in the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with randomly placed colored cells."""
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
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by expanding each colored cell to n×n blocks."""
        input_size = taskvars["input_size"]
        num_colors = taskvars["num_colors"]
        block_size = num_colors  # Block size equals number of colors
        output_size = input_size * block_size  # Output size is input_size × num_colors

        output_grid = np.zeros((output_size, output_size), dtype=int)

        for r in range(input_size):
            for c in range(input_size):
                color = grid[r, c]  # Fixed: use 'grid' instead of 'input_grid'
                if color > 0:
                    row_start = r * block_size
                    col_start = c * block_size
                    output_grid[row_start:row_start+block_size, col_start:col_start+block_size] = color

        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Set input grid size and number of colors
        input_size = 3  # Always 3x3
        num_colors = random.randint(2, 6)  # 2 to 6 colors (minimum 2)
        
        taskvars = {
            "input_size": input_size,
            "input_grid_size": input_size,
            "num_colors": num_colors
        }
        
        # Create 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}  # No grid-specific variables needed
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }