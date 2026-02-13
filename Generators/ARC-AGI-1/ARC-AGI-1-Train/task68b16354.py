from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task68b16354Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size nxn, where n is an odd number that varies across examples.",
            "They contain a completely filled grid with multiple colored objects in the colors {color('color1')}, {color('color2')}, {color('color3')}, {color('color4')}, and {color('color5')}.",
            "Each object is composed of 4-way connected cells of the same color.",
            "The positions and shapes of the objects vary across examples.",
            "There are no empty (0) cells in the grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input.",
            "It is constructed by reflecting the input grid vertically, using the middle row as the line of reflection.",
            "Rows below the middle row are reflected above, and rows above are reflected below.",
            "The middle row remains unchanged."
        ]
        
        # Initialize task variables dictionary
        taskvars_definitions = {}
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Define task variables with 5 distinct colors
        colors = random.sample(range(1, 10), 5)
        taskvars = {
            'color1': colors[0],
            'color2': colors[1],
            'color3': colors[2],
            'color4': colors[3],
            'color5': colors[4]
        }
        
        # Generate 3-4 train examples and 1 test example
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        for _ in range(num_train_examples):
            # Create a random size grid (odd number between 11 and 29)
            grid_size = random.choice([11, 13, 15, 17, 19, 21, 23, 25, 27, 29])
            
            input_grid = self.create_input(taskvars, {'size': grid_size})
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create a test example with a different grid size than the training examples
        test_size = random.choice([11, 13, 15, 17, 19, 21, 23, 25, 27, 29])
        while any(example['input'].shape[0] == test_size for example in train_examples):
            test_size = random.choice([11, 13, 15, 17, 19, 21, 23, 25, 27, 29])
        
        test_input = self.create_input(taskvars, {'size': test_size})
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        grid_size = gridvars.get('size', 15)  # Default size 15 if not specified
        
        # Initialize grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create a list of available colors
        colors = [taskvars['color1'], taskvars['color2'], taskvars['color3'], 
                  taskvars['color4'], taskvars['color5']]
        
        # Calculate how many "seed objects" to create
        num_seeds = random.randint(10, 20)
        
        # Create initial objects at random positions
        for _ in range(num_seeds):
            size = random.randint(3, 7)  # Random size for the object
            row = random.randint(0, grid_size - size)
            col = random.randint(0, grid_size - size)
            color = random.choice(colors)
            
            obj = create_object(
                size, size, 
                color_palette=color,
                contiguity=Contiguity.FOUR, 
                background=0
            )
            
            # Paste the object onto the grid
            for r in range(size):
                for c in range(size):
                    if obj[r, c] != 0:
                        grid[row + r, col + c] = obj[r, c]
        
        # Fill any remaining empty cells with random colors
        empty_cells = np.where(grid == 0)
        for i in range(len(empty_cells[0])):
            r, c = empty_cells[0][i], empty_cells[1][i]
            grid[r, c] = random.choice(colors)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Create a copy of the input grid
        output = grid.copy()
        
        # Get the grid size
        grid_size = grid.shape[0]
        
        # Find the middle row (since grid size is always odd)
        middle_row = grid_size // 2
        
        # Reflect the grid vertically around the middle row
        # Middle row stays the same, other rows are reflected
        for i in range(middle_row):
            # Swap row i with its mirror row (grid_size - 1 - i)
            output[i] = grid[grid_size - 1 - i]
            output[grid_size - 1 - i] = grid[i]
        
        return output

