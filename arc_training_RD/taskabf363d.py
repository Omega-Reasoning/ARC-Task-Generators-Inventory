from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject

class Tasktaskabf363dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size {{vars['grid_size']}}x{{vars['grid_size']}}.",
            "The grid consists of one cell randomly placed on the grid of {{color(\"main_color\")}} color and there exists an object of any pattern on the grid of {{color(\"object_color\")}} color."
        ]
        transformation_reasoning_chain = [
            "The output grid is copied  from the input grid.",
            "The pattern is also copied from the input grid, but the pattern is replaced with {{color(\"object_color\")}} color",
            "The random single cell is removed in the output grid."
        ]
        taskvars_definitions = {
            "grid_size": "Size of the square grid (e.g., 8x8, 15x15)",
            "main_color": "Color of the single cell",
            "object_colors": "Color of the pattern object",
        }
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, gridvars):
        # Generate a grid of size MxM (between 5 and 20)
        grid_size = gridvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Get colors from gridvars
        main_color = gridvars['block_color']
        object_color = gridvars['object_color']
        
        # Create a random pattern object
        object_size = random.randint(3, min(grid_size-2, 10))
        pattern_grid = create_object(
            height=object_size, 
            width=object_size, 
            color_palette=object_color, 
            contiguity=Contiguity.EIGHT
        )
        
        # Place the pattern at a random position
        r_pos = random.randint(0, grid_size - object_size)
        c_pos = random.randint(0, grid_size - object_size)
        
        for r in range(object_size):
            for c in range(object_size):
                if pattern_grid[r, c] != 0:
                    grid[r + r_pos, c + c_pos] = pattern_grid[r, c]
        
        # Place a single cell of main_color at a random empty position
        empty_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r, c] == 0]
        if empty_cells:
            r_main, c_main = random.choice(empty_cells)
            grid[r_main, c_main] = main_color
        else:
            # Unlikely, but just in case the pattern fills the entire grid
            # Find a position to overwrite
            r_main, c_main = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            grid[r_main, c_main] = main_color
            
        return grid

    def transform_input(self, input_grid):
        # Copy the input grid
        output_grid = input_grid.copy()
        
        # Get all objects from the grid
        objects = find_connected_objects(input_grid, diagonal_connectivity=True, background=0)
        
        # Remove the single-cell object (which is our "random red cell")
        for obj in objects:
            if len(obj) == 1:  # This should be our single cell
                obj.cut(output_grid)  # Remove from output grid
        
        return output_grid

    def create_grids(self):
        # Initialize task variables with random values
        size = random.randint(8, 15)  # Larger grid for better spacing
        main_color = random.randint(1, 9)
        object_color = random.randint(1, 9)
        
        # Ensure different colors
        while object_color == main_color:
            object_color = random.randint(1, 9)
        
        # Create variables dictionary for grid generation
        gridvars = {
            'grid_size': size,  # Use integer for computation
            'grid_size_display': f"{size}x{size}",  # Use string for display
            'block_color': main_color,
            'object_color': object_color
        }
        
        # Create 3-5 training pairs
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            input_grid = self.create_input(gridvars)    
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(gridvars)
        test_output = self.transform_input(test_input)
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        # Update taskvars with display version of grid_size
        taskvars = {
            'grid_size': gridvars['grid_size_display'],  # Use display format for task variables
            'main_color': main_color,
            'object_color': object_color
        }
        
        return taskvars, TrainTestData(train=train_pairs, test=test_examples)