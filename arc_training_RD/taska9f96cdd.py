import random
import numpy as np
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject

class AddFourColorsAroundCellGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size MxN.",
            "The grid consists of only one cell randomly placed on the grid of {{color(\"main_color\")}} color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied  from the input grid.", 
            "The main cell is also copied from the input grid.",
            "For the transformation in the output grid, the position of the main cell plays an important role. First we need to check if there are cells available for 8 way connection from the main cell; if it is available, then fills 4 cells that are 8-way diaginally connected to the main cells with each cell of different colors namely {{color(\"object_color1\")}} color, {{color(\"object_color2\")}} color, {{color(\"object_color3\")}} color and {{color(\"object_color4\")}} color.",
            "Note that the requirement is to fill only the available cells around the main cell with different colors. If there are fewer than 4 cells available around the main cell due to its position (e.g., if it's near or at the grid edge), then we should simply fill whatever cells are available."
            "The main cell color must not change."
        ]
        
        taskvars_definitions = {
            "main_color": "Color of the main cell",
            "object_colors": "Colors of the 4 cells around the main cell"
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Define color mapping
        main_color = random.randint(1, 9)
        
        # Choose 4 distinct colors different from main_color
        available_colors = [i for i in range(1, 10) if i != main_color]
        object_colors = random.sample(available_colors, 4)
        
        # Set grid variables
        gridvars = {
            "main_color": main_color,
            "object_colors": object_colors
        }
        
        # Generate train/test data
        num_train_pairs = random.randint(3, 5)
        train_pairs = [self.create_example(gridvars) for _ in range(num_train_pairs)]
        test_pairs = [self.create_example(gridvars)]
        
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)
    
    def create_example(self, gridvars):
        input_grid = self.create_input(gridvars)
        output_grid = self.transform_input(input_grid.copy(), gridvars)
        return GridPair(input = input_grid, output= output_grid)
    
    def create_input(self, gridvars):
        # Create a random size grid between 5x5 and 20x20
        size = random.randint(5, 20)
        grid = np.zeros((size, size), dtype=np.int32)
        
        # Allow the main cell to be placed anywhere in the grid
        # This creates more diversity including edge cases
        row = random.randint(0, size - 1)
        col = random.randint(0, size - 1)
        
        grid[row, col] = gridvars["main_color"]
        
        return grid
    
    def transform_input(self, grid, gridvars):
        # Find the main cell
        main_objects = find_connected_objects(grid, background=0)
        main_cell = None
        
        for obj in main_objects:
            if gridvars["main_color"] in obj.colors:
                main_cell = obj
                break
        
        if main_cell is None:
            return grid  # No main cell found
        
        # Get the position of the main cell
        for r, c, color in main_cell.cells:
            main_r, main_c = r, c
            break
        
        # Define the 8-way connections (orthogonal and diagonal)
        neighbor_positions = [
            (-1, -1),          (-1, 1),

            (1, -1),            (1, 1)
        ]
        
        # Check which positions are available (within grid bounds)
        available_positions = []
        for dr, dc in neighbor_positions:
            new_r, new_c = main_r + dr, main_c + dc
            if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                available_positions.append((new_r, new_c))
        
        # Randomly shuffle available positions to add variety
        random.shuffle(available_positions)
        
        # Fill available cells (up to 4) with different colors
        for i, (r, c) in enumerate(available_positions[:min(4, len(available_positions))]):
            grid[r, c] = gridvars["object_colors"][i]
        
        return grid
