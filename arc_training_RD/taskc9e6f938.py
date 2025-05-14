from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, random_cell_coloring

class Taskc9e6f938Generator(ARCTaskGenerator):
    def __init__(self):

        input_reasoning_chain = [
            "Input grids are square grids of size M*M.",
            "The grid consists of one object of {{color(\"object_color\")}} color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same height but twice the width of the input grid (2M * M).",
            "The left half of the output grid is identical to the input grid.",
            "The right half contains the vertical mirror reflection of the entire input grid."
        ]
        
        taskvars_definitions = {
            "object_color": "Color of the object (between 1 and 9)"
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, gridvars=None):
        if gridvars is None:
            gridvars = {}
        
        object_color = gridvars.get("object_color", 1)
        grid_size = random.randint(3, 6)  # M can be between 3 and 6
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create a connected object anywhere in the grid
        while True:
            # Start with a random cell
            start_row = random.randint(0, grid_size - 1)
            start_col = random.randint(0, grid_size - 1)
            grid[start_row, start_col] = object_color
            
            # Add 2-4 more connected cells
            num_cells = random.randint(2, 4)
            cells_added = 1
            current_row, current_col = start_row, start_col
            
            while cells_added < num_cells:
                # Try to add adjacent cells (4-connectivity)
                directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]  # left, up, down, right
                random.shuffle(directions)
                
                for dr, dc in directions:
                    new_row, new_col = current_row + dr, current_col + dc
                    if (0 <= new_row < grid_size and 0 <= new_col < grid_size and 
                        grid[new_row, new_col] == 0):
                        grid[new_row, new_col] = object_color
                        current_row, current_col = new_row, new_col
                        cells_added += 1
                        break
                
                if cells_added >= num_cells:
                    break
            
            # Verify we have a valid connected object
            objects = find_connected_objects(grid, diagonal_connectivity=False)
            if len(objects.objects) == 1:  # Removed the rightmost column requirement
                break
            
            grid = np.zeros((grid_size, grid_size), dtype=int)
        
        return grid

    def transform_input(self, input_grid):
        output_rows = input_grid.shape[0]
        output_cols = input_grid.shape[1] * 2
        output_grid = np.zeros((output_rows, output_cols), dtype=int)
        
        # Copy input grid to left half
        output_grid[:, :input_grid.shape[1]] = input_grid
        
        # Mirror the entire input grid in the right half
        right_half = np.fliplr(input_grid)
        output_grid[:, input_grid.shape[1]:] = right_half
        
        return output_grid
    
    def create_grids(self):
        # Create a set of random colors for variety
        object_color = random.randint(1, 9)
        
        # Create grid variables
        gridvars = {
            "object_color": object_color
        }
        
        # Generate training pairs
        train_pairs = []
        num_examples = random.randint(3, 5)
        
        for _ in range(num_examples):
            input_grid = self.create_input(gridvars=gridvars)
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test example
        test_input = self.create_input(gridvars=gridvars)
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return gridvars, TrainTestData(train=train_pairs, test=test_pairs)

#{{color(\"object_color\")}}