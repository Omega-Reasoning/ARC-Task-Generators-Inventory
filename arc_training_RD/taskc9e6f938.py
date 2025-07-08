from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, Contiguity, random_cell_coloring

class Taskc9e6f938Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square grids of size {vars['grid_size']} x {vars['grid_size']}.",
            "The grid consists of one object of {color('object_color')} color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same height but twice the width of the input grid.",
            "The left half of the output grid is identical to the input grid.",
            "The right half contains the vertical mirror reflection of the entire input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create a square grid with one connected object."""
        object_color = taskvars["object_color"]
        grid_size = taskvars["grid_size"]
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
            if len(objects.objects) == 1:
                break
            
            grid = np.zeros((grid_size, grid_size), dtype=int)
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input by creating a horizontally mirrored output grid."""
        output_rows = grid.shape[0]
        output_cols = grid.shape[1] * 2
        output_grid = np.zeros((output_rows, output_cols), dtype=int)
        
        # Copy input grid to left half
        output_grid[:, :grid.shape[1]] = grid
        
        # Mirror the entire input grid in the right half
        right_half = np.fliplr(grid)
        output_grid[:, grid.shape[1]:] = right_half
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Randomly select colors ensuring they are all different
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)

        # Choose a random grid size
        grid_size = random.randint(3, 6)

        taskvars = {
            'object_color': available_colors[0],
            'grid_size': grid_size,
        }

        # Generate 3-5 training examples with same task variables
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example with same task variables
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

# Test code
if __name__ == "__main__":
    generator = Taskc9e6f938Generator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)