from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject

class Taskaabf363dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid and has size {vars['grid_size']} x {vars['grid_size']}.",
            "The grid consists of one cell randomly placed on the grid of a distinct color and there exists an object of any pattern on the grid of another distinct color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The random single cell is removed from the output grid.",
            "The pattern object takes on the color of the removed single cell."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with a pattern object and a single colored cell."""
        grid_size = gridvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Randomly select 2 different colors for each grid
        available_colors = list(range(1, 10))
        main_color, object_color = random.sample(available_colors, 2)
        
        # Create a random pattern object
        object_size = random.randint(3, min(grid_size-2, 10))
        pattern_grid = create_object(
            height=object_size, 
            width=object_size, 
            color_palette=[object_color], 
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

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by removing single cell and changing pattern color."""
        # Copy the input grid
        output_grid = grid.copy()  # Fixed: use 'grid' instead of 'input_grid'
        
        # Get all objects from the grid
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)  # Fixed: use 'grid'
        
        # Find and store the color of the single-cell object
        single_cell_color = None
        for obj in objects.objects:
            if len(obj.cells) == 1:
                r, c = next(iter(obj.cells))[:2]
                single_cell_color = grid[r, c]  # Fixed: use 'grid'
                # Remove the single cell from output grid
                output_grid[r, c] = 0
                break  # Assume only one single-cell object

        # Find the largest object (main pattern) and set its color to the single cell's color
        if objects.objects and single_cell_color is not None:
            # Find the pattern object (not the single cell)
            pattern_obj = None
            for obj in objects.objects:
                if len(obj.cells) > 1:  # This is the pattern, not the single cell
                    pattern_obj = obj
                    break
            
            if pattern_obj:
                # Change the pattern's color to the single cell's color
                for cell in pattern_obj.cells:
                    r, c = cell[:2]
                    output_grid[r, c] = single_cell_color

        return output_grid

    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # No task-level variables needed for this task
        taskvars = {}
        
        # Create 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate different grid sizes
        grid_sizes = [random.randint(5, 20) for _ in range(num_train_examples + 1)]
        
        for i in range(num_train_examples):
            gridvars = {'grid_size': grid_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': grid_sizes[-1]}
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