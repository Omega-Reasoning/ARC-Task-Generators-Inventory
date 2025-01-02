from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object
from transformation_library import GridObject
import numpy as np
import random

class Tasktaske5vP8V72jk8pw3xU7zUfMmGenerator(ARCTaskGenerator):
    def __init__(self):
        # Initialize reasoning chains
        input_reasoning_chain = [
            "Input grids are of size nxn.",
            "Each input grid contains multi-colored (1-9) cells."
        ]
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid and replace the color of each column in the output grid with the color of its corresponding cell in the first row of the input grid."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        size = taskvars['grid_size']
        grid = np.random.randint(1, 10, (size, size))  # Random colors (1-9)
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        output_grid = grid.copy()
        first_row = grid[0, :]
        
        for col in range(grid.shape[1]):
            output_grid[:, col] = first_row[col]
        
        return output_grid

    def create_grids(self) -> tuple[dict, TrainTestData]:
        # Randomize grid size between 5 and 30
        grid_size = random.randint(5, 30)
        taskvars = {'grid_size': grid_size}
        
        # Create train/test grids
        train_examples = random.randint(3, 6)
        train = [
            {
                'input': (grid := self.create_input(taskvars, {})),
                'output': self.transform_input(grid, taskvars)
            }
            for _ in range(train_examples)
        ]
        
        test = [
            {
                'input': (grid := self.create_input(taskvars, {})),
                'output': self.transform_input(grid, taskvars)
            }
        ]
        
        return taskvars, {'train': train, 'test': test}

# Test the generator
if __name__ == "__main__":
    generator = ColumnColoringTask()
    taskvars, data = generator.create_grids()
    ARCTaskGenerator.visualize_train_test_data(data)
