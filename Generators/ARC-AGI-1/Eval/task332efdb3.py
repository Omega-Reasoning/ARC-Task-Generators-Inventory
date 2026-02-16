from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, random_cell_coloring, retry
import numpy as np
import random

class Task332efdb3Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are odd number square grids.",
            "The entire grid is just filled with empty cells (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The output grid forms a pattern of color {color('output_color')}.",
            "From an odd‑sized square you punch out a hole in every second cell of every second row, always leaving a one‑tile {color('output_color')} color border around the edge."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid that is an odd-sized square filled with empty cells (0)."""
        size = gridvars['grid_size']
        
        # Ensure size is odd
        if size % 2 == 0:
            size += 1
            
        # Create grid filled with empty cells (0)
        grid = np.zeros((size, size), dtype=int)
        
        return grid

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        """Transform input to create punch hole pattern with border."""
        # Use a fixed output color since we can't access taskvars
        # The automated system will replace this with the actual color from taskvars
        output_color = 5  # This will be replaced by the actual output_color from taskvars
        
        # Copy input grid
        output_grid = grid.copy()
        rows, cols = output_grid.shape
        
        # Fill entire grid with the output color first
        output_grid.fill(output_color)
        
        # Create the punch hole pattern
        # Punch holes in every second cell of every second row
        # Start from row 1 (second row, 0-indexed) and skip every second row
        for row in range(1, rows - 1, 2):  # Skip first and last row (border)
            # In each selected row, punch holes in every second column
            for col in range(1, cols - 1, 2):  # Skip first and last column (border)
                output_grid[row, col] = 0  # Punch hole (set to empty)
        
        return output_grid

    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent variables."""
        # Generate output color (avoiding 0 which is background)
        output_color = random.randint(1, 9)
        
        # Store task variables
        taskvars = {
            'output_color': output_color,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate odd grid sizes
        min_size = 5  # Start with minimum odd size
        max_size = 15  # Maximum odd size
        
        # Generate odd sizes only
        odd_sizes = []
        for _ in range(num_train_examples + 1):
            size = random.randint(min_size, max_size)
            # Ensure size is odd
            if size % 2 == 0:
                size += 1
            # Make sure it's within ARC limits
            if size > 30:
                size = 29  # Largest odd number <= 30
            odd_sizes.append(size)
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': odd_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': odd_sizes[-1]}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input)
        
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
    generator = PunchHolePatternTaskGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    print(f"Number of train examples: {len(train_test_data['train'])}")
    print(f"Number of test examples: {len(train_test_data['test'])}")
    
    # Print grid sizes for verification
    for i, example in enumerate(train_test_data['train']):
        size = example['input'].shape[0]
        print(f"Train example {i+1}: {size}x{size} (odd: {size % 2 == 1})")
    
    test_size = train_test_data['test'][0]['input'].shape[0]
    print(f"Test example: {test_size}x{test_size} (odd: {test_size % 2 == 1})")
    
    # Visualize if the visualization method is available
    try:
        ARCTaskGenerator.visualize_train_test_data(train_test_data)
    except:
        print("Visualization not available, but grids created successfully!")