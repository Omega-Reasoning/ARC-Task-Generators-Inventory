from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskd631b094Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each input grid contains a random number of colored cells, all of which share the same randomly chosen color.",
            "Color of the colored cells varies in each input grid.",
            "The positions and count of these colored cells vary, in each input grid.",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The colored cells in the input grid are identified and counted, and the color of these cells is recorded.",
            "An output grid is constructed with a single row and a number of columns equal to the counted number, with each cell filled using the recorded color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        grid = np.zeros((n, n), dtype=int)
        
        # Get color for this specific grid (from gridvars if provided, otherwise random)
        if 'color' in gridvars:
            color = gridvars['color']
        else:
            # Choose a random non-background color
            color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # Randomly determine density (ensuring at least one colored cell)
        total_cells = n * n
        min_cells = 1
        max_cells = min(total_cells // 2, 15)  # Cap to reasonable number
        num_colored_cells = random.randint(min_cells, max_cells)
        density = num_colored_cells / total_cells
        
        # Apply random coloring
        random_cell_coloring(grid, color, density=density, background=0, overwrite=False)
        
        # Ensure we have at least one colored cell
        if np.sum(grid != 0) == 0:
            r, c = random.randint(0, n-1), random.randint(0, n-1)
            grid[r, c] = color
            
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find all non-zero (colored) cells
        colored_positions = np.where(grid != 0)
        
        if len(colored_positions[0]) == 0:
            # Edge case: no colored cells, return empty 1x1 grid
            return np.zeros((1, 1), dtype=int)
        
        # Count the colored cells
        count = len(colored_positions[0])
        
        # Get the color (all colored cells should have the same color)
        color = grid[colored_positions[0][0], colored_positions[1][0]]
        
        # Create output grid: single row with 'count' columns, all filled with the color
        output = np.full((1, count), color, dtype=int)
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        n = random.randint(5, 30)  # Grid size between 5 and 30
        taskvars = {'n': n}
        
        # Generate number of training examples (3-6)
        num_train = random.randint(3, 6)
        
        def generate_example():
            # Each example can have a different color
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            return {'input': input_grid, 'output': output_grid}
        
        # Generate training examples
        train_examples = []
        for _ in range(num_train):
            example = generate_example()
            train_examples.append(example)
        
        # Generate test example
        test_example = generate_example()
        test_examples = [test_example]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

