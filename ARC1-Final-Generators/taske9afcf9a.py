from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taske9afcf9aGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size 2 x {vars['columns']}.",
            "The first row is filled with one color.",
            "The second row is filled with a different color.",
            "The specific colors used vary from one input grid to another."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The colors in the first and second rows are identified.",
            "The first row is filled with the two colors in an alternating pattern, beginning with the color that was originally in the leftmost cell of the first row.",
            "The second row is filled with the two colors in an alternating pattern, beginning with the color that was originally in the leftmost cell of the second row."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create a 2×N grid with each row filled with a different solid color."""
        columns = taskvars['columns']
        
        # Choose two different colors
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        colors = random.sample(available_colors, 2)
        
        # Create the grid
        grid = np.zeros((2, columns), dtype=int)
        grid[0, :] = colors[0]  # First row with first color
        grid[1, :] = colors[1]  # Second row with second color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform the input by making each row alternate between the two colors."""
        columns = taskvars['columns']
        
        # Get the two colors from the input grid
        color1 = grid[0, 0]  # Original color of first row
        color2 = grid[1, 0]  # Original color of second row
        
        # Create output grid
        output = grid.copy()
        
        # Fill first row with alternating pattern starting with its original color
        for col in range(columns):
            if col % 2 == 0:
                output[0, col] = color1
            else:
                output[0, col] = color2
        
        # Fill second row with alternating pattern starting with its original color
        for col in range(columns):
            if col % 2 == 0:
                output[1, col] = color2
            else:
                output[1, col] = color1
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and generate train/test grids."""
        # Randomly choose number of training examples (3-6)
        num_train = random.randint(3, 6)
        num_test = 1
        
        # Choose a random number of columns (5-30)
        columns = random.randint(5, 30)
        
        taskvars = {
            'columns': columns
        }
        
        # Generate training examples
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_examples = []
        for _ in range(num_test):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            test_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
