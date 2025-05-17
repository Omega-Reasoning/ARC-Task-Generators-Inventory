from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects
from input_library import Contiguity, retry

class Taskba26e723Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have exactly 3 rows and can have any number of columns.",
            "The grid follows an alternating pattern: in the first column, the top two cells are filled; in the next column, the bottom two cells are filled. This alternation continues across all columns.",
            "The filled cells are evenly spaced and use the color {color(\"fill_color\")}.",
            "In some grids, the pattern starts with the bottom two cells of the first column being filled, followed by the top two cells in the second column, maintaining the alternating sequence."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The first column is filled with a {color(\"highlight_color\")} color, referred to as the highlight color.",
            "Starting from the first column, every third column (i.e., columns at positions 0, 3, 6, 9, ...) is also filled with the highlight color.",
            "All other filled columns follow the standard color {color(\"fill_color\")}.",
            "This pattern continues uniformly across the entire grid."
        ]
        
        taskvars_definitions = {
            "fill_color": "The primary color used for filling cells (yellow)",
            "highlight_color": "The color used to highlight every third column (pink)",
            "start_pattern": "Whether the pattern starts with top two cells filled (0) or bottom two cells filled (1)"
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars):
        # Fixed number of rows
        rows = 3
        
        # Random number of columns, between 5 and 20
        cols = random.randint(5, 20)
        
        # Define colors
        fill_color = taskvars.get("fill_color", 4)  # Default to yellow (4)
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Decide starting pattern (top two filled or bottom two filled)
        start_pattern = taskvars.get("start_pattern", random.choice([0, 1]))
        
        # Fill the grid with alternating pattern
        for col in range(cols):
            # If column is even, follow starting pattern. If odd, do the opposite
            is_top_filled = (col % 2 == 0) if start_pattern == 0 else (col % 2 == 1)
            
            if is_top_filled:
                # Fill top two cells
                grid[0, col] = fill_color
                grid[1, col] = fill_color
            else:
                # Fill bottom two cells
                grid[1, col] = fill_color
                grid[2, col] = fill_color
        
        return grid
    
    def transform_input(self, input_grid, taskvars):
        # Copy the input grid
        output_grid = input_grid.copy()
        
        # Get colors
        fill_color = taskvars.get("fill_color", 4)  # Default to yellow (4)
        highlight_color = taskvars.get("highlight_color", 6)  # Default to pink (6)
        
        # Apply the highlight color to every third column starting from 0
        for col in range(0, output_grid.shape[1], 3):
            # Find non-zero cells in this column and change their color
            for row in range(output_grid.shape[0]):
                if output_grid[row, col] == fill_color:
                    output_grid[row, col] = highlight_color
        
        return output_grid
    
    def create_grids(self):
        # Generate random values for task variables
        # Make sure fill_color and highlight_color are different
        fill_color = random.randint(1, 9)
        highlight_color = random.choice([c for c in range(1, 10) if c != fill_color])
        start_pattern = random.choice([0, 1])  # 0 for top-first, 1 for bottom-first
        
        taskvars = {
            "fill_color": fill_color,
            "highlight_color": highlight_color
        }
        
        # Generate 3-5 training pairs
        num_train_pairs = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train_pairs):
            # For some variety, allow different starting patterns in training examples
            train_taskvars = taskvars.copy()
            train_taskvars["start_pattern"] = random.choice([0, 1])
            
            input_grid = self.create_input(train_taskvars)
            output_grid = self.transform_input(input_grid, train_taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test pair
        test_taskvars = taskvars.copy()
        test_input = self.create_input(test_taskvars)
        test_output = self.transform_input(test_input, test_taskvars)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)