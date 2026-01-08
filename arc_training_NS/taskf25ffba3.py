from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskf25ffba3(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['columns']}.",
            "In each input grid, each column contains a vertical bar starting from the bottom of the grid and extending upward to a height that is at most half the number of rows.",
            "In each input grid, the bars vary in both height and color.",
            "Most of the bars are of a single, uniform color, but a few contain two distinct colors, one stacked on top of the other."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is created by first copying the entire input grid, then reflecting the bottom part vertically onto the upper part.",
            "This results in a grid that is symmetrical along the horizontal center line, with the bottom section mirrored at the top."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        rows = random.randint(6, 30)  # Keep reasonable size, ensure rows > columns
        columns = random.randint(3, min(rows - 1, 15))  # Ensure columns < rows
        
        # Available colors (excluding background 0)
        available_colors = list(range(1, 10))
        
        taskvars = {
            'rows': rows,
            'columns': columns
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        
        # Create empty grid
        grid = np.zeros((rows, columns), dtype=int)
        
        # Available colors (excluding background 0)
        available_colors = list(range(1, 10))
        
        # Maximum bar height is half the number of rows
        max_bar_height = rows // 2
        
        # Create a bar for each column
        for col in range(columns):
            # Random height for this bar (at least 1, at most max_bar_height)
            bar_height = random.randint(1, max_bar_height)
            
            # Decide if this bar should have two colors (20% chance)
            use_two_colors = random.random() < 0.2
            
            if use_two_colors and bar_height >= 2:
                # Two colors: split the bar height
                color1 = random.choice(available_colors)
                color2 = random.choice([c for c in available_colors if c != color1])
                
                # Split height randomly but ensure both parts have at least 1 cell
                split_point = random.randint(1, bar_height - 1)
                
                # Bottom part (color1)
                for row in range(rows - split_point, rows):
                    grid[row, col] = color1
                
                # Top part (color2)
                for row in range(rows - bar_height, rows - split_point):
                    grid[row, col] = color2
            else:
                # Single color bar
                color = random.choice(available_colors)
                for row in range(rows - bar_height, rows):
                    grid[row, col] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        
        # Create output grid by copying input
        output_grid = grid.copy()
        
        # Find the center line
        center = rows // 2
        
        # Reflect bottom part to top part
        for row in range(center):
            # Mirror row: bottom row (rows-1-row) maps to top row (row)
            source_row = rows - 1 - row
            if source_row < rows:  # Safety check
                output_grid[row, :] = output_grid[source_row, :]
        
        return output_grid
