from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
import numpy as np
import random
from typing import Dict, Tuple

class Taska61g2674Generator(ARCTaskGenerator):
    def __init__(self):
        # Use placeholders for colors in the reasoning chains
        input_reasoning_chain = [
            "Input grids are squares of size {vars['rows']}x{vars['cols']}.",
            "They only contain {color('object_color')} color objects and empty (0) cells.",
            "The {color('object_color')} cells form tower-like vertical structures of varying sizes and are placed randomly.",
            "Every alternative column must be left empty beside the tower.",
            "No two towers can have the same height."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the tallest and the shortest tower only.",
            "Once identified, fill the longest tower with {color('fill_color1')} and the shortest tower with {color('fill_color2')}."
        ]
        
        # Initialize with placeholder reasoning chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
        
    def create_input(self) -> np.ndarray:
        """Create an input grid with towers of different heights"""
        # Randomly determine grid size (between 5x5 and 12x12)
        size = random.randint(5, 12)
        
        # Create empty grid
        grid = np.zeros((size, size), dtype=int)
        
        # Get object color from taskvars (will be set in create_grids)
        object_color = 1  # Default value, will be overridden in create_grids
        if hasattr(self, 'taskvars') and 'object_color' in self.taskvars:
            object_color = self.taskvars['object_color']
        
        # Determine number of towers (2-6, but limited by available space)
        max_possible_towers = size // 2
        min_towers = min(2, max_possible_towers)
        max_towers = min(6, max_possible_towers)
        
        num_towers = random.randint(min_towers, max_towers)
        
        # Generate unique tower heights (between 2 and size-1)
        possible_heights = list(range(2, size))
        random.shuffle(possible_heights)
        tower_heights = possible_heights[:num_towers]
        
        # Select column positions (every other column)
        available_columns = list(range(0, size, 2))
        random.shuffle(available_columns)
        tower_columns = available_columns[:num_towers]
        
        # Create towers fixed to the bottom row
        for height, col in zip(tower_heights, tower_columns):
            start_row = size - height  # Ensure the tower is fixed to the bottom
            for row in range(start_row, size):
                grid[row, col] = object_color
        
        return grid

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        """
        Transform input grid by coloring the tallest tower with one color and the shortest tower with another color.
        
        1. Find all connected components in the input grid
        2. Identify the tallest tower (with maximum height)
        3. Identify the shortest tower (with minimum height)
        4. Create a new grid where only these two towers are colored
        5. The tallest tower is colored with fill_color1
        6. The shortest tower is colored with fill_color2
        """
        # Get fill colors from taskvars (will be set in create_grids)
        fill_color1 = 2  # Default value, will be overridden in create_grids
        fill_color2 = 3  # Default value, will be overridden in create_grids
        
        if hasattr(self, 'taskvars'):
            if 'fill_color1' in self.taskvars:
                fill_color1 = self.taskvars['fill_color1']
            if 'fill_color2' in self.taskvars:
                fill_color2 = self.taskvars['fill_color2']
        
        # Make a copy of the input grid
        output_grid = np.zeros_like(grid)
        
        # Find all connected objects (towers)
        objects = find_connected_objects(grid, diagonal_connectivity=False)
        
        # Get all non-zero objects
        non_zero_objects = []
        for obj in objects:
            if any(color != 0 for _, _, color in obj.cells):
                non_zero_objects.append(obj)
        
        # If fewer than 2 objects found, return the input grid
        if len(non_zero_objects) < 2:
            return grid
        
        # Sort objects by height (tallest first)
        sorted_objects = sorted(non_zero_objects, key=lambda obj: obj.height, reverse=True)
        
        # Get the tallest and shortest objects
        tallest = sorted_objects[0]
        shortest = sorted_objects[-1]
        
        # Color the tallest tower with fill_color1
        for r, c, _ in tallest.cells:
            output_grid[r, c] = fill_color1
        
        # Color the shortest tower with fill_color2
        for r, c, _ in shortest.cells:
            output_grid[r, c] = fill_color2
        
        return output_grid

    def create_grids(self) -> Tuple[Dict, TrainTestData]:
        # Define the test grid size
        test_grid_size = random.randint(7, 12)
        
        # Generate random colors for objects and fill
        object_color = random.randint(1, 9)
        
        # Choose different colors for fill_color1 and fill_color2
        available_colors = [c for c in range(1, 10) if c != object_color]
        fill_color1 = random.choice(available_colors)
        available_colors.remove(fill_color1)
        fill_color2 = random.choice(available_colors)
        
        # Store task variables
        self.taskvars = {
            'object_color': object_color,
            'fill_color1': fill_color1,
            'fill_color2': fill_color2,
            'rows': test_grid_size,
            'cols': test_grid_size
        }
        
        # Create 3-5 train pairs
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            # Use random sizes for training examples
            input_grid = self.create_input()
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair with the size specified in taskvars
        test_input = self.create_input()
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        # Return taskvars and TrainTestData object
        return self.taskvars, TrainTestData(train=train_pairs, test=test_pairs)