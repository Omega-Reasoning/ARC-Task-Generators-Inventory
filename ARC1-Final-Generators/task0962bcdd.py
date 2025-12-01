from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import find_connected_objects
import numpy as np
import random

class Task0962bcddGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have size {vars['grid_size']} x {vars['grid_size']}",
            "The input grid consists of two plus shaped objects where each plus shape has:",
            "* Arms (top, left, bottom, and right) of color between(1-9)",
            "* Center cell of color between(1-9)",
            "* All the remaining cells are empty cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid i.e. {vars['grid_size']} x {vars['grid_size']}.",
            "Get all the 8-way connected plus shaped objects first.",
            "Each object in output grid is formed by scaling the arm length of plus shaped object by 2 with arm color and also filling the diagonals of the subgrid formed by the plus-shaped object, i.e. (top-left to bottom right and top-right to bottom-left) with the color of the center cell of the plus-shaped object.",
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        object_color_1 = gridvars['object_color_1']
        object_color_2 = gridvars['object_color_2']

        grid = np.zeros((grid_size, grid_size), dtype=int)

        def place_plus():
            # Keep arm length fixed at 1
            arm_length = 1
            
            # Reduce padding to give more space for placement
            padding = arm_length + 1  # Changed from 2 * arm_length
            if grid_size <= 2 * padding:
                return None
            
            center_row = random.randint(padding, grid_size - padding - 1)
            center_col = random.randint(padding, grid_size - padding - 1)

            coords = []
            # Center
            coords.append((center_row, center_col))
            # Arms
            for i in range(1, arm_length + 1):
                coords.append((center_row + i, center_col))
                coords.append((center_row - i, center_col))
                coords.append((center_row, center_col + i))
                coords.append((center_row, center_col - i))

            # Reduce minimum distance requirement
            min_distance = 4  # Changed from 4
            for r in range(grid_size):
                for c in range(grid_size):
                    if grid[r, c] != 0:
                        for coord_r, coord_c in coords:
                            if abs(r - coord_r) + abs(c - coord_c) < min_distance:
                                return None

            # Place plus shape
            for r, c in coords:
                grid[r, c] = object_color_1 if (r, c) != (center_row, center_col) else object_color_2

            return center_row, center_col

        # Place first plus shape
        retry(
            lambda: place_plus(),
            lambda result: result is not None,
            max_attempts=10
        )

        # Place second plus shape
        retry(
            lambda: place_plus(),
            lambda result: result is not None,
            max_attempts=10
        )

        return grid

    def transform_input(self, grid, taskvars):
        grid_size = grid.shape[0]
        
        # Find the two unique non-zero colors in the grid
        unique_colors = sorted(list(set(grid.flatten()) - {0}))
        
        # Count occurrences of each color
        # Arms will have more occurrences (4 cells) than center (1 cell)
        color_counts = {color: np.sum(grid == color) for color in unique_colors}
        object_color_1 = max(color_counts.items(), key=lambda x: x[1])[0]  # Color with more occurrences (arms)
        object_color_2 = min(color_counts.items(), key=lambda x: x[1])[0]  # Color with fewer occurrences (center)
        
        output_grid = np.zeros_like(grid)
        
        # Find center positions of plus shapes (cells with color_2)
        center_positions = [(r, c) for r in range(grid_size) 
                           for c in range(grid_size) 
                           if grid[r, c] == object_color_2]
        
        for center_row, center_col in center_positions:
            # Scale arm length by 2 (original arm length is 1)
            scaled_length = 2
            
            # Draw scaled plus shape
            output_grid[center_row, center_col] = object_color_2  # Center
            
            # Vertical and horizontal arms
            for i in range(-scaled_length, scaled_length + 1):
                if i != 0:  # Skip center
                    if 0 <= center_row + i < grid_size:
                        output_grid[center_row + i, center_col] = object_color_1
                    if 0 <= center_col + i < grid_size:
                        output_grid[center_row, center_col + i] = object_color_1
            
            # Fill diagonals
            for i in range(-scaled_length, scaled_length + 1):
                if (0 <= center_row + i < grid_size and 
                    0 <= center_col + i < grid_size):
                    output_grid[center_row + i, center_col + i] = object_color_2
                if (0 <= center_row + i < grid_size and 
                    0 <= center_col - i < grid_size):
                    output_grid[center_row + i, center_col - i] = object_color_2

        return output_grid

    def create_grids(self):
        # Create initial color pairs
        color_pairs = []
        available_colors = list(range(1, 10))
        
        # Generate unique color pairs for each example (3 train + 1 test)
        for _ in range(4):
            random.shuffle(available_colors)
            color_1 = available_colors[0]
            color_2 = available_colors[1]
            color_pairs.append((color_1, color_2))
        
        # Initialize base_taskvars with both grid_size and the first pair of colors
        base_taskvars = {
            'grid_size': random.randint(15, 26)
        }
        
        train_pairs = []
        test_pairs = []
        
        # Generate training examples with different colors
        for i in range(3):
            gridvars = base_taskvars.copy()
            gridvars['object_color_1'] = color_pairs[i][0]
            gridvars['object_color_2'] = color_pairs[i][1]
            grid = self.create_input(base_taskvars, gridvars)
            transformed = self.transform_input(grid, base_taskvars)
            train_pairs.append(GridPair({"input": grid, "output": transformed}))
        
        # Generate test example
        gridvars = base_taskvars.copy()
        gridvars['object_color_1'] = color_pairs[3][0]
        gridvars['object_color_2'] = color_pairs[3][1]
        grid = self.create_input(base_taskvars, gridvars)
        transformed = self.transform_input(grid, base_taskvars)
        test_pairs = [GridPair({"input": grid, "output": transformed})]
        
        return base_taskvars, TrainTestData({"train": train_pairs, "test": test_pairs})