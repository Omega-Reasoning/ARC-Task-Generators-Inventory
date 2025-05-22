from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, random_cell_coloring, retry
from transformation_library import find_connected_objects, GridObject
import numpy as np
import random

class Taska5f85a15Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and of different sizes.",
            "The grid has multiple diagonal lines of {color('object_color')} color that run across from top-left to bottom-right, they can start from any column.",
            "The diagonal lines are placed well spaced."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "Each diagonal line has alternating fill colors in the output grid, namely {color('object_color')} color and {color('fill_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def color_name(self, color: int) -> str:
        color_map = {
            0: "black",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "cyan",
            9: "brown"
        }
        return color_map.get(color, f"color_{color}")
    
    def create_input(self, taskvars):
        # Define colors
        object_color = taskvars["object_color"]
        
        # Determine grid size (square)
        grid_size = random.randint(5, 20)
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Set a large minimum distance between diagonals (at least 25% of grid size)
        min_spacing = max(4, grid_size // 4)  # Substantial spacing
        
        # Based on spacing, limit the number of diagonals we can fit
        max_possible_diagonals = 1 + (grid_size // min_spacing)
        num_diagonals = random.randint(1, min(3, max_possible_diagonals))
        
        # For very small grids, limit to just 1 diagonal
        if grid_size < 8:
            num_diagonals = 1
        
        # For large spacing needs, we'll place diagonals at specific intervals
        if num_diagonals == 1:
            # Single diagonal - place randomly
            start_col = random.randint(0, grid_size-1)
            start_positions = [start_col]
        else:
            # Multiple diagonals - place with large, consistent spacing
            start_positions = []
            
            # Place first diagonal near start
            first_pos = random.randint(0, min(3, grid_size//6))
            start_positions.append(first_pos)
            
            # Calculate remaining positions with large spacing
            spacing = (grid_size - first_pos) // num_diagonals
            
            for i in range(1, num_diagonals):
                next_pos = first_pos + (i * spacing)
                # Make sure we're still in bounds
                if next_pos < grid_size:
                    start_positions.append(next_pos)
        
        # Create diagonal lines
        for start_col in start_positions:
            r, c = 0, start_col
            while r < grid_size and c < grid_size:
                grid[r, c] = object_color
                r += 1
                c += 1
        
        return grid
    
    def transform_input(self, input_grid):
        # Create output grid by copying input grid
        output_grid = np.copy(input_grid)
        
        object_color = self.taskvars["object_color"]
        fill_color = self.taskvars["fill_color"]
        
        # Find starting points of diagonal lines (top row or leftmost column)
        height, width = input_grid.shape
        start_positions = []
        
        # Check top row for diagonal starts
        for c in range(width):
            if input_grid[0, c] == object_color:
                start_positions.append((0, c))
        
        # Check leftmost column (excluding corner already checked)
        for r in range(1, height):
            if input_grid[r, 0] == object_color:
                start_positions.append((r, 0))
        
        # For each diagonal line, alternate colors
        for start_r, start_c in start_positions:
            r, c = start_r, start_c
            position = 0  # Position counter
            
            # Follow the diagonal
            while 0 <= r < height and 0 <= c < width and input_grid[r, c] == object_color:
                # Alternate colors based on position
                output_grid[r, c] = object_color if position % 2 == 0 else fill_color
                r += 1
                c += 1
                position += 1
        
        return output_grid
    
    def create_grids(self):
        # Create variables
        taskvars = {}
        
        # Pick a color for the diagonal lines and a fill color
        taskvars["object_color"] = random.randint(1, 8)  # Avoiding 0 (background)
        
        # Pick a different color for the alternating pattern
        fill_colors = list(range(1, 10))
        fill_colors.remove(taskvars["object_color"])
        taskvars["fill_color"] = random.choice(fill_colors)
        
        # Store taskvars as instance variable for access in transform_input
        self.taskvars = taskvars
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Replace {color('object_color')} etc. in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('object_color')}", color_fmt('object_color'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('object_color')}", color_fmt('object_color'))
                 .replace("{color('fill_color')}", color_fmt('fill_color'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate train and test pairs
        train_pairs = []
        num_train_pairs = random.randint(3, 5)
            
        for _ in range(num_train_pairs):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input)
        test_pairs = [GridPair(input=test_input, output=test_output)]
        
        # Return taskvars and TrainTestData object
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)