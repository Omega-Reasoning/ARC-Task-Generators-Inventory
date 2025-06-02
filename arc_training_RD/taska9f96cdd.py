import random
import numpy as np
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject

class Taska9f96cddGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are grids of size {vars['grid_rows']} x {vars['grid_cols']}.",
            "The grid consists of only one cell randomly placed on the grid of {color('main_color')} color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.", 
            "The main cell is not copied from the input grid.",
            "For the transformation in the output grid, the position of the main cell plays an important role. First we need to check if there are cells available for 8 way connection from the main cell; if it is available, then fills 4 cells that are 8-way diagonally connected to the main cells with each cell of different colors namely {color('object_color1')}, {color('object_color2')}, {color('object_color3')} and {color('object_color4')}.",
            "Note that the requirement is to fill only the available cells around the main cell with different colors. If there are fewer than 4 cells available around the main cell due to its position (e.g., if it is near or at the grid edge), then we should simply fill whatever cells are available.",
            "The main cell color must not change."
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
    
    def create_grids(self):
        # Define color mapping
        main_color = random.randint(1, 9)
        
        # Choose 4 distinct colors different from main_color
        available_colors = [i for i in range(1, 10) if i != main_color]
        object_colors = random.sample(available_colors, 4)
        
        # Set task variables with grid dimensions and colors
        taskvars = {
            "grid_rows": 3,
            "grid_cols": 5,
            "main_color": main_color,
            "object_colors": object_colors,
            "object_color1": object_colors[0],  # Top-left diagonal
            "object_color2": object_colors[1],  # Top-right diagonal
            "object_color3": object_colors[2],  # Bottom-left diagonal
            "object_color4": object_colors[3]   # Bottom-right diagonal
        }
        
        # Store taskvars as instance variable for access in transform_input
        self.taskvars = taskvars
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Replace placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{color('main_color')}", color_fmt('main_color'))
                 .replace("{vars['grid_rows']}", str(taskvars['grid_rows']))
                 .replace("{vars['grid_cols']}", str(taskvars['grid_cols']))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('main_color')}", color_fmt('main_color'))
                 .replace("{color('object_color1')}", color_fmt('object_color1'))
                 .replace("{color('object_color2')}", color_fmt('object_color2'))
                 .replace("{color('object_color3')}", color_fmt('object_color3'))
                 .replace("{color('object_color4')}", color_fmt('object_color4'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Generate train/test data
        num_train_pairs = random.randint(3, 5)
        train_pairs = [self.create_example(taskvars) for _ in range(num_train_pairs)]
        test_pairs = [self.create_example(taskvars)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)
    
    def create_example(self, taskvars):
        input_grid = self.create_input(taskvars)
        output_grid = self.transform_input(input_grid.copy(), taskvars)
        return GridPair(input=input_grid, output=output_grid)
    
    def create_input(self, taskvars):
        # Use fixed grid size from task variables
        rows = taskvars['grid_rows']
        cols = taskvars['grid_cols']
        grid = np.zeros((rows, cols), dtype=np.int32)
        
        # Allow the main cell to be placed anywhere in the grid
        # This creates more diversity including edge cases
        row = random.randint(0, rows - 1)
        col = random.randint(0, cols - 1)
        
        grid[row, col] = taskvars["main_color"]
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Find the main cell
        main_objects = find_connected_objects(grid, background=0)
        main_cell = None
        
        for obj in main_objects.objects:  # Use .objects to access the list of objects
            if taskvars["main_color"] in obj.colors:
                main_cell = obj
                break
        
        if main_cell is None:
            return grid  # No main cell found
        
        # Get the position of the main cell
        for r, c, color in main_cell.cells:
            main_r, main_c = r, c
            break
        
        # Create output grid without the main cell
        output_grid = np.zeros_like(grid)
        
        # Define the 4 diagonal positions with their consistent color mapping
        diagonal_positions = {
            (-1, -1): taskvars["object_color1"],  # Top-left diagonal
            (-1, 1): taskvars["object_color2"],   # Top-right diagonal
            (1, -1): taskvars["object_color3"],   # Bottom-left diagonal
            (1, 1): taskvars["object_color4"]     # Bottom-right diagonal
        }
        
        # Fill each diagonal position if it's within bounds
        for (dr, dc), color in diagonal_positions.items():
            new_r, new_c = main_r + dr, main_c + dc
            if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                output_grid[new_r, new_c] = color
        
        return output_grid