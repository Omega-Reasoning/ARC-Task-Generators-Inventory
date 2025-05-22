from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject

class Tasktaskabf363dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid and has size {grid_size} x {grid_size}.",
            "The grid consists of one cell randomly placed on the grid of {color('main_color')} color and there exists an object of any pattern on the grid of {color('object_color')} color."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "The pattern is also copied from the input grid with its {color('object_color')} color maintained.",
            "The random single cell of {color('main_color')} color is removed in the output grid."
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
        # Generate a grid of size MxM (between 5 and 20)
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Get colors from taskvars
        main_color = taskvars['main_color']
        object_color = taskvars['object_color']
        
        # Create a random pattern object
        object_size = random.randint(3, min(grid_size-2, 10))
        pattern_grid = create_object(
            height=object_size, 
            width=object_size, 
            color_palette=[object_color], 
            contiguity=Contiguity.EIGHT
        )
        
        # Place the pattern at a random position
        r_pos = random.randint(0, grid_size - object_size)
        c_pos = random.randint(0, grid_size - object_size)
        
        for r in range(object_size):
            for c in range(object_size):
                if pattern_grid[r, c] != 0:
                    grid[r + r_pos, c + c_pos] = pattern_grid[r, c]
        
        # Place a single cell of main_color at a random empty position
        empty_cells = [(r, c) for r in range(grid_size) for c in range(grid_size) if grid[r, c] == 0]
        if empty_cells:
            r_main, c_main = random.choice(empty_cells)
            grid[r_main, c_main] = main_color
        else:
            # Unlikely, but just in case the pattern fills the entire grid
            # Find a position to overwrite
            r_main, c_main = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            grid[r_main, c_main] = main_color
            
        return grid

    def transform_input(self, input_grid):
        # Copy the input grid
        output_grid = input_grid.copy()
        
        # Get all objects from the grid
        objects = find_connected_objects(input_grid, diagonal_connectivity=True, background=0)
        
        # Find and store the color of the single-cell object
        single_cell_color = None
        for obj in objects.objects:
            if len(obj.cells) == 1:
                r, c = next(iter(obj.cells))[:2]
                single_cell_color = input_grid[r, c]
                obj.cut(output_grid)
                break  # Assume only one single-cell object

        # Find the largest object (main pattern) and set its color to the single cell's color
        if objects.objects and single_cell_color is not None:
            main_obj = max(objects.objects, key=lambda o: len(o.cells))
            for cell in main_obj.cells:
                r, c = cell[:2]
                output_grid[r, c] = single_cell_color

        return output_grid

    def create_grids(self):
        # Randomize grid size between 5 and 20 (or any range you prefer)
        size = random.randint(5, 20)
        main_color = random.randint(1, 9)
        object_color = random.randint(1, 9)
        
        # Ensure different colors
        while object_color == main_color:
            object_color = random.randint(1, 9)
        
        # Create variables dictionary for grid generation
        taskvars = {
            'grid_size': size,
            'main_color': main_color,
            'object_color': object_color
        }
        
        # Store taskvars as instance variable for access in transform_input
        self.taskvars = taskvars
        
        # Helper for reasoning chain formatting
        def color_fmt(key):
            color_id = taskvars[key]
            return f"{self.color_name(color_id)} ({color_id})"

        # Replace {grid_size} and color placeholders in reasoning chains
        self.input_reasoning_chain = [
            chain.replace("{grid_size}", str(taskvars['grid_size']))
                 .replace("{color('main_color')}", color_fmt('main_color'))
                 .replace("{color('object_color')}", color_fmt('object_color'))
            for chain in self.input_reasoning_chain
        ]
        self.transformation_reasoning_chain = [
            chain.replace("{color('main_color')}", color_fmt('main_color'))
                 .replace("{color('object_color')}", color_fmt('object_color'))
            for chain in self.transformation_reasoning_chain
        ]
        
        # Create 3-5 training pairs
        num_train = random.randint(3, 5)
        train_pairs = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars)    
            output_grid = self.transform_input(input_grid)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Create test pair
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input)
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_pairs, test=test_examples)