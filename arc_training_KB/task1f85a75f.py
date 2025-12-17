from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects
import numpy as np
import random

class Task1f85a75fGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} x {vars['rows']}.",
            "The input grid contains exactly one main 4-way connected object.",
            "A random number of differently colored (1–9) cells are also present in the grid in the background.",
            "The remaining cells are empty (0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "The input grid contains exactly one main single-colored 4-way connected object of a specific color, with other randomly colored cells in the background.",
            "The output is formed by extracting the bounding box of the main object from the input grid."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)



    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        # Input color is stored in gridvars (not taskvars). If not provided, choose randomly.
        input_color = gridvars.get('input', random.randint(1, 9))
        

        # Create a blank grid
        grid = np.zeros((rows, rows), dtype=int)

        # Generate the 4-way connected object
        obj_height, obj_width = gridvars['object_size']
        # Create the object; `create_object` doesn't support `min_cells`,
        # so retry until we have at least `min_cells` colored cells.
        min_cells = 3
        object_matrix = create_object(obj_height, obj_width, color_palette=input_color, contiguity=Contiguity.FOUR)
        attempts = 0
        while (object_matrix != 0).sum() < min_cells:
            object_matrix = create_object(obj_height, obj_width, color_palette=input_color, contiguity=Contiguity.FOUR)
            attempts += 1
            if attempts > 100:
                # Give up after many attempts to avoid infinite loop; proceed with whatever we have
                break

        # Create a buffer zone around the object (1 cell minimum, 2 cells maximum)
        buffer = random.randint(1, 2)
        
        # Determine random position ensuring buffer zone
        max_row = rows - obj_height - (2 * buffer)
        max_col = rows - obj_width - (2 * buffer)
        
        # Ensure valid range
        if max_row < buffer or max_col < buffer:
            # If grid is too small, place object with minimal buffer
            buffer = 1
            max_row = rows - obj_height - buffer
            max_col = rows - obj_width - buffer
            top_left_row = max(0, buffer)
            top_left_col = max(0, buffer)
        else:
            top_left_row = random.randint(buffer, max_row)
            top_left_col = random.randint(buffer, max_col)

        # Create a mask for the object's area including buffer zone
        object_area = np.zeros_like(grid, dtype=bool)
        buffer_top = max(0, top_left_row - buffer)
        buffer_left = max(0, top_left_col - buffer)
        buffer_bottom = min(rows, top_left_row + obj_height + buffer)
        buffer_right = min(rows, top_left_col + obj_width + buffer)
        
        object_area[buffer_top:buffer_bottom, buffer_left:buffer_right] = True

        # Paste the object into the grid
        grid[top_left_row:top_left_row + obj_height, 
             top_left_col:top_left_col + obj_width] = object_matrix

        # Select up to 3 random colors different from input_color
        available_colors = [c for c in range(1, 10) if c != input_color]
        num_colors = random.randint(1, min(3, len(available_colors)))
        selected_colors = random.sample(available_colors, num_colors)

        # Create a mask for where we can add random colors (outside object area)
        valid_positions = ~object_area

        # Add random cell coloring only outside object area and buffer
        random_positions = np.random.random(grid.shape) < 0.3
        random_positions &= valid_positions  # Only color cells outside object area and buffer
        
        if np.any(random_positions):
            random_colors = np.random.choice(selected_colors, size=np.sum(random_positions))
            grid[random_positions] = random_colors

        return grid

    def transform_input(self, grid, taskvars):
        # `gridvars` should carry the example's input color; create_grids passes gridvars
        # as the second parameter when calling this method.
        gridvars = taskvars
        input_color = gridvars.get('input', None)

        # Identify objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)

        # Find the target object by color. If `input` not provided, pick the largest object.
        if input_color is not None:
            target_object = next(obj for obj in objects if obj.has_color(input_color))
        else:
            target_object = max(objects, key=lambda o: o.size)

        # Extract the bounding box
        box = target_object.bounding_box
        subgrid = grid[box[0], box[1]]

        return subgrid

    def create_grids(self):
        # Calculate minimum grid size needed
        # Max object size is 6x6, max buffer is 2, so minimum is 6 + 2*2 + 2 = 12
        min_grid_size = 12
        
        taskvars = {
            'rows': random.randint(min_grid_size, 30),
        }

        # Generate training and test grids
        train_data = []
        used_colors = set()
        for _ in range(random.randint(3, 4)):
            # Select a random input color for each example and put it in gridvars
            color = random.randint(1, 9)
            gridvars = {
                'object_size': (random.randint(3, 6), random.randint(3, 6)),
                'input': color
            }
            used_colors.add(color)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))

        # Create test data with its own random color
        # Choose a test color different from all training colors
        available_test_colors = [c for c in range(1, 10) if c not in used_colors]
        if available_test_colors:
            test_color = random.choice(available_test_colors)
        else:
            test_color = random.randint(1, 9)

        gridvars = {
            'object_size': (random.randint(3, 6), random.randint(3, 6)),
            'input': test_color
        }
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, gridvars)

        return taskvars, TrainTestData(train=train_data, test=[GridPair(input=test_input, output=test_output)])