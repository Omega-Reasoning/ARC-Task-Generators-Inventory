from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, random_cell_coloring, Contiguity
from Framework.transformation_library import find_connected_objects
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
        input_color = gridvars.get('input', random.randint(1, 9))

        grid = np.zeros((rows, rows), dtype=int)

        obj_height, obj_width = gridvars['object_size']
        min_cells = 3
        object_matrix = create_object(obj_height, obj_width, color_palette=input_color, contiguity=Contiguity.FOUR)
        attempts = 0
        while (object_matrix != 0).sum() < min_cells:
            object_matrix = create_object(obj_height, obj_width, color_palette=input_color, contiguity=Contiguity.FOUR)
            attempts += 1
            if attempts > 100:
                break

        buffer = random.randint(1, 2)
        max_row = rows - obj_height - (2 * buffer)
        max_col = rows - obj_width - (2 * buffer)
        
        if max_row < buffer or max_col < buffer:
            buffer = 1
            max_row = rows - obj_height - buffer
            max_col = rows - obj_width - buffer
            top_left_row = max(0, buffer)
            top_left_col = max(0, buffer)
        else:
            top_left_row = random.randint(buffer, max_row)
            top_left_col = random.randint(buffer, max_col)

        object_area = np.zeros_like(grid, dtype=bool)
        buffer_top = max(0, top_left_row - buffer)
        buffer_left = max(0, top_left_col - buffer)
        buffer_bottom = min(rows, top_left_row + obj_height + buffer)
        buffer_right = min(rows, top_left_col + obj_width + buffer)
        
        object_area[buffer_top:buffer_bottom, buffer_left:buffer_right] = True

        grid[top_left_row:top_left_row + obj_height,
             top_left_col:top_left_col + obj_width] = object_matrix

        available_colors = [c for c in range(1, 10) if c != input_color]
        num_colors = random.randint(1, min(3, len(available_colors)))
        selected_colors = random.sample(available_colors, num_colors)

        valid_positions = ~object_area

        # Place background cells ensuring each one is isolated (no 4-way neighbors)
        for r in range(rows):
            for c in range(rows):
                if valid_positions[r, c] and np.random.random() < 0.1:
                    neighbors_empty = all(
                        not (0 <= r+dr < rows and 0 <= c+dc < rows and grid[r+dr, c+dc] != 0)
                        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]
                    )
                    if neighbors_empty:
                        grid[r, c] = random.choice(selected_colors)

        return grid

    def transform_input(self, grid, taskvars):
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)

        # Main object is always the largest since background cells are isolated singles
        target_object = max(objects, key=lambda o: o.size)

        box = target_object.bounding_box
        subgrid = grid[box[0], box[1]]

        return subgrid

    def create_grids(self):
        min_grid_size = 12
        
        taskvars = {
            'rows': random.randint(min_grid_size, 30),
        }

        train_data = []
        used_colors = set()
        for _ in range(random.randint(3, 4)):
            color = random.randint(1, 9)
            gridvars = {
                'object_size': (random.randint(3, 6), random.randint(3, 6)),
                'input': color
            }
            used_colors.add(color)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))

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
        test_output = self.transform_input(test_input, taskvars)

        return taskvars, TrainTestData(train=train_data, test=[GridPair(input=test_input, output=test_output)])