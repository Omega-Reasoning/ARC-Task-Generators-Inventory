from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity
import numpy as np
import random

class ARCTask25ff71a9Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "A single 4-way connected object is present in the input grid of color input_color(between 1 and 9).",
            "The remaining cells of the input grid are empty(0)."
        ]

        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "First copy the input grid to the output grid and identify the 4 way connected object.",
            "Identify the subgrid containing the object and shift the subgrid by one cell downwards."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        input_color = gridvars['input_color']

        def generate_grid():
            grid = np.zeros((rows, cols), dtype=int)
            object_height = random.randint(2, rows - 2)
            object_width = random.randint(2, cols - 2)

            obj = create_object(
                object_height, object_width, 
                color_palette=input_color,
                contiguity=Contiguity.FOUR
            )

            # Random placement ensuring it fits within the grid
            top_left_row = random.randint(0, rows - object_height - 1)
            top_left_col = random.randint(0, cols - object_width - 1)

            grid[top_left_row:top_left_row + object_height, top_left_col:top_left_col + object_width] = obj
            return grid

        # Ensure constraints are met
        return generate_grid()

    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        output_grid = np.zeros_like(grid)

        def flood_fill(grid, row, col, target_color, visited):
            if (row < 0 or row >= rows or col < 0 or col >= cols or 
                grid[row, col] != target_color or (row, col) in visited):
                return visited
            
            visited.add((row, col))
            
            # Check 4-way connected neighbors
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dr, dc in directions:
                flood_fill(grid, row + dr, col + dc, target_color, visited)
            
            return visited

        # Find the first non-zero cell and get all object cells
        object_cells = set()
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    object_color = grid[r, c]
                    object_cells = flood_fill(grid, r, c, object_color, set())
                    break
            if object_cells:
                break

        # Find the bounding box of the object
        if object_cells:
            min_row = min(r for r, _ in object_cells)
            max_row = max(r for r, _ in object_cells)
            min_col = min(c for _, c in object_cells)
            max_col = max(c for _, c in object_cells)

            # Extract the subgrid
            subgrid_height = max_row - min_row + 1
            subgrid_width = max_col - min_col + 1
            
            # Copy the entire subgrid region (including any empty cells within the bounding box)
            if max_row + 1 < rows:  # Only shift if there's space below
                # Copy the subgrid to the new position (shifted down by 1)
                output_grid[min_row + 1:max_row + 2, min_col:max_col + 1] = \
                    grid[min_row:max_row + 1, min_col:max_col + 1]

        return output_grid

    def create_grids(self):
        # Task variables
        taskvars = {
            'rows': random.randint(10, 22),
            'cols': random.randint(10, 22),
        }

        gridvars = {}
        train_pairs = []
        for _ in range(random.randint(3, 4)):
            gridvars['input_color'] = random.randint(1, 9)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))

        gridvars['input_color'] = random.randint(1, 9)
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)

        test_pair = [GridPair(input=test_input, output=test_output)]

        return taskvars, {'train': train_pairs, 'test': test_pair}

