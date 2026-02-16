from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Utility functions from libraries
from Framework.input_library import Contiguity
from Framework.transformation_library import find_connected_objects, GridObject

class TaskWQYEzNmJ3MSqQYjGYJEenWGenerator(ARCTaskGenerator):

    def __init__(self):
        # 1) Input Reasoning Chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains a column with two objects; a single {color('cell_color1')} cell and one or two connected {color('cell_color2')} cells.",
            "The remaining cells are empty (0)."
        ]
        # 2) Transformation Reasoning Chain
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid and check if the two objects are connected.",
            "If connected, remove the {color('cell_color1')} cell and replace the {color('cell_color2')} cell(s) with {color('cell_color3')} cell(s).",
            "If not connected, remove both the objects."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Create random task variables (e.g., distinct colors) and build the train/test data
        with the required conditions. Ensures at least one train example is connected,
        one is disconnected, and the test example uses a different column from all trains.
        """
        # Step 1: Initialize task variables
        # We want distinct colors for cell_color1, cell_color2, cell_color3
        distinct_colors = random.sample(range(1, 10), 3)
        taskvars = {
            'cell_color1': distinct_colors[0],
            'cell_color2': distinct_colors[1],
            'cell_color3': distinct_colors[2],
        }

        # We will generate three train grids and one test grid
        train_data = []
        used_columns = []

        # We enforce: one connected, one disconnected among train examples
        # We'll do exactly two: one connected, one disconnected, plus one random.
        scenarios = ['connected', 'disconnected', random.choice(['connected', 'disconnected'])]
        
        # Create the train examples
        for scenario in scenarios:
            input_grid = self.create_input(taskvars, {'scenario': scenario, 'used_columns': used_columns})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })

        # Create the test example. Make sure the test example's column is different
        # from all used_columns in the train data.
        test_input = self.create_input(taskvars, {'scenario': random.choice(['connected', 'disconnected']),
                                                  'used_columns': used_columns,
                                                  'force_different_col': True})
        test_output = self.transform_input(test_input, taskvars)

        test_data = [{'input': test_input, 'output': test_output}]

        train_test_data = {
            'train': train_data,
            'test': test_data
        }

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid:
          - Random size between 6 and 10 rows and columns.
          - Exactly two objects in a single column:
              (1) One cell of color cell_color1.
              (2) Two adjacent (connected) cells of color cell_color2.
          - If gridvars['scenario'] == 'connected', place the single cell so
            it touches the 2-cell object in 4-direction adjacency.
          - If gridvars['scenario'] == 'disconnected', place them apart so
            they do not touch.
          - gridvars may contain 'force_different_col' to ensure the column is not in used_columns.
        """
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']

        scenario = gridvars.get('scenario', 'connected')
        used_cols = gridvars.get('used_columns', [])
        force_different_col = gridvars.get('force_different_col', False)

        # Randomly choose grid shape
        rows = random.randint(6, 30)
        cols = random.randint(6, 30)

        # We want to find a column that is not used if force_different_col is True
        # or we allow any otherwise. We'll keep trying a random column until we
        # succeed or we've tried multiple times.
        attempts = 0
        max_attempts = 50
        while True:
            attempts += 1
            chosen_col = random.randint(0, cols-1)
            if force_different_col and (chosen_col in used_cols):
                if attempts >= max_attempts:
                    # fallback: just pick any column if we can't find a new one
                    break
                continue
            # If we get here, chosen_col is valid
            break

        # Keep track of the chosen column to ensure the test example is different
        if chosen_col not in used_cols:
            used_cols.append(chosen_col)

        # Initialize grid to all zeros
        grid = np.zeros((rows, cols), dtype=int)

        # We place two connected color2 cells in chosen_col, and one color1 cell in the same column
        # so that they are in the same column but possibly connected or disconnected

        # For the 2 connected cells of color2
        # pick a starting row for the top cell so that we have space for the second cell below
        if rows > 1:
            top_cell_row = random.randint(0, rows - 2)  # so we can place top_cell_row and top_cell_row+1
        else:
            top_cell_row = 0

        grid[top_cell_row, chosen_col] = color2
        grid[top_cell_row + 1, chosen_col] = color2

        # For the single color1 cell, we choose a row so that it's adjacent or not
        # depending on scenario
        if scenario == 'connected':
            # Ensure adjacency (4-neighbors). We'll randomly pick whether it touches
            # the top or bottom cell of color2
            which_to_touch = random.choice([0, 1])  # 0 or 1
            base_row = top_cell_row + which_to_touch
            # We can place color1 either above, below, left, or right. But we want same column -> must place above or below
            # to maintain "same column" in the puzzle statement. For guaranteed adjacency, place it above or below.
            # Because the puzzle states: "Each input grid contains a column with two objects."
            # So let's place it either above or below the color2 cell in the same column to ensure adjacency.
            if random.random() < 0.5:
                # Place above
                new_row = max(0, base_row - 1)
            else:
                # Place below
                new_row = min(rows - 1, base_row + 1)
            grid[new_row, chosen_col] = color1

        else:
            # scenario == 'disconnected'
            # Place the color1 cell far away (in same column but leaving at least
            # one empty row in between so they cannot be adjacent).
            # E.g., if color2 block is at [top_cell_row, top_cell_row+1],
            # we place color1 at a row that is at least 2 rows away from that block.
            # If there's not enough space, we can place it in row 0 or row (rows-1).
            dist = random.randint(2, rows // 2)  # random distance away
            # randomly place above or below
            if random.random() < 0.5:
                new_row = max(0, top_cell_row - dist)
            else:
                new_row = min(rows - 1, top_cell_row + 1 + dist)
            grid[new_row, chosen_col] = color1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars):
        """
        Transform according to the transformation reasoning chain:
          1) Copy input grid and check if the two objects are connected
             (here, 'connected' means color1 object touches color2 object in 4-direction adjacency).
          2) If connected: remove color1 cell and replace color2 with color3.
          3) If not connected: remove both objects.
        """
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']
        color3 = taskvars['cell_color3']

        # Make a copy
        output_grid = grid.copy()

        # Find objects with 4-connectivity, monochromatic
        objects = find_connected_objects(output_grid,
                                         diagonal_connectivity=False,
                                         background=0,
                                         monochromatic=True)

        # We expect exactly two objects: one of color1 and one of color2
        # But let's handle the possibility of extra noise or repeated placement gracefully.
        obj_color1 = objects.with_color(color1)
        obj_color2 = objects.with_color(color2)

        # Check adjacency
        connected = False
        if len(obj_color1) > 0 and len(obj_color2) > 0:
            # We only consider the first object in each collection (assuming there's exactly one).
            # If multiple, we just use the first as the puzzle states "a single cell" and "two connected cells".
            c1 = obj_color1[0]
            c2 = obj_color2[0]
            # The puzzle states "connected" means physically touching in 4 directions
            connected = c1.touches(c2, diag=False)

        if connected:
            # Remove color1 object, replace color2 with color3
            if len(obj_color1) > 0:
                for (r, c, _) in obj_color1[0].cells:
                    output_grid[r, c] = 0  # remove it
            if len(obj_color2) > 0:
                for (r, c, _) in obj_color2[0].cells:
                    output_grid[r, c] = color3
        else:
            # Remove both objects
            if len(obj_color1) > 0:
                for (r, c, _) in obj_color1[0].cells:
                    output_grid[r, c] = 0
            if len(obj_color2) > 0:
                for (r, c, _) in obj_color2[0].cells:
                    output_grid[r, c] = 0

        return output_grid


