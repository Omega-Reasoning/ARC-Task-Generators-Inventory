from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, random_cell_coloring
import numpy as np
import random

class ARCTask1c786137Generator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "Each cell in the input grid can have any color or be empty (0).",
            "The input grid contains a rectangular subgrid whose perimeter (frame) is a single color; that frame color may vary per grid."
        ]

        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "First identify the rectangular subgrid whose perimeter is a single color (same color around its border).",
            "Extract and return the interior of that subgrid (the output grid)."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        # Only 'rows' and 'cols' are provided in taskvars. Subgrid size and its frame
        # color are chosen locally and may vary per grid.
        rows, cols = taskvars['rows'], taskvars['cols']

        # Choose subgrid size and position first so we can ensure the interior
        # does not contain the frame color.
        # Requirements:
        #  - frame (subgrid) must be bigger than 3x3 -> subrows, subcols >= 4
        #  - interior must contain at least 3 cells -> (subrows-2)*(subcols-2) >= 3
        min_sub = 4
        max_subrows = max(min_sub, rows // 2)
        max_subcols = max(min_sub, cols // 2)

        # Pick sizes until the interior has at least 3 cells. Add a deterministic
        # fallback if random attempts fail (shouldn't for typical rows/cols range).
        attempts = 0
        while True:
            subrows = random.randint(min_sub, max_subrows)
            subcols = random.randint(min_sub, max_subcols)
            if (subrows - 2) * (subcols - 2) >= 3:
                break
            attempts += 1
            if attempts > 100:
                found = False
                for sr in range(min_sub, max_subrows + 1):
                    for sc in range(min_sub, max_subcols + 1):
                        if (sr - 2) * (sc - 2) >= 3:
                            subrows, subcols = sr, sc
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
                # As a last resort, force minimal valid sizes
                subrows, subcols = min_sub, min_sub
                break

        # Position the subgrid randomly
        start_row = random.randint(0, rows - subrows)
        start_col = random.randint(0, cols - subcols)

        # Available colors and chosen frame color
        all_colors = list(range(1, 10))  # colors 1..9
        frame_color = random.choice(all_colors)
        # Background colors exclude the frame color to ensure the interior and
        # other cells do not accidentally use the same color as the frame.
        background_colors = [c for c in all_colors if c != frame_color]

        grid = np.zeros((rows, cols), dtype=int)
        # Fill the grid using colors that exclude the frame color
        grid = random_cell_coloring(grid, background_colors, density=0.7)

        # Draw subgrid perimeter with the chosen frame color (guaranteed unique around frame)
        grid[start_row, start_col:start_col + subcols] = frame_color
        grid[start_row + subrows - 1, start_col:start_col + subcols] = frame_color
        grid[start_row:start_row + subrows, start_col] = frame_color
        grid[start_row:start_row + subrows, start_col + subcols - 1] = frame_color

        # Ensure the interior contains at least 3 colored (non-zero) cells. If
        # not, fill some interior cells with random background colors until the
        # requirement is satisfied.
        interior_r0 = start_row + 1
        interior_r1 = start_row + subrows - 1
        interior_c0 = start_col + 1
        interior_c1 = start_col + subcols - 1
        # Only proceed if there is an interior (should be, since subrows/subcols >=4)
        if interior_r1 > interior_r0 and interior_c1 > interior_c0:
            interior = grid[interior_r0:interior_r1, interior_c0:interior_c1]
            nonzero = np.count_nonzero(interior)
            if nonzero < 3:
                # Get coordinates inside interior and shuffle
                coords = [(r, c) for r in range(interior_r0, interior_r1) for c in range(interior_c0, interior_c1)]
                random.shuffle(coords)
                for (r, c) in coords:
                    if np.count_nonzero(grid[interior_r0:interior_r1, interior_c0:interior_c1]) >= 3:
                        break
                    if grid[r, c] == 0:
                        grid[r, c] = random.choice(background_colors)

        return grid

    def transform_input(self, grid, taskvars):
        # We no longer rely on taskvars for subgrid size or color. Search the grid
        # for any rectangular perimeter whose border is a single non-zero color and
        # return its interior.
        rows, cols = grid.shape
        # Iterate over all possible top-left corners
        for i in range(rows):
            for j in range(cols):
                # Skip if starting cell is empty (perimeter must be non-zero color)
                c0 = grid[i, j]
                if c0 == 0:
                    continue
                # Try all possible sizes from 3x3 upwards that fit in the grid
                max_sr = rows - i
                max_sc = cols - j
                for sr in range(3, max_sr + 1):
                    for sc in range(3, max_sc + 1):
                        # Check top and bottom edges and left/right edges for uniform color
                        if not all(grid[i, j:j+sc] == c0):
                            continue
                        if not all(grid[i+sr-1, j:j+sc] == c0):
                            continue
                        if not all(grid[i:i+sr, j] == c0):
                            continue
                        if not all(grid[i:i+sr, j+sc-1] == c0):
                            continue
                        # Found a rectangular perimeter with a single color; return interior
                        return grid[i+1:i+sr-1, j+1:j+sc-1]

        return None

    def create_grids(self):
        # Pick a single grid size (rows, cols) for all examples so the input
        # reasoning statement that references {vars['rows']} and {vars['cols']}
        # is accurate.
        rows = random.randint(8, 25)
        cols = random.randint(8, 25)
        # Only include rows and cols in taskvars. All other choices (subgrid size,
        # frame color) vary per grid and are chosen locally inside create_input.
        taskvars = {'rows': rows, 'cols': cols}

        train_data = []
        for _ in range(random.randint(3, 4)):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})

        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_data, 'test': test_data}

