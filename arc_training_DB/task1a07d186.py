from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import find_connected_objects, GridObject
import numpy as np
import random

class ARCAGITaskGenerator(ARCTaskGenerator):

    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "The input grid has either a maximum of 3 rows or 3 columns filled with colors(1-9).",
            "Each filled row or column has the same color.",
            "The color of rows or columns are different from each other.",
            "Few cells are colored(1-9) randomly and the other cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "The cells with the same color as row or column are translated such that they touch their respective row or column of the same color.",
            "The cells which have different colors than the colored rows or columns maintain their position in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)

        # Randomly select 3 rows or columns to fill with minimum spacing
        fill_type = random.choice(['row', 'col'])
        num_to_fill = random.randint(1, 3)
        max_dim = rows if fill_type == 'row' else cols
        min_spacing = 2  # Minimum spacing between filled rows/columns

        # Generate spaced indices, excluding borders
        available_indices = list(range(1, max_dim - 1))  # Exclude 0 and max_dim-1
        filled_indices = []
        for _ in range(num_to_fill):
            if not available_indices:
                break
            index = random.choice(available_indices)
            filled_indices.append(index)
            # Remove nearby indices to maintain spacing
            for offset in range(-min_spacing, min_spacing + 1):
                if index + offset in available_indices:
                    available_indices.remove(index + offset)

        # Sort indices to help with placement constraints
        filled_indices.sort()
        colors = random.sample(range(1, 10), len(filled_indices))

        # Fill the rows/columns
        for index, color in zip(filled_indices, colors):
            if fill_type == 'row':
                grid[index, :] = color
            else:
                grid[:, index] = color

        # Initialize color_counts with all possible colors (1-9)
        color_counts = {i: 0 for i in range(1, 10)}

        # Ensure at least one cell of each color is present in a different row/column
        for i, (index, color) in enumerate(zip(filled_indices, colors)):
            while True:
                if fill_type == 'row':
                    r = random.choice([i for i in range(rows) if i != index])
                    c = random.randint(0, cols - 1)
                else:
                    r = random.randint(0, rows - 1)
                    c = random.choice([i for i in range(cols) if i != index])
                
                if grid[r, c] == 0:  # Only place if cell is empty
                    grid[r, c] = color
                    color_counts[color] += 1
                    break

        # Add additional random cells with colors, maintaining the constraints
        random_fill_density = random.uniform(0.02, 0.1)
        num_random_cells = int(random_fill_density * rows * cols)
        
        # Add some different colors not used in rows/columns
        additional_colors = [c for c in range(1, 10) if c not in colors]
        all_possible_colors = colors + additional_colors

        for _ in range(num_random_cells):
            attempts = 0
            while attempts < 100:  # Limit attempts to prevent infinite loops
                r, c = random.randint(0, rows - 1), random.randint(0, cols - 1)
                color = random.choice(all_possible_colors)
                
                # Check if position is valid
                if grid[r, c] != 0:
                    attempts += 1
                    continue
                
                # For row-colored grids
                if fill_type == 'row':
                    row_idx = next((idx for idx in filled_indices if color == grid[idx, 0]), None)
                    if row_idx is not None:
                        # Check if there's already a cell with this color above or below the row
                        cells_above = [grid[i, c] for i in range(row_idx)]
                        cells_below = [grid[i, c] for i in range(row_idx + 1, rows)]
                        if (color in cells_above and r < row_idx) or (color in cells_below and r > row_idx):
                            attempts += 1
                            continue
                
                # For column-colored grids
                else:
                    col_idx = next((idx for idx in filled_indices if color == grid[0, idx]), None)
                    if col_idx is not None:
                        # Check if there's already a cell with this color to the left or right of the column
                        cells_left = [grid[r, i] for i in range(col_idx)]
                        cells_right = [grid[r, i] for i in range(col_idx + 1, cols)]
                        if (color in cells_left and c < col_idx) or (color in cells_right and c > col_idx):
                            attempts += 1
                            continue
                
                grid[r, c] = color
                color_counts[color] += 1
                break

        return grid

    def transform_input(self, grid, taskvars):
        rows, cols = grid.shape
        output_grid = grid.copy()

        # Find colored rows and columns
        colored_rows = {r: grid[r, 0] for r in range(rows) if len(set(grid[r, :]) - {0}) == 1}
        colored_cols = {c: grid[0, c] for c in range(cols) if len(set(grid[:, c]) - {0}) == 1}

        # Process each cell individually
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] != 0:
                    self._translate_item([(r, c, grid[r, c])], {grid[r, c]}, colored_rows, colored_cols, output_grid)

        return output_grid

    def _translate_item(self, cells, colors, colored_rows, colored_cols, output_grid):
        # We only need the first cell since we're dealing with individual cells
        r, c, color = cells[0]

        # Check if color matches any colored row
        if color in colored_rows.values():
            row_idx = next(r_idx for r_idx, col in colored_rows.items() if col == color)
            
            # Skip if cell is part of the colored row itself
            if r == row_idx:
                return
            
            # Clear original position
            output_grid[r, c] = 0
            
            # Move cell up or down to touch the row
            new_r = row_idx - 1 if r < row_idx else row_idx + 1
            if 0 <= new_r < output_grid.shape[0]:
                output_grid[new_r, c] = color

        # Check if color matches any colored column
        elif color in colored_cols.values():
            col_idx = next(c_idx for c_idx, col in colored_cols.items() if col == color)
            
            # Skip if cell is part of the colored column itself
            if c == col_idx:
                return
            
            # Clear original position
            output_grid[r, c] = 0
            
            # Move cell left or right to touch the column
            new_c = col_idx - 1 if c < col_idx else col_idx + 1
            if 0 <= new_c < output_grid.shape[1]:
                output_grid[r, new_c] = color

    def create_grids(self):
        taskvars = {
            'rows': random.randint(15, 26),
            'cols': random.randint(15, 26)
        }

        num_train = random.randint(3, 4)
        num_test = 1

        train_test_data = self.create_grids_default(num_train, num_test, taskvars)

        return taskvars, train_test_data

