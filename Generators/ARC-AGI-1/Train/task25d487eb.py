import numpy as np
import random
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import create_object, retry

#
class Task25d487ebGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid size varies across examples.",
            "The input grid contains a single 4-way connected object shaped like an isosceles triangle.",
            "The size of the triangle varies across examples.",
            "In some examples, the triangle is rotated anticlockwise by a multiple of 90 degrees.",
            "The triangle is mostly filled with one color, except for a single cell located at the center of the triangle’s base, which is a different color.",
            "The main triangle color varies across examples.",
            "The single different-colored cell also varies across examples.",
            "All remaining cells in the grid are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Identify the single 4-way connected isosceles triangle in the input grid.",
            "Locate the uniquely colored cell positioned at the center of the triangle base.",
            "Determine the central axis of the triangle based on its orientation.",
            "Extend the color of the uniquely colored cell along the central axis of the triangle, moving from the base toward the peak of the triangle and continuing in that direction until the edge of the grid is reached.",
            "During this extension, only empty cells are filled with the extending color; the original triangle cells remain unchanged.",
            "All remaining cells in the grid are empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        cell_color = gridvars['cell_color']
        
        def generate_valid_grid():
            grid = np.zeros((rows, cols), dtype=int)
            # Allow explicit rotation override via gridvars for deterministic examples
            rotation_angle = gridvars.get('rotation_angle', random.choice([0, 90, 180, 270]))

            # Define triangle properties
            triangle_color = random.randint(1, 9)
            while triangle_color == cell_color:
                triangle_color = random.randint(1, 9)
            
            height = random.randint(2, rows // 3)
            base_width = 2 * height - 1
            
            # Create a subgrid that can contain the triangle
            subgrid_size = max(height, base_width)
            if subgrid_size % 2 == 0:  # Ensure odd size for proper rotation center
                subgrid_size += 1
            subgrid = np.zeros((subgrid_size, subgrid_size), dtype=int)
            
            # Calculate center of subgrid
            center = subgrid_size // 2
            
            # Draw triangle in subgrid
            for i in range(height):
                for j in range(-i, i + 1):
                    row = center - i
                    col = center + j
                    subgrid[row, col] = triangle_color
            
            # Set the center cell color
            subgrid[center - height + 1, center] = cell_color
            
            # Rotate subgrid if needed
            if rotation_angle == 90:
                subgrid = np.rot90(subgrid, k=3)
            elif rotation_angle == 180:
                subgrid = np.rot90(subgrid, k=2)
            elif rotation_angle == 270:
                subgrid = np.rot90(subgrid, k=1)
            
            # Calculate valid placement ranges for the rotated subgrid
            min_row = subgrid_size // 2
            max_row = rows - (subgrid_size // 2) - 1
            min_col = subgrid_size // 2
            max_col = cols - (subgrid_size // 2) - 1
            
            # Place the rotated subgrid in the main grid
            if max_row >= min_row and max_col >= min_col:
                place_row = random.randint(min_row, max_row)
                place_col = random.randint(min_col, max_col)
                
                # Copy the rotated subgrid to the main grid
                start_row = place_row - subgrid_size // 2
                start_col = place_col - subgrid_size // 2
                grid[start_row:start_row + subgrid_size, 
                    start_col:start_col + subgrid_size] = subgrid
            
            if np.any(grid != 0):
                return grid
            else:
                return None

        return retry(generate_valid_grid, lambda x: x is not None, max_attempts=100)
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        
        # Identify the triangle object
        objects = find_connected_objects(grid, diagonal_connectivity=False)
        triangle = max(objects, key=lambda obj: obj.size)
        
        # Count cells in each row and column
        row_counts = {}
        col_counts = {}
        for r, c in triangle.coords:
            row_counts[r] = row_counts.get(r, 0) + 1
            col_counts[c] = col_counts.get(c, 0) + 1
        
        # Find the maximum length for rows and columns
        max_row_count = max(row_counts.values())
        max_col_count = max(col_counts.values())
        
        # Determine whether to extend horizontally or vertically based on which has more cells
        if max_row_count >= max_col_count:
            # Extend horizontally - find the row with most cells
            max_row = max(row_counts.items(), key=lambda x: x[1])[0]
            max_row_cells = [c for r, c in triangle.coords if r == max_row]
            center_col = min(max_row_cells) + (max(max_row_cells) - min(max_row_cells)) // 2
            center_color = grid[max_row, center_col]
            
            # Check neighboring cells to determine direction of extension
            if max_row == 0:  # If at top edge, can only extend downward
                for r in range(max_row, grid.shape[0]):  # Extend downward
                    if output_grid[r, center_col] == 0:
                        output_grid[r, center_col] = center_color
            elif max_row == grid.shape[0] - 1:  # If at bottom edge, can only extend upward
                for r in range(max_row, -1, -1):  # Extend upward
                    if output_grid[r, center_col] == 0:
                        output_grid[r, center_col] = center_color
            else:  # Check both directions when not at edges
                if grid[max_row + 1, center_col] != 0:  # Check cell below
                    for r in range(max_row, grid.shape[0]):  # Extend downward
                        if output_grid[r, center_col] == 0:
                            output_grid[r, center_col] = center_color
                elif grid[max_row - 1, center_col] != 0:  # Check cell above
                    for r in range(max_row, -1, -1):  # Extend upward
                        if output_grid[r, center_col] == 0:
                            output_grid[r, center_col] = center_color
        else:
            # Extend vertically - find the column with most cells
            max_col = max(col_counts.items(), key=lambda x: x[1])[0]
            max_col_cells = [r for r, c in triangle.coords if c == max_col]
            center_row = min(max_col_cells) + (max(max_col_cells) - min(max_col_cells)) // 2
            center_color = grid[center_row, max_col]

            if max_col == 0:  # If at left edge, can only extend right
                for c in range(max_col, grid.shape[1]):
                    if output_grid[center_row, c] == 0:
                        output_grid[center_row, c] = center_color
            elif max_col == grid.shape[1] - 1:  # If at right edge, can only extend left
                for c in range(max_col, -1, -1):
                    if output_grid[center_row, c] == 0:
                        output_grid[center_row, c] = center_color
            else:  # Check both directions when not at edges
                if grid[center_row, max_col + 1] != 0:  # Check cell to the right
                    for c in range(max_col, grid.shape[1]):
                        if output_grid[center_row, c] == 0:
                            output_grid[center_row, c] = center_color
                elif grid[center_row, max_col - 1] != 0:  # Check cell to the left
                    for c in range(max_col, -1, -1):
                        if output_grid[center_row, c] == 0:
                            output_grid[center_row, c] = center_color
        
        return output_grid
    
    def create_grids(self):
        taskvars = {
            'rows': random.randint(10, 30),
            'cols': random.randint(10, 30),
            'x': random.choice([90, 180, 270])
        }

        # Ensure at least one train example with triangle rotated 90 (horizontal extension)
        # and one with rotation 0 (vertical extension). Fill remaining examples randomly.
        create_count = random.randint(3, 4)
        train_examples = []

        # Guaranteed examples for required rotations
        for rot in (90, 0):
            gridvars = {'cell_color': random.randint(1, 9), 'rotation_angle': rot}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        # Remaining examples (random rotations)
        for _ in range(max(0, create_count - 2)):
            gridvars = {'cell_color': random.randint(1, 9)}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        test_gridvars = {'cell_color': random.randint(1, 9)}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)

        return taskvars, {'train': train_examples, 'test': [{'input': test_input, 'output': test_output}]}