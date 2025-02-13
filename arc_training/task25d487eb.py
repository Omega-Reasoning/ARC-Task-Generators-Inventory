import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import create_object, retry

class ARCTask25d487ebGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['cols']}.",
            "The input grid has a single 4-way connected object which is an isosceles triangle.",
            "The triangle is rotated anticlockwise by multiple of 90 degrees in some examples.",
            "The triangles has color input_color(between 1 and 9), except for the center cell of the last row in the triangle, which is of color cell_color(between 1 and 9).",
            "The triangle color varies between the examples.",
            "All remaining cells in the grid are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Identify the 4-way connected triangle object in the input grid.",
            "Determine the color of the center cell of the row/column with the highest number of cells.",
            "Extend the cell color along the triangles central axis, moving upward toward its peak rather than from the bottom, and continue the extension outward to the edge of the grid.",
            "While extending the color, the 4-way connected object cells do not change color, only the empty cells are colored.",
            "All remaining cells are empty (0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        rows, cols = taskvars['rows'], taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)
        rotation_angle = random.choice([0, 90, 180, 270])

        # Define triangle properties
        triangle_color = random.randint(1, 9)
        while triangle_color == taskvars['cell_color']:
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
        subgrid[0 , center] = taskvars['cell_color']
        print(subgrid)
        
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
        
        return grid
    
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
            'x': random.choice([90, 180, 270]),
            'cell_color': random.randint(1, 9)
        }
        
        train_examples = [
            {'input': (input_grid := self.create_input(taskvars, {})),
             'output': self.transform_input(input_grid, taskvars)}
            for _ in range(random.randint(3, 4))
        ]
        
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {'train': train_examples, 'test': [{'input': test_input, 'output': test_output}]}
    
