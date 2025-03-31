from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple
from input_library import random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task5daaa586Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['cols']}.",
            "They contain two completely filled rows and two completely filled columns, each in a different color, forming a subgrid made up of four intersecting colored lines.",
            "The filled rows and columns should always be separated by at least one empty row and one empty column, respectively.",
            "The vertical colored lines always take priority over horizontal lines wherever they intersect.",
            "Additionally, there are several randomly scattered colored cells in the grid.",
            "The color of these scattered cells is always the same and matches one of the four line colors.",
            "All other cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are always smaller than the input grids.",
            "They are constructed by identifying the four intersecting colored lines made by two completely filled rows and two completely filled columns.",
            "Once identified, only the subgrid enclosed by the four lines is copied to the output, including the lines themselves.",
            "Next, identify the non-border colored cells in the output grid and fill in all empty (0) cells between them and the corresponding same-colored line (row or column), effectively connecting the cells to its matching line.",
            "By filling in the empty (0) cells several vertical or horizontal lines are formed."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'rows': random.randint(10, 20),
            'cols': random.randint(10, 20)
        }
        
        # Generate 3-6 train examples
        num_train_examples = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate 1 test example
        test_gridvars = {}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Choose 4 different colors for the lines
        available_colors = list(range(1, 10))  # Colors 1-9
        line_colors = random.sample(available_colors, 4)
        
        # To ensure the subgrid covers more than 1/3 of the grid area, 
        # we need to carefully choose the line positions
        
        # Calculate minimum subgrid size to cover more than 1/3 of grid area
        min_area = (rows * cols) / 3
        
        # Function to get subgrid area from row and column indices
        def get_subgrid_area(r1, r2, c1, c2):
            return (r2 - r1 + 1) * (c2 - c1 + 1)
        
        # Keep trying row and column positions until we get a suitable subgrid size
        valid_positions = False
        attempts = 0
        
        while not valid_positions and attempts < 100:
            attempts += 1
            
            # For rows: divide grid height into 3 parts and place lines in first and last part
            r1_range = range(1, rows // 3)
            r2_range = range(2 * rows // 3, rows - 1)
            
            if not r1_range or not r2_range:
                continue  # Grid too small, try again
                
            row_indices = [random.choice(r1_range), random.choice(r2_range)]
            
            # For columns: similar approach with width
            c1_range = range(1, cols // 3)
            c2_range = range(2 * cols // 3, cols - 1)
            
            if not c1_range or not c2_range:
                continue  # Grid too small, try again
                
            col_indices = [random.choice(c1_range), random.choice(c2_range)]
            
            # Calculate subgrid area
            subgrid_area = get_subgrid_area(row_indices[0], row_indices[1], 
                                           col_indices[0], col_indices[1])
            
            # Check if subgrid is large enough and has proper spacing
            if (subgrid_area > min_area and 
                row_indices[1] - row_indices[0] > 2 and 
                col_indices[1] - col_indices[0] > 2):
                valid_positions = True
        
        if not valid_positions:
            # Fallback strategy if we couldn't find optimal positions
            # Place rows at 1/4 and 3/4 of grid height
            row_indices = [rows // 4, 3 * rows // 4]
            # Place columns at 1/4 and 3/4 of grid width
            col_indices = [cols // 4, 3 * cols // 4]
        
        # Draw horizontal lines
        grid[row_indices[0], :] = line_colors[0]
        grid[row_indices[1], :] = line_colors[1]
        
        # Draw vertical lines (with priority over horizontal)
        grid[:, col_indices[0]] = line_colors[2]
        grid[:, col_indices[1]] = line_colors[3]
        
        # Store indices and colors for later use
        gridvars['row_indices'] = row_indices
        gridvars['col_indices'] = col_indices
        gridvars['line_colors'] = line_colors
        
        # Calculate and store subgrid area percentage
        subgrid_area = get_subgrid_area(row_indices[0], row_indices[1], 
                                       col_indices[0], col_indices[1])
        grid_area = rows * cols
        area_percentage = (subgrid_area / grid_area) * 100
        gridvars['subgrid_percentage'] = area_percentage
        
        # Pick one of the line colors for scattered cells
        scatter_color = random.choice(line_colors)
        
        # Add randomly scattered cells across the entire grid
        # Number of scattered cells (between 5 and 15 for larger grids)
        num_cells = random.randint(5, 15)
        
        # Place scattered cells randomly in the grid
        subgrid_cells = 0  # Counter to ensure we have cells inside the subgrid
        
        for _ in range(num_cells):
            # Generate a random position
            r = random.randint(0, rows-1)
            c = random.randint(0, cols-1)
            
            # Skip if on a line
            if r in row_indices or c in col_indices:
                continue
                
            # Check if this is inside the subgrid
            is_in_subgrid = (row_indices[0] < r < row_indices[1]) and (col_indices[0] < c < col_indices[1])
            if is_in_subgrid:
                subgrid_cells += 1
                
            grid[r, c] = scatter_color
        
        # Ensure at least one point is inside the subgrid (needed for our transformation)
        if subgrid_cells == 0:
            r = random.randint(row_indices[0]+1, row_indices[1]-1)
            c = random.randint(col_indices[0]+1, col_indices[1]-1)
            grid[r, c] = scatter_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find the four colored lines
        rows, cols = grid.shape
        
        # Identify rows and columns that are fully filled
        filled_row_indices = []
        filled_row_colors = []
        filled_col_indices = []
        filled_col_colors = []
        
        for r in range(rows):
            if np.all(grid[r, :] > 0):
                # Get the color (assuming mostly one color in the row)
                color_counts = np.bincount(grid[r, :], minlength=10)
                most_common_color = np.argmax(color_counts[1:]) + 1
                filled_row_indices.append(r)
                filled_row_colors.append(most_common_color)
        
        for c in range(cols):
            if np.all(grid[:, c] > 0):
                # Get the color (assuming mostly one color in the column)
                color_counts = np.bincount(grid[:, c], minlength=10)
                most_common_color = np.argmax(color_counts[1:]) + 1
                filled_col_indices.append(c)
                filled_col_colors.append(most_common_color)
        
        # Sort the indices to get the boundary of the subgrid
        row_indices = sorted(filled_row_indices)[:2]
        col_indices = sorted(filled_col_indices)[:2]
        
        # Create a new output grid of the exact size of the subgrid
        subgrid_height = row_indices[1] - row_indices[0] + 1
        subgrid_width = col_indices[1] - col_indices[0] + 1
        output_grid = np.zeros((subgrid_height, subgrid_width), dtype=int)
        
        # Copy only the subgrid from the input to the output
        for i in range(subgrid_height):
            for j in range(subgrid_width):
                input_r = row_indices[0] + i
                input_c = col_indices[0] + j
                output_grid[i, j] = grid[input_r, input_c]
        
        # Now connect any scattered cells within the subgrid to their matching colored lines
        # We need to adjust indices since we're working with the smaller output grid
        
        # Map original row/col indices to the new grid indices
        new_row_indices = [0, subgrid_height-1]  # First and last rows
        new_col_indices = [0, subgrid_width-1]   # First and last columns
        
        # Connect scattered cells to their matching lines
        for r in range(1, subgrid_height-1):  # Skip border rows
            for c in range(1, subgrid_width-1):  # Skip border columns
                if output_grid[r, c] > 0:  # If cell is colored
                    cell_color = output_grid[r, c]
                    
                    # Connect to horizontal lines with matching color
                    for idx, new_idx in enumerate(new_row_indices):
                        if output_grid[new_idx, c] == cell_color:
                            # Connect to this line
                            r_start, r_end = min(r, new_idx), max(r, new_idx)
                            output_grid[r_start:r_end+1, c] = cell_color
                    
                    # Connect to vertical lines with matching color
                    for idx, new_idx in enumerate(new_col_indices):
                        if output_grid[r, new_idx] == cell_color:
                            # Connect to this line
                            c_start, c_end = min(c, new_idx), max(c, new_idx)
                            output_grid[r, c_start:c_end+1] = cell_color
        
        return output_grid
