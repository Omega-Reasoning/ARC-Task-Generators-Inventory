from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry, Contiguity, random_cell_coloring
from transformation_library import find_connected_objects
import numpy as np
import random


class Task8e5a5113Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']} x {(vars['rows']*3)+2}.",
            "Each input grid is divided into three equal parts by completely filling columns {vars['rows']+1} and {(vars['rows']*2)+2} with {color('col_colour')} color.",
            "This results in three subgrids of size {vars['rows']} x {vars['rows']}, separated by {color('col_colour')} columns.",
            "The first {vars['rows']} x {vars['rows']} subgrid, on the left side of the grid, is completely filled with 4-way connected objects, with no empty cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the {color('col_colour')} dividers and the {vars['rows']} x {vars['rows']} subgrids separated by them.",
            "The first {vars['rows']} x {vars['rows']} subgrid is completely filled.",
            "The output grid is created by rotating the first subgrid 90 degree and 180 degree clockwise and placing it in the empty middle and last empty subgrids respectively."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        col_colour = taskvars['col_colour']
        width = (rows * 3) + 2
        
        # Create the full grid with background color 0
        grid = np.zeros((rows, width), dtype=int)
        
        # Fill the divider columns with col_colour
        grid[:, rows] = col_colour  # First divider
        grid[:, (rows * 2) + 1] = col_colour  # Second divider
        
        # Create a completely filled first subgrid with 4-way connected objects
        first_subgrid = self._create_filled_subgrid(rows, taskvars)
        
        # Place the first subgrid in the leftmost section
        grid[:rows, :rows] = first_subgrid
        
        return grid
    
    def _create_filled_subgrid(self, size, taskvars):
        # Create a subgrid that is completely filled with no empty cells
        # Use exactly (size-1) different colors, avoiding background 0 and col_colour
        
        # Choose colors (excluding background 0 and col_colour)
        available_colors = [c for c in range(1, 10) if c != taskvars['col_colour']]
        
        # Use exactly (rows-1) colors as specified
        num_colors = size - 1
        if num_colors > len(available_colors):
            raise ValueError(f"Not enough colors available. Need {num_colors}, have {len(available_colors)}")
        
        colors = random.sample(available_colors, num_colors)
        
        # Start with the first color filling everything
        subgrid = np.full((size, size), colors[0], dtype=int)
        
        # Distribute other colors randomly across the grid
        if len(colors) > 1:
            # Calculate how many cells each color should get (roughly equal distribution)
            total_cells = size * size
            cells_per_color = total_cells // num_colors
            remaining_cells = total_cells % num_colors
            
            # Get all cell positions
            all_positions = [(r, c) for r in range(size) for c in range(size)]
            random.shuffle(all_positions)
            
            # Assign colors to positions
            pos_idx = 0
            for i, color in enumerate(colors):
                # Determine how many cells this color gets
                cells_for_this_color = cells_per_color
                if i < remaining_cells:
                    cells_for_this_color += 1
                
                # Assign this color to the positions
                for _ in range(cells_for_this_color):
                    r, c = all_positions[pos_idx]
                    subgrid[r, c] = color
                    pos_idx += 1
        
        return subgrid

    def transform_input(self, grid, taskvars):
        rows = taskvars['rows']
        output_grid = grid.copy()
        
        # Extract the first subgrid
        first_subgrid = grid[:rows, :rows]
        
        # Rotate 90 degrees clockwise (equivalent to -1 in np.rot90)
        rotated_90 = np.rot90(first_subgrid, k=-1)
        
        # Rotate 180 degrees clockwise (equivalent to -2 in np.rot90)  
        rotated_180 = np.rot90(first_subgrid, k=-2)
        
        # Place rotated versions in middle and last subgrids
        # Middle subgrid starts at column rows+1 (after first divider)
        middle_start = rows + 1
        output_grid[:rows, middle_start:middle_start + rows] = rotated_90
        
        # Last subgrid starts at column (rows*2)+2 (after second divider)
        last_start = (rows * 2) + 2
        output_grid[:rows, last_start:last_start + rows] = rotated_180
        
        return output_grid

    def create_grids(self):
        # Generate random task variables
        rows = random.choice([3, 4, 5, 6, 7, 8, 9])  # odd between 3-9 as specified
        col_colour = random.randint(1, 9)  # divider color between 1-9
        
        taskvars = {
            'rows': rows,
            'col_colour': col_colour
        }
        
        # Generate 3-5 training examples and 1 test example
        num_train = random.randint(3, 5)
        
        return taskvars, self.create_grids_default(num_train, 1, taskvars)


