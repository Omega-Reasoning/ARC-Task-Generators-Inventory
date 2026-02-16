from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import random_cell_coloring, retry
from Framework.transformation_library import find_connected_objects, GridObjects

class Task5bd6f4acGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain multi-colored (1–9) and empty (0) cells.",
            "The multi-colored cells are randomly distributed across the grid, appearing in all quadrants without clustering in any specific area.",
            "Some of the multi-colored cells may be 4-way connected to each other."
        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {vars['grid_size']//3}x{vars['grid_size']//3}.",
            "They are constructed by copying the top-right {vars['grid_size']//3}x{vars['grid_size']//3} subgrid from the input grid and pasting it into the output grid.",
            "The colors and relative positions of the cell in the top-right {vars['grid_size']//3}x{vars['grid_size']//3} subgrid remain unchanged in the output."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Choose a random grid size from the allowed values
        grid_size_options = [6, 9, 12, 15, 18, 21, 24, 27, 30]
        taskvars = {'grid_size': random.choice(grid_size_options)}
        
        # Generate 3-4 training examples and 1 test example
        num_train = random.randint(3, 4)
        
        return taskvars, self.create_grids_default(num_train, 1, taskvars)
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        subgrid_size = grid_size // 3
        
        # Start with an empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Define a predicate to ensure colors exist in all quadrants of the grid
        def has_colors_in_all_quadrants(g):
            # Divide grid into quadrants (accounting for odd grid sizes)
            mid_row = grid_size // 2
            mid_col = grid_size // 2
            
            # Check if each quadrant has at least some colored cells
            top_left = np.any(g[:mid_row, :mid_col] > 0)
            top_right = np.any(g[:mid_row, mid_col:] > 0)
            bottom_left = np.any(g[mid_row:, :mid_col] > 0)
            bottom_right = np.any(g[mid_row:, mid_col:] > 0)
            
            return top_left and top_right and bottom_left and bottom_right
        
        # Define a predicate to ensure there are some 4-way connected cells
        def has_connected_cells(g):
            objects = find_connected_objects(g, diagonal_connectivity=False, monochromatic=True)
            # Check if at least one object has more than one cell
            return any(len(obj) > 1 for obj in objects.objects)
        
        # Define a predicate to ensure top-right subgrid has minimum colored cells
        def has_enough_cells_in_top_right(g):
            top_right = g[:subgrid_size, -subgrid_size:]
            return np.sum(top_right > 0) >= subgrid_size
        
        # Generate the grid with random colored cells
        def generate_colored_grid():
            # Clear the grid
            g = np.zeros((grid_size, grid_size), dtype=int)
            
            # Fill with random colors (1-9) with moderate density
            colors = list(range(1, 10))
            grid_with_colors = random_cell_coloring(g, colors, density=0.3)
            
            # Ensure top-right subgrid has enough colored cells
            top_right = grid_with_colors[:subgrid_size, -subgrid_size:]
            if np.sum(top_right > 0) < subgrid_size:
                # Add more colored cells specifically to top-right if needed
                empty_cells = np.where(top_right == 0)
                cells_needed = max(0, subgrid_size - np.sum(top_right > 0))
                
                if len(empty_cells[0]) >= cells_needed:
                    # Randomly select empty cells to color
                    indices = np.random.choice(len(empty_cells[0]), 
                                               size=int(cells_needed), 
                                               replace=False)
                    for i in indices:
                        r, c = empty_cells[0][i], empty_cells[1][i]
                        grid_with_colors[r, c - grid_size + subgrid_size] = random.randint(1, 9)
            
            return grid_with_colors
        
        # Retry until we get a grid that meets all our requirements
        return retry(
            generate_colored_grid,
            lambda g: has_colors_in_all_quadrants(g) and 
                     has_connected_cells(g) and 
                     has_enough_cells_in_top_right(g)
        )
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        subgrid_size = grid_size // 3
        
        # Extract the top-right subgrid
        top_right_subgrid = grid[:subgrid_size, -subgrid_size:]
        
        # Create output grid of size subgrid_size x subgrid_size
        output_grid = top_right_subgrid.copy()
        
        return output_grid

