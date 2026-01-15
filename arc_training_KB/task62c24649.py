from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task62c24649Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain colored objects, where each object is made of 4-way connected cells of the same color.",
            "The object colors are {color('color1')}, {color('color2')}, and {color('color3')}.",
            "The colored objects are packed closely together, filling more than three-fourths of the grid cells.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {2*vars['grid_size']} x {2*vars['grid_size']}.",
            "They are constructed by copying the input grid to the top-left (first) quadrant of the output grid and reflecting it horizontally into the top-right (second) quadrant.",
            "Then, the entire top half of the output grid is reflected vertically downwards to fill the bottom half."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple:
        """Initialize task variables and create train/test data grids."""
        
        taskvars = {
            'grid_size': random.randint(3, 15),  # Random grid size between 3 and 7
            'color1': random.randint(1, 9),  # Random colors between 1 and 9
            'color2': 0,
            'color3': 0
        }
        
        # Ensure colors are different
        while taskvars['color2'] == 0 or taskvars['color2'] == taskvars['color1']:
            taskvars['color2'] = random.randint(1, 9)
        
        while taskvars['color3'] == 0 or taskvars['color3'] == taskvars['color1'] or taskvars['color3'] == taskvars['color2']:
            taskvars['color3'] = random.randint(1, 9)
        
        # Create train and test examples
        num_train_examples = 4
        train_test_data = self.create_grids_default(num_train_examples, 1, taskvars)
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars, gridvars):
        """Create an input grid based on the task variables."""
        grid_size = taskvars['grid_size']
        colors = [taskvars['color1'], taskvars['color2'], taskvars['color3']]
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Calculate number of cells to fill (more than 3/4 of the grid)
        min_cells_to_fill = int((grid_size * grid_size) * 0.75) + 1
        
        # Create at least 3 objects of different colors
        remaining_cells = grid_size * grid_size
        objects_created = 0
        
        while objects_created < 3 or remaining_cells > grid_size * grid_size - min_cells_to_fill:
            # Choose a random color from our color palette
            color = random.choice(colors)
            
            # Create a small object with 4-way connectivity
            object_height = random.randint(1, max(1, grid_size // 2))
            object_width = random.randint(1, max(1, grid_size // 2))
            
            object_matrix = create_object(
                object_height, 
                object_width, 
                color, 
                contiguity=Contiguity.FOUR, 
                background=0
            )
            
            # Find a position to place the object
            max_attempts = 100
            for _ in range(max_attempts):
                row = random.randint(0, grid_size - object_height)
                col = random.randint(0, grid_size - object_width)
                
                # Check if position is valid (has some overlap with existing objects or touches grid border)
                valid_position = False
                for r in range(row, min(row + object_height, grid_size)):
                    for c in range(col, min(col + object_width, grid_size)):
                        if object_matrix[r - row, c - col] != 0:
                            # Check if this would overlap with existing cells
                            if grid[r, c] != 0:
                                break
                            
                            # Check if it touches another object
                            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr, nc] != 0:
                                    valid_position = True
                                    break
                            
                            # Also valid if it touches the border
                            if r == 0 or r == grid_size - 1 or c == 0 or c == grid_size - 1:
                                valid_position = True
                            
                            if valid_position:
                                break
                    if valid_position:
                        break
                
                if valid_position or objects_created == 0:
                    # Place the object
                    object_cells = 0
                    for r in range(object_height):
                        for c in range(object_width):
                            if object_matrix[r, c] != 0 and row + r < grid_size and col + c < grid_size:
                                if grid[row + r, col + c] == 0:  # Only place if cell is empty
                                    grid[row + r, col + c] = object_matrix[r, c]
                                    object_cells += 1
                    
                    if object_cells > 0:
                        objects_created += 1
                        remaining_cells -= object_cells
                        break
        
        # Verify we have enough filled cells and at least 3 objects
        filled_cells = np.sum(grid != 0)
        if filled_cells < min_cells_to_fill or objects_created < 3:
            # If not enough, add random colors to empty cells
            empty_coords = np.where(grid == 0)
            indices = list(zip(empty_coords[0], empty_coords[1]))
            random.shuffle(indices)
            
            for i, (r, c) in enumerate(indices):
                if filled_cells >= min_cells_to_fill and objects_created >= 3:
                    break
                    
                # Check if adding this cell creates a new object or extends an existing one
                color = random.choice(colors)
                grid[r, c] = color
                filled_cells += 1
                
                # Check object count after adding cell
                objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
                objects_created = len(objects)
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """Transform the input grid according to the transformation reasoning chain."""
        grid_size = taskvars['grid_size']
        output_size = 2 * grid_size
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Step 1: Copy input grid to top-left quadrant
        output_grid[:grid_size, :grid_size] = grid
        
        # Step 2: Reflect horizontally to top-right quadrant
        for r in range(grid_size):
            for c in range(grid_size):
                output_grid[r, output_size - 1 - c] = grid[r, c]
        
        # Step 3: Reflect vertically to fill bottom half
        for r in range(grid_size):
            for c in range(output_size):
                output_grid[output_size - 1 - r, c] = output_grid[r, c]
        
        return output_grid

