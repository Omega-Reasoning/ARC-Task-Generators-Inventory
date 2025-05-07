from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class Task0692e18cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each grid contains a single object composed of either {vars['grid_size']} or {vars['grid_size']+1} 8-way connected, same-colored cells; all remaining cells are empty (0).",
            "The shape of the object varies across examples but must ensure that all its cells are 8-way connected, with at least one cell present in every row and every column.",
            "The color of the object varies across examples."
        ]

        transformation_reasoning_chain = [
            "Output grids are of size {vars['grid_size'] * vars['grid_size']}x {vars['grid_size'] * vars['grid_size']}.",
            "The output grid is divided into subgrids of size {vars['grid_size']}x{vars['grid_size']}, forming a grid of subgrids.",
            "For each cell in the input grid; if the cell is empty (0), the corresponding subgrid in the output remains completely empty.",
            "Otherwise if the cell is filled, the corresponding subgrid in the output is filled with a specific fixed pattern.",
            "To understand the mapping: each cell in the input grid corresponds to a {vars['grid_size']}x{vars['grid_size']} subgrid in the output grid. For example, cell (0, 0) in the input maps to the top-left {vars['grid_size']}x{vars['grid_size']} subgrid in the output.",
            "When a cell in the input is filled, the corresponding output subgrid is filled using a transformed version of the original input grid: all originally filled cells are emptied (0), and all originally empty cells are filled with the object color.",
            "This transformed {vars['grid_size']}x{vars['grid_size']} grid is used as a pattern to fill each corresponding subgrid.",
            "The result is that the overall structure in the output grid resembles the input grid but with inverted fill patterns."
        ]

        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        # Define task variables
        grid_size = random.choice([2, 3, 4, 5])
        taskvars = {'grid_size': grid_size}
        
        # Generate 3-4 train examples and 1 test example
        train_count = random.randint(3, 4)
        
        # Ensure we use a different color for each example (including test)
        colors = random.sample(range(1, 10), train_count + 1)
        
        train_pairs = []
        for i in range(train_count):
            color = colors[i]
            input_grid = self.create_input(taskvars, {'color': color})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Generate test pair
        test_color = colors[-1]
        test_input = self.create_input(taskvars, {'color': test_color})
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_pairs,
            'test': [{'input': test_input, 'output': test_output}]
        }

    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        color = gridvars['color']
        
        # Choose random size for the object: either grid_size or grid_size+1
        object_size = random.choice([grid_size, grid_size + 1])
        
        # Use a different approach to generate the input grid
        max_attempts = 100
        for _ in range(max_attempts):
            # Start with a grid where every cell has a 50% chance of being filled
            grid = np.zeros((grid_size, grid_size), dtype=int)
            for r in range(grid_size):
                for c in range(grid_size):
                    if random.random() < 0.5:
                        grid[r, c] = color
            
            # Check if every row and column has at least one filled cell
            if not (np.all(np.any(grid > 0, axis=1)) and np.all(np.any(grid > 0, axis=0))):
                continue  # Skip this grid if constraint not met
            
            # Find connected components
            objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
            
            # If there's exactly one connected component with the right size, return it
            if len(objects) == 1 and len(objects[0]) == object_size:
                return grid
            
            # If there's exactly one component but wrong size, try to adjust it
            if len(objects) == 1:
                obj = objects[0]
                curr_size = len(obj)
                
                if curr_size > object_size:
                    # Remove cells while preserving connectivity and row/column constraint
                    cells_to_remove = curr_size - object_size
                    for _ in range(cells_to_remove):
                        # Find cells that can be removed without breaking constraints
                        removable = []
                        for r, c, _ in obj:
                            # Temporarily remove this cell
                            original = grid[r, c]
                            grid[r, c] = 0
                            
                            # Check row/column constraint
                            row_has_cell = np.any(grid[r, :] > 0)
                            col_has_cell = np.any(grid[:, c] > 0)
                            
                            # Check connectivity
                            temp_objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
                            still_connected = len(temp_objects) <= 1
                            
                            # Restore cell
                            grid[r, c] = original
                            
                            if row_has_cell and col_has_cell and still_connected:
                                removable.append((r, c))
                        
                        if not removable:
                            break  # Can't remove more cells
                        
                        # Remove a random cell
                        r, c = random.choice(removable)
                        grid[r, c] = 0
                
                elif curr_size < object_size:
                    # Add cells while preserving connectivity
                    cells_to_add = object_size - curr_size
                    for _ in range(cells_to_add):
                        # Find empty cells adjacent to the object
                        addable = []
                        for r in range(grid_size):
                            for c in range(grid_size):
                                if grid[r, c] == 0:
                                    # Check if adjacent to existing object
                                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                                        nr, nc = r + dr, c + dc
                                        if 0 <= nr < grid_size and 0 <= nc < grid_size and grid[nr, nc] > 0:
                                            addable.append((r, c))
                                            break
                        
                        if not addable:
                            break  # Can't add more cells
                        
                        # Add a random cell
                        r, c = random.choice(addable)
                        grid[r, c] = color
                
                # Check if the adjustment worked
                objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
                if len(objects) == 1 and len(objects[0]) == object_size:
                    return grid
        
        # If we couldn't generate a valid grid, create a simple pattern that meets criteria
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create a simple snake pattern
        for i in range(object_size):
            r = i // grid_size
            c = i % grid_size
            grid[r, c] = color
        
        return grid

    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        output_size = grid_size * grid_size
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Get the color of the object in the input grid
        object_color = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] > 0:
                    object_color = grid[r, c]
                    break
            if object_color > 0:
                break
        
        # Create the inverted pattern: filled cells become empty, empty cells get the color
        inverted_pattern = np.zeros_like(grid)
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] == 0:
                    inverted_pattern[r, c] = object_color
        
        # For each cell in the input grid, fill the corresponding subgrid in the output
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] > 0:  # Only filled cells get a pattern
                    # Calculate the top-left corner of the corresponding subgrid
                    sub_r_start = r * grid_size
                    sub_c_start = c * grid_size
                    
                    # Fill the subgrid with the inverted pattern
                    for dr in range(grid_size):
                        for dc in range(grid_size):
                            output_grid[sub_r_start + dr, sub_c_start + dc] = inverted_pattern[dr, dc]
        
        return output_grid

