from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects
from Framework.input_library import retry, random_cell_coloring
import numpy as np
import random

class Task140c817eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square-shaped and can vary in size.",
            "Each grid has a solid background filled with a single color (1–9) and contains several {color('cell_color')} cells.",
            "All {color('cell_color')} cells must be separated by at least three layers of background cells.",
            "No {color('cell_color')} cell should appear on the grid border."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are created by copying the input grids and identifying all {color('cell_color')} cells.",
            "For each {color('cell_color')} cell, fill its entire row and column with the {color('cell_color')} color, and change the original {color('cell_color')} cell itself to {color('new_color')}.",
            "Then, for every {color('new_color')} cell, color its four diagonally adjacent cells (top-left, top-right, bottom-left, bottom-right) using 8-way connectivity with {color('fill_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        grid_size = gridvars['grid_size']
        background = gridvars['background']
        cell_color = taskvars['cell_color']
        num_cells = gridvars['num_cells']
        
        def generate_grid():
            # Create grid with solid background
            grid = np.full((grid_size, grid_size), background, dtype=int)
            
            # Generate positions for cell_color cells
            positions = []
            max_attempts = 1000
            
            for _ in range(num_cells):
                for attempt in range(max_attempts):
                    # Ensure cells are not on border (at least 1 cell away from edge)
                    r = random.randint(1, grid_size - 2)
                    c = random.randint(1, grid_size - 2)
                    
                    # Check if position is valid (at least 3 cells away from other cell_color cells)
                    valid = True
                    for existing_r, existing_c in positions:
                        if abs(r - existing_r) < 3 or abs(c - existing_c) < 3:
                            valid = False
                            break
                    
                    if valid:
                        positions.append((r, c))
                        grid[r, c] = cell_color
                        break
                else:
                    # If we can't place all cells, try with a smaller number
                    if len(positions) >= 2:  # At least 2 cells
                        break
                    else:
                        # Restart the entire process
                        return None
            
            return grid if len(positions) >= 2 else None
        
        return retry(generate_grid, lambda x: x is not None, max_attempts=100)
    
    def transform_input(self, grid, taskvars):
        output_grid = grid.copy()
        cell_color = taskvars['cell_color']
        new_color = taskvars['new_color']
        fill_color = taskvars['fill_color']
        
        # Find all cell_color cells
        cell_positions = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == cell_color:
                    cell_positions.append((r, c))
        
        # Step 1: Fill rows and columns for each cell_color cell
        for r, c in cell_positions:
            # Fill entire row
            output_grid[r, :] = cell_color
            # Fill entire column
            output_grid[:, c] = cell_color
            # Change the original cell to new_color
            output_grid[r, c] = new_color
        
        # Step 2: For every new_color cell, color its diagonal neighbors with fill_color
        new_color_positions = []
        for r in range(output_grid.shape[0]):
            for c in range(output_grid.shape[1]):
                if output_grid[r, c] == new_color:
                    new_color_positions.append((r, c))
        
        # Add diagonal neighbors for each new_color cell
        diagonal_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # top-left, top-right, bottom-left, bottom-right
        
        for r, c in new_color_positions:
            for dr, dc in diagonal_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < output_grid.shape[0] and 0 <= nc < output_grid.shape[1]:
                    output_grid[nr, nc] = fill_color
        
        return output_grid
    
    def create_grids(self):
        # Select colors for the task (cell_color, new_color, fill_color remain consistent)
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        
        cell_color = all_colors[0]
        new_color = all_colors[1]
        fill_color = all_colors[2]
        
        taskvars = {
            'cell_color': cell_color,
            'new_color': new_color,
            'fill_color': fill_color
        }
        
        # Generate training examples
        train_examples = []
        for _ in range(3):
            # For each example, select a different background color
            available_colors = [c for c in range(1, 10) if c not in [cell_color, new_color, fill_color]]
            background = random.choice(available_colors)
            
            grid_size = random.randint(8, 30)
            num_cells = random.randint(2, 5)
            gridvars = {'grid_size': grid_size, 'num_cells': num_cells, 'background': background}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        available_colors = [c for c in range(1, 10) if c not in [cell_color, new_color, fill_color]]
        background = random.choice(available_colors)
        
        grid_size = random.randint(8, 30)
        num_cells = random.randint(2, 5)
        gridvars = {'grid_size': grid_size, 'num_cells': num_cells, 'background': background}
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

