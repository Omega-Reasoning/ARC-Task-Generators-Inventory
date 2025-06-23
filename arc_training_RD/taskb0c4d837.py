from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random

class Taskb0c4d837Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid depicts a tank-like open structure.",
            "The tank structure is outlined in a specific {color('tank_color')}.",
            "The tank contains a uniformly filled liquid using a specific {color('liquid_color')} color.",
            "The liquid level may fill part or all of the tank, but its filled evenly (i.e., full rows from the bottom up)."
        ]
        
        transformation_reasoning_chain = [
            "The output is a 3x3 grid.",
            "Determine the liquid level in the input tank by counting the number of rows filled with {color('liquid_color')} color.",
            "Then, in the 3x3 output grid, shade cells in an alternating pattern using the {color('liquid_color')} color, corresponding to the liquid level identified.",
            "The pattern alternates: first row left-to-right, second row right-to-left, third row left-to-right.",
            "All other cells remain empty or uncolored.",
            "For example: If 5 rows are filled in the input, shade the first 5 cells following the alternating pattern: (0,0), (0,1), (0,2), (1,2), (1,1) of the 3x3 output grid with {color('liquid_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with tank structure and liquid."""
        # Random grid size - make it larger to accommodate more liquid rows
        grid_height = random.randint(8, 15)  # Increased minimum height
        grid_width = random.randint(5, 10)
        
        grid = np.zeros((grid_height, grid_width), dtype=np.int32)
        
        # Get colors from taskvars
        tank_color = taskvars['tank_color']
        liquid_color = taskvars['liquid_color']
        
        # Create tank structure (open at top, closed at bottom and sides)
        tank_width = random.randint(3, min(grid_width - 2, 6))
        tank_height = random.randint(6, min(grid_height - 1, 10))  # Increased minimum tank height
        
        # Position tank in grid
        start_col = random.randint(1, grid_width - tank_width - 1)
        start_row = grid_height - tank_height
        
        # Draw tank outline
        # Bottom row
        for c in range(start_col, start_col + tank_width):
            grid[grid_height - 1, c] = tank_color
        
        # Left and right walls
        for r in range(start_row, grid_height - 1):
            grid[r, start_col] = tank_color
            grid[r, start_col + tank_width - 1] = tank_color
        
        # Fill with liquid from bottom up - NOW ALLOW 1-6 ROWS
        max_possible_liquid_rows = min(tank_height - 1, 6)  # Allow up to 6 rows
        liquid_level = random.randint(1, max_possible_liquid_rows)
        
        for level in range(liquid_level):
            row = grid_height - 2 - level  # Start from second-to-bottom row
            for c in range(start_col + 1, start_col + tank_width - 1):
                grid[row, c] = liquid_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by creating 3x3 grid with liquid level representation"""
        output_grid = np.zeros((3, 3), dtype=np.int32)
        
        # Count liquid level in input grid
        liquid_color = taskvars['liquid_color']
        liquid_level = 0
        
        # Count from bottom up how many complete rows have liquid
        height, width = grid.shape
        for row in range(height - 2, -1, -1):  # Start from second-to-bottom row going up
            row_has_liquid = False
            for col in range(width):
                if grid[row, col] == liquid_color:
                    row_has_liquid = True
                    break
            
            if row_has_liquid:
                liquid_level += 1
            else:
                break  # Stop when we hit a row without liquid
        
        # Create alternating/zigzag fill pattern
        # Row 0: left-to-right (0,0), (0,1), (0,2)
        # Row 1: right-to-left (1,2), (1,1), (1,0) 
        # Row 2: left-to-right (2,0), (2,1), (2,2)
        zigzag_order = [
            (0, 0), (0, 1), (0, 2),  # Row 0: left to right
            (1, 2), (1, 1), (1, 0),  # Row 1: right to left
            (2, 0), (2, 1), (2, 2)   # Row 2: left to right
        ]
        
        # Fill cells based on liquid level using zigzag pattern
        cells_to_fill = min(liquid_level, 9)  # Cap at 9 since output is 3x3
        
        for i in range(cells_to_fill):
            row, col = zigzag_order[i]
            output_grid[row, col] = liquid_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict, dict]:
        """Create train and test grids with consistent variables."""
        # Generate colors ensuring they're different
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        tank_color = available_colors[0]
        liquid_color = available_colors[1]
        
        # Store task variables
        taskvars = {
            "tank_color": tank_color,
            "liquid_color": liquid_color
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            gridvars = {}  # No grid-specific variables needed
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }