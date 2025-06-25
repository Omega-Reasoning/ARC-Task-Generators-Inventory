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
            "The liquid level may fill part of the tank, but its filled evenly (i.e., full rows from the bottom up).",
            "There is always at least one unfilled row in the tank."
        ]
        
        transformation_reasoning_chain = [
            "The output is a 3x3 grid.",
            "Determine the number of unfilled rows in the input tank by counting the empty rows above the liquid level.",
            "Then, in the 3x3 output grid, shade cells in an alternating pattern using the {color('liquid_color')} color, corresponding to the number of unfilled rows identified.",
            "The pattern alternates: first row left-to-right, second row right-to-left, third row left-to-right.",
            "All other cells remain empty or uncolored.",
            "For example: If 5 unfilled rows exist in the input, shade the first 5 cells following the alternating pattern: (0,0), (0,1), (0,2), (1,2), (1,1) of the 3x3 output grid with {color('liquid_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """Create input grid with tank structure and liquid."""
        # Random grid size - make it larger to accommodate more liquid rows
        grid_height = random.randint(10, 15)  # Increased minimum height
        grid_width = random.randint(8, 12)    # Increased minimum width to ensure tank fits
        
        grid = np.zeros((grid_height, grid_width), dtype=np.int32)
        
        # Get colors from taskvars
        tank_color = taskvars['tank_color']
        liquid_color = taskvars['liquid_color']
        
        # Create tank structure (open at top, closed at bottom and sides)
        # Ensure tank width is valid: minimum 5, maximum based on grid width
        max_tank_width = min(grid_width - 2, 8)  # Leave at least 1 cell on each side
        min_tank_width = 5  # Minimum 5 width (3 inside + 2 walls)
        
        # Ensure we have a valid range
        if max_tank_width < min_tank_width:
            max_tank_width = min_tank_width
        
        tank_width = random.randint(min_tank_width, max_tank_width)
        
        # Tank height calculation
        max_tank_height = min(grid_height - 1, 12)
        min_tank_height = 8
        
        if max_tank_height < min_tank_height:
            max_tank_height = min_tank_height
        
        tank_height = random.randint(min_tank_height, max_tank_height)
        
        # Position tank in grid - ensure it fits
        max_start_col = grid_width - tank_width
        if max_start_col < 0:
            max_start_col = 0
        
        start_col = random.randint(0, max_start_col) if max_start_col > 0 else 0
        start_row = max(0, grid_height - tank_height)
        
        # Draw tank outline
        # Bottom row
        for c in range(start_col, min(start_col + tank_width, grid_width)):
            if grid_height - 1 >= 0:
                grid[grid_height - 1, c] = tank_color
        
        # Left and right walls
        for r in range(start_row, grid_height - 1):
            if start_col < grid_width:
                grid[r, start_col] = tank_color
            if start_col + tank_width - 1 < grid_width:
                grid[r, start_col + tank_width - 1] = tank_color
        
        # Calculate liquid level ensuring at least one unfilled row
        tank_interior_height = tank_height - 1  # Exclude bottom row
        max_liquid_rows = max(1, tank_interior_height - 1)  # Ensure at least 1 unfilled row
        
        # Fill with liquid from bottom up - ensure at least 1 unfilled row
        liquid_level = random.randint(1, max_liquid_rows)
        
        # Fill liquid rows
        for level in range(liquid_level):
            row = grid_height - 2 - level  # Start from second-to-bottom row
            if row >= 0:
                for c in range(start_col + 1, min(start_col + tank_width - 1, grid_width)):
                    if c >= 0:
                        grid[row, c] = liquid_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """Transform input by creating 3x3 grid with unfilled rows representation"""
        output_grid = np.zeros((3, 3), dtype=np.int32)
        
        # Count unfilled rows in input grid
        liquid_color = taskvars['liquid_color']
        tank_color = taskvars['tank_color']
        
        # Find the tank boundaries
        height, width = grid.shape
        
        # Find top of liquid (highest row with liquid)
        top_liquid_row = None
        for row in range(height):
            row_has_liquid = False
            for col in range(width):
                if grid[row, col] == liquid_color:
                    row_has_liquid = True
                    break
            
            if row_has_liquid:
                top_liquid_row = row
                break
        
        # Find the top of tank (first row with tank walls)
        top_tank_row = None
        for row in range(height):
            row_has_tank = False
            for col in range(width):
                if grid[row, col] == tank_color:
                    row_has_tank = True
                    break
            
            if row_has_tank:
                top_tank_row = row
                break
        
        # Count unfilled rows
        unfilled_rows = 0
        if top_liquid_row is not None and top_tank_row is not None:
            # Count rows between top of tank and top of liquid
            unfilled_rows = top_liquid_row - top_tank_row
        elif top_tank_row is not None:
            # If no liquid found, count all interior rows as unfilled
            # Find bottom of tank
            bottom_tank_row = None
            for row in range(height - 1, -1, -1):
                row_has_tank = False
                for col in range(width):
                    if grid[row, col] == tank_color:
                        row_has_tank = True
                        break
                
                if row_has_tank:
                    bottom_tank_row = row
                    break
            
            if bottom_tank_row is not None:
                unfilled_rows = max(1, bottom_tank_row - top_tank_row - 1)  # Exclude top and bottom tank walls
        
        # Ensure we have at least 1 unfilled row (as guaranteed by input generation)
        unfilled_rows = max(1, unfilled_rows)
        
        # Create alternating/zigzag fill pattern
        # Row 0: left-to-right (0,0), (0,1), (0,2)
        # Row 1: right-to-left (1,2), (1,1), (1,0) 
        # Row 2: left-to-right (2,0), (2,1), (2,2)
        zigzag_order = [
            (0, 0), (0, 1), (0, 2),  # Row 0: left to right
            (1, 2), (1, 1), (1, 0),  # Row 1: right to left
            (2, 0), (2, 1), (2, 2)   # Row 2: left to right
        ]
        
        # Fill cells based on unfilled rows using zigzag pattern
        cells_to_fill = min(unfilled_rows, 9)  # Cap at 9 since output is 3x3
        
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