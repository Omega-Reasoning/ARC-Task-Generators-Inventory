from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity, retry
from Framework.transformation_library import find_connected_objects, GridObject

class Task6d58a25dGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain an arrow-shaped object of the form [[0,0,0, c, 0, 0,0], [0,0, c, c, c, 0,0], [0,c, c, 0, c, c,0],[c, 0,0,0, 0, 0, c]] for a color c, and several single-colored cells.",
            "The arrow-shaped object and the single-colored cells are always differently colored, with their colors varying across examples.",
            "The single-colored cells are uniformly distributed throughout the grid and remain completely separated from each other and from all other colored regions.",
            "The arrow shaped object should always appear in top-half of the grid.",
            "The number of single-colored cells is always less than {vars['grid_size']}."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grid and identifying the arrow-shaped object.",
            "An arrow shape is defined as [[0, 0, c, 0, 0], [0, c, c, c, 0], [c, c, 0, c, c],[c, 0, 0, 0, c] ] for a color c.",
            "Once identified, locate all single-colored cells that appear under the arrow-shaped object and occupy the same columns as the arrow-shaped object.",
            "For each of these columns, fill in all empty (0) cells from the last row up to the first cell belonging to the arrow-shaped object.",
            "The arrow shape remains unchanged after the transformation."
        ]
        
        taskvars_definitions = {
            "grid_size": lambda: random.randint(15, 30)
        }
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Set task variables
        taskvars = {
            "grid_size": random.randint(10, 30)
        }
        
        # Generate 3-4 training examples
        num_train_examples = random.randint(3, 4)
        
        train_data = []
        for _ in range(num_train_examples):
            # Create a unique gridvars dictionary for each example
            gridvars = {
                "arrow_color": random.randint(1, 9),
                "cell_color": random.randint(1, 9),
                "num_cells": random.randint(taskvars["grid_size"]//2, taskvars["grid_size"]-1),
                "cells_below_arrow": random.randint(2, 4)  # Ensure 2-4 cells below arrow
            }
            
            # Ensure arrow and cell colors are different
            while gridvars["arrow_color"] == gridvars["cell_color"]:
                gridvars["cell_color"] = random.randint(1, 9)
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_data.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {
            "arrow_color": random.randint(1, 9),
            "cell_color": random.randint(1, 9),
            "num_cells": random.randint(taskvars["grid_size"]//2, taskvars["grid_size"]-1),
            "cells_below_arrow": random.randint(2, 4)  # Ensure 2-4 cells below arrow
        }
        
        # Ensure arrow and cell colors are different
        while test_gridvars["arrow_color"] == test_gridvars["cell_color"]:
            test_gridvars["cell_color"] = random.randint(1, 9)
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_data = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        # Initialize an empty grid
        grid_size = taskvars["grid_size"]
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Define the arrow shape
        arrow_rows = [
            [0, 0, 0, gridvars["arrow_color"], 0, 0, 0],
            [0, 0, gridvars["arrow_color"], gridvars["arrow_color"], gridvars["arrow_color"], 0, 0],
            [0, gridvars["arrow_color"], gridvars["arrow_color"], 0, gridvars["arrow_color"], gridvars["arrow_color"], 0],
            [gridvars["arrow_color"], 0, 0, 0, 0, 0, gridvars["arrow_color"]]
        ]
        arrow = np.array(arrow_rows)
        
        # Position the arrow in the top half of the grid
        arrow_height, arrow_width = arrow.shape
        max_row_pos = grid_size // 2 - arrow_height
        arrow_row = random.randint(1, max_row_pos)
        max_col_pos = grid_size - arrow_width
        arrow_col = random.randint(0, max_col_pos)
        
        # Place the arrow on the grid
        grid[arrow_row:arrow_row+arrow_height, arrow_col:arrow_col+arrow_width] = arrow
        
        # Get arrow columns (where arrow has colored cells)
        arrow_cols = []
        for j in range(arrow_width):
            if any(arrow[:, j] == gridvars["arrow_color"]):
                arrow_cols.append(arrow_col + j)
        
        # Place cells below arrow (2-4 cells as specified)
        cells_below_arrow = gridvars["cells_below_arrow"]
        cells_placed_below = 0
        
        # Choose random arrow columns to place cells below
        columns_to_fill = random.sample(arrow_cols, min(cells_below_arrow, len(arrow_cols)))
        
        for col in columns_to_fill:
            # Find the lowest row of the arrow in this column
            arrow_bottom = 0
            for r in range(arrow_row, arrow_row+arrow_height):
                if grid[r, col] == gridvars["arrow_color"]:
                    arrow_bottom = r
            
            # Place cell below the arrow
            available_rows = list(range(arrow_bottom + 2, grid_size))
            if not available_rows:  # Skip if no available space
                continue
                
            row = random.choice(available_rows)
            
            # Check if surrounding cells (4-way) are empty
            if row > 0 and grid[row-1, col] != 0:
                continue
            if row < grid_size-1 and grid[row+1, col] != 0:
                continue
            if col > 0 and grid[row, col-1] != 0:
                continue
            if col < grid_size-1 and grid[row, col+1] != 0:
                continue
            
            grid[row, col] = gridvars["cell_color"]
            cells_placed_below += 1
            
        # If we couldn't place enough cells below the arrow, retry with more columns
        attempts = 0
        while cells_placed_below < cells_below_arrow and attempts < 100:
            col = random.choice(arrow_cols)
            
            # Find the lowest row of the arrow in this column
            arrow_bottom = 0
            for r in range(arrow_row, arrow_row+arrow_height):
                if grid[r, col] == gridvars["arrow_color"]:
                    arrow_bottom = r
            
            row = random.randint(arrow_bottom + 2, grid_size - 1)
            
            # Check if surrounding cells (4-way) are empty
            if row > 0 and grid[row-1, col] != 0:
                attempts += 1
                continue
            if row < grid_size-1 and grid[row+1, col] != 0:
                attempts += 1
                continue
            if col > 0 and grid[row, col-1] != 0:
                attempts += 1
                continue
            if col < grid_size-1 and grid[row, col+1] != 0:
                attempts += 1
                continue
            
            if grid[row, col] == 0:  # Cell is empty
                grid[row, col] = gridvars["cell_color"]
                cells_placed_below += 1
            
            attempts += 1
        
        # Place remaining cells randomly throughout the grid
        cells_to_place = gridvars["num_cells"] - cells_placed_below
        cells_placed = 0
        attempts = 0
        
        while cells_placed < cells_to_place and attempts < 1000:
            row = random.randint(0, grid_size - 1)
            col = random.randint(0, grid_size - 1)
            
            # Check if surrounding cells (4-way) are empty
            if row > 0 and grid[row-1, col] != 0:
                attempts += 1
                continue
            if row < grid_size-1 and grid[row+1, col] != 0:
                attempts += 1
                continue
            if col > 0 and grid[row, col-1] != 0:
                attempts += 1
                continue
            if col < grid_size-1 and grid[row, col+1] != 0:
                attempts += 1
                continue
            
            if grid[row, col] == 0:  # Cell is empty
                grid[row, col] = gridvars["cell_color"]
                cells_placed += 1
            
            attempts += 1
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Create a copy of the input grid
        output_grid = grid.copy()
        grid_size = grid.shape[0]
        
        # Find objects in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        # Find the arrow object (largest object)
        arrow_obj = objects.sort_by_size(reverse=True)[0]
        arrow_color = list(arrow_obj.colors)[0]  # Get the color of the arrow
        
        # Find single-colored cells
        single_cells = objects.with_size(1)
        
        # We need to ensure we're using the correct color for the single cells,
        # not the arrow color. First, determine what color the single cells are.
        if len(single_cells) > 0:
            cell_color = list(single_cells[0].colors)[0]
            
            # Make sure this is not the arrow color
            if cell_color == arrow_color and len(single_cells) > 1:
                # Look for a different cell color
                for cell in single_cells:
                    this_color = list(cell.colors)[0]
                    if this_color != arrow_color:
                        cell_color = this_color
                        break
        else:
            # Fallback - shouldn't happen with valid grids
            cell_color = 1 if arrow_color != 1 else 2
        
        # Identify columns occupied by the arrow
        arrow_columns = set()
        for r, c, _ in arrow_obj.cells:
            arrow_columns.add(c)
        
        # Find the bottom row of the arrow for each column
        arrow_bottom = {}
        for r, c, _ in arrow_obj.cells:
            if c not in arrow_bottom or r > arrow_bottom[c]:
                arrow_bottom[c] = r
        
        # Find single cells below the arrow in the same columns
        columns_to_fill = set()
        for cell_obj in single_cells:
            for r, c, _ in cell_obj.cells:
                if c in arrow_columns and r > arrow_bottom[c]:
                    columns_to_fill.add(c)
        
        # Fill the columns with the single cell color
        for col in columns_to_fill:
            bottom_arrow_row = arrow_bottom[col]
            
            # Fill from the bottom of the grid up to the arrow
            for row in range(bottom_arrow_row + 1, grid_size):
                output_grid[row, col] = cell_color
        
        return output_grid

