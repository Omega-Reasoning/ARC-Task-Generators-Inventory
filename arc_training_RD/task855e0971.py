from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import random_cell_coloring, retry
import numpy as np
import random

class Task855e0971Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']} x {vars['columns']}.",
            "Each input grid is partitioned into complete large solid colored strips which either be horizontal bands or vertical bars.",
            "The input grid can have either completely horizontal bands or vertical bars, but not both.",
            "There must be a minimum 2 colored stripes for each occurring and maximum depending on the size of the grid. But make sure that they cover a large area.",
            "The stripes contain only their stripe color and empty cells (black/0). No other colors are present within stripes.",
            "The stripes also have a minimum of one empty cell (black/0) and maximum of two. It is not necessary that all the stripes must have this empty cell."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is copied from the input grid.",
            "In the output grid we activate that stripe by drawing a full black line across it, going all the way from one edge of the stripe to the other, perpendicular to the stripe long axis, passing through the original empty cell position.",
            "All empty cells in all stripes are activated and converted into lines.",
            "The activation line stays strictly within the boundaries of its own stripe and does not cross into other stripes.",
            "Horizontal stripes get vertical lines at the column of their empty cell, but only within that stripe row boundaries.",
            "Vertical stripes get horizontal lines at the row of their empty cell, but only within that stripe column boundaries."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        """Create input grid with colored stripes and empty cells."""
        rows = taskvars['rows']
        columns = taskvars['columns']
        stripe_orientation = gridvars['stripe_orientation']
        num_stripes = gridvars['num_stripes']
        stripe_colors = gridvars['stripe_colors']
        
        grid = np.zeros((rows, columns), dtype=int)
        
        if stripe_orientation == 'horizontal':
            # Create horizontal stripes
            stripe_height = rows // num_stripes
            for i in range(num_stripes):
                start_row = i * stripe_height
                end_row = min((i + 1) * stripe_height, rows)
                if i == num_stripes - 1:  # Last stripe takes remaining rows
                    end_row = rows
                
                color = stripe_colors[i]
                grid[start_row:end_row, :] = color
                
                # Randomly decide if this stripe gets empty cells (not all stripes need them)
                if random.choice([True, False]):  # 50% chance
                    # Add 1-2 empty cells randomly in this stripe
                    num_empty_cells = random.randint(1, 2)
                    for _ in range(num_empty_cells):
                        empty_row = random.randint(start_row, end_row - 1)
                        empty_col = random.randint(0, columns - 1)
                        grid[empty_row, empty_col] = 0  # Empty cell
        else:  # vertical
            # Create vertical stripes
            stripe_width = columns // num_stripes
            for i in range(num_stripes):
                start_col = i * stripe_width
                end_col = min((i + 1) * stripe_width, columns)
                if i == num_stripes - 1:  # Last stripe takes remaining columns
                    end_col = columns
                
                color = stripe_colors[i]
                grid[:, start_col:end_col] = color
                
                # Randomly decide if this stripe gets empty cells (not all stripes need them)
                if random.choice([True, False]):  # 50% chance
                    # Add 1-2 empty cells randomly in this stripe
                    num_empty_cells = random.randint(1, 2)
                    for _ in range(num_empty_cells):
                        empty_row = random.randint(0, rows - 1)
                        empty_col = random.randint(start_col, end_col - 1)
                        grid[empty_row, empty_col] = 0  # Empty cell
        
        return grid

    def transform_input(self, grid, taskvars):
        """Transform input by drawing lines through empty cells perpendicular to stripes, staying within stripe boundaries."""
        rows, columns = grid.shape
        output_grid = grid.copy()
        
        # Find all empty cells (value 0)
        empty_cells = np.where(grid == 0)
        
        # Determine stripe orientation by analyzing the grid structure
        # Check if we have horizontal stripes by looking at color consistency in rows
        horizontal_stripes = True
        for row in range(rows):
            unique_colors_in_row = len(np.unique(grid[row, :]))
            if unique_colors_in_row > 2:  # More than background and one stripe color
                horizontal_stripes = False
                break
        
        if horizontal_stripes:
            # Check if it's actually horizontal by verifying columns have variety
            for col in range(columns):
                unique_colors_in_col = len(np.unique(grid[:, col]))
                if unique_colors_in_col <= 2:  # Not enough variety for horizontal stripes
                    horizontal_stripes = False
                    break
        
        # Process each empty cell
        for empty_row, empty_col in zip(empty_cells[0], empty_cells[1]):
            if horizontal_stripes:
                # For horizontal stripes, draw vertical line within the stripe boundaries
                # Find the stripe this empty cell belongs to
                stripe_start_row = 0
                stripe_end_row = rows
                
                # Find stripe boundaries by looking for color changes above and below
                for r in range(empty_row, -1, -1):
                    if r == 0:
                        stripe_start_row = 0
                        break
                    # Check if there's a color change (different stripe)
                    current_colors = set(grid[r, :]) - {0}
                    above_colors = set(grid[r-1, :]) - {0}
                    if current_colors != above_colors and len(above_colors) > 0:
                        stripe_start_row = r
                        break
                
                for r in range(empty_row, rows):
                    if r == rows - 1:
                        stripe_end_row = rows
                        break
                    # Check if there's a color change (different stripe)
                    current_colors = set(grid[r, :]) - {0}
                    below_colors = set(grid[r+1, :]) - {0}
                    if current_colors != below_colors and len(below_colors) > 0:
                        stripe_end_row = r + 1
                        break
                
                # Draw vertical line only within this stripe
                for r in range(stripe_start_row, stripe_end_row):
                    output_grid[r, empty_col] = 0  # Draw with empty color
            else:
                # For vertical stripes, draw horizontal line within the stripe boundaries
                # Find the stripe this empty cell belongs to
                stripe_start_col = 0
                stripe_end_col = columns
                
                # Find stripe boundaries by looking for color changes left and right
                for c in range(empty_col, -1, -1):
                    if c == 0:
                        stripe_start_col = 0
                        break
                    # Check if there's a color change (different stripe)
                    current_colors = set(grid[:, c]) - {0}
                    left_colors = set(grid[:, c-1]) - {0}
                    if current_colors != left_colors and len(left_colors) > 0:
                        stripe_start_col = c
                        break
                
                for c in range(empty_col, columns):
                    if c == columns - 1:
                        stripe_end_col = columns
                        break
                    # Check if there's a color change (different stripe)
                    current_colors = set(grid[:, c]) - {0}
                    right_colors = set(grid[:, c+1]) - {0}
                    if current_colors != right_colors and len(right_colors) > 0:
                        stripe_end_col = c + 1
                        break
                
                # Draw horizontal line only within this stripe
                for c in range(stripe_start_col, stripe_end_col):
                    output_grid[empty_row, c] = 0  # Draw with empty color
        
        return output_grid

    def create_grids(self):
        """Create training and test grids with variety."""
        # Randomize grid dimensions as task variables
        rows = random.randint(8, 15)
        columns = random.randint(8, 15)
        
        # Store task variables
        taskvars = {
            'rows': rows,
            'columns': columns
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Create training examples
        for i in range(num_train_examples):
            # Randomize stripe orientation
            stripe_orientation = random.choice(['horizontal', 'vertical'])
            
            # Determine number of stripes based on grid size
            if stripe_orientation == 'horizontal':
                max_stripes = min(rows // 2, 6)  # At least 2 rows per stripe
            else:
                max_stripes = min(columns // 2, 6)  # At least 2 columns per stripe
            
            num_stripes = random.randint(2, max_stripes)
            
            # Generate stripe colors (avoiding background=0)
            available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            stripe_colors = random.sample(available_colors, num_stripes)
            
            gridvars = {
                'stripe_orientation': stripe_orientation,
                'num_stripes': num_stripes,
                'stripe_colors': stripe_colors
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        stripe_orientation = random.choice(['horizontal', 'vertical'])
        if stripe_orientation == 'horizontal':
            max_stripes = min(rows // 2, 6)
        else:
            max_stripes = min(columns // 2, 6)
        
        num_stripes = random.randint(2, max_stripes)
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        stripe_colors = random.sample(available_colors, num_stripes)
        
        test_gridvars = {
            'stripe_orientation': stripe_orientation,
            'num_stripes': num_stripes,
            'stripe_colors': stripe_colors
        }
        
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

