from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry
from transformation_library import find_connected_objects

class Task3ac3eb23Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different numbers of columns but each has {vars['rows']} rows.",
            "They contain one or more multi-colored (1-9) cells in the first row, excluding the first and last columns .",
            "Each colored cell in the first row is separated by the other by at least three empty (0) cells."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and generating a checkerboard pattern below each colored cell.",
            "The checkerboard alternates between empty and colored cells across rows and columns, starting from the second row and extending from the column to the left of each colored cell to the column on its right.",
            "The color of the checkerboard pattern matches the color of the respective cell in the first row."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables (rows) with even value between 5 and 30
        taskvars = {'rows': random.randrange(4, 28, 2)}  # ensure even number
        
        # Create 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Choose one example index to have exactly one colored cell
        singleton_example_index = random.randint(0, num_train_examples - 1)
        
        for i in range(num_train_examples):
            # Decide if this should be the singleton example
            force_single_color = (i == singleton_example_index)
            
            # For non-singleton examples, ensure they have more than one colored cell
            force_multiple_colors = not force_single_color
            
            # Create random grid dimensions with columns between 5 and 30
            cols = random.randint(5, 30)
            
            # Create grid variables for this example
            gridvars = {
                'cols': cols, 
                'force_single_color': force_single_color,
                'force_multiple_colors': force_multiple_colors
            }
            
            # Create input and output grids
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create one test example with multiple colored cells
        test_cols = random.randint(5, 30)
        gridvars = {
            'cols': test_cols, 
            'force_single_color': False,
            'force_multiple_colors': True  # Ensure test has multiple colors
        }
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        # Extract variables
        rows = taskvars['rows']
        cols = gridvars['cols']
        force_single_color = gridvars.get('force_single_color', False)
        force_multiple_colors = gridvars.get('force_multiple_colors', False)
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Determine number of colored cells based on grid width
        if force_single_color:
            num_colored_cells = 1
        elif force_multiple_colors:
            max_cells = min(5, (cols - 4) // 4)  # Need at least 3 spaces between cells
            num_colored_cells = random.randint(2, max(2, max_cells))  # At least 2 cells
        else:
            max_cells = min(5, (cols - 4) // 4)
            num_colored_cells = random.randint(1, max(1, max_cells))
        
        # Available columns (exclude first and last)
        available_columns = list(range(1, cols - 1))
        
        # Available colors (1-9)
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Place colored cells in first row
        placed_columns = []
        for i in range(num_colored_cells):
            def valid_column(col):
                # Check if column is at least 3 cells away from any placed column
                return all(abs(col - placed_col) > 3 for placed_col in placed_columns)
            
            valid_columns = [col for col in available_columns if valid_column(col)]
            if not valid_columns:
                break  # No valid columns left
                
            col = random.choice(valid_columns)
            color = available_colors[i % len(available_colors)]
            
            grid[0, col] = color
            placed_columns.append(col)
            available_columns.remove(col)
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Create a copy of the input grid
        output_grid = grid.copy()
        rows, cols = grid.shape
        
        # Find all colored cells in the first row
        colored_cells = [(0, c, grid[0, c]) for c in range(cols) if grid[0, c] > 0]
        
        # For each colored cell, create a checkerboard pattern below it
        for r, c, color in colored_cells:
            # Define pattern boundaries (left, right, bottom)
            left_col = max(0, c - 1)
            right_col = min(cols - 1, c + 1)
            
            # Generate checkerboard pattern starting from second row
            for row in range(1, rows):
                for col in range(left_col, right_col + 1):
                    # Determine if this cell should be colored or empty based on checkerboard pattern
                    # Start with empty at diagonal from bottom-left of colored cell
                    if (row + col) % 2 == ((1 + left_col) % 2):
                        output_grid[row, col] = color
                    else:
                        output_grid[row, col] = 0
        
        return output_grid

