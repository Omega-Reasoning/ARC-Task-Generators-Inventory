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
            "Each colored cell in the first row is separated by the other by at least two empty (0) cells."
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

        # Create 3-5 training examples (plus 1 test) -> total 4-6 examples
        num_train_examples = random.randint(3, 5)
        train_examples = []

        total_examples = num_train_examples + 1  # include the test example

        # Sample unique numbers of colored cells between 1 and 6 for each example
        # This guarantees strictly different counts across all examples
        counts = random.sample(list(range(1, 7)), total_examples)

        # Create train examples with assigned unique counts
        for i in range(num_train_examples):
            desired_colored = counts[i]

            # Ensure columns chosen can fit the desired number of colored cells
            min_cols_needed = max(5, 3 * desired_colored)  # derived from spacing rules (>=2 empty cells between)
            cols = random.randint(min_cols_needed, 30)

            gridvars = {
                'cols': cols,
                'desired_colored': desired_colored
            }

            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)

            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })

        # Create one test example with the remaining unique count
        desired_colored_test = counts[-1]
        min_cols_needed = max(5, 3 * desired_colored_test)
        test_cols = random.randint(min_cols_needed, 30)

        gridvars = {
            'cols': test_cols,
            'desired_colored': desired_colored_test
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
        desired_colored = gridvars.get('desired_colored', None)
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Determine desired number of colored cells. If provided, use it.
        if desired_colored is not None:
            num_colored_cells = int(desired_colored)
        else:
            # Fallback: compute a sensible number based on grid width
            max_cells = min(6, (cols - 4) // 4 + 1)
            num_colored_cells = random.randint(1, max(1, max_cells))
        
        # Available columns (exclude first and last)
        available_columns = list(range(1, cols - 1))

        # Available colors (1-9)
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)

        # Place colored cells in first row ensuring at least 3 empty cells between any two
        placed_columns = []

        # Try randomized sampling a few times to get well-separated columns
        success = False
        attempts = 0
        while not success and attempts < 200:
            attempts += 1
            if num_colored_cells == 1:
                candidate = [random.choice(available_columns)]
            else:
                candidate = random.sample(available_columns, num_colored_cells)
            candidate.sort()
            if all(abs(candidate[i] - candidate[i - 1]) > 2 for i in range(1, len(candidate))):
                placed_columns = candidate
                success = True

        # If randomized attempts fail (shouldn't if cols >= minimal), fall back to greedy placement
        if not success:
            placed_columns = []
            col = available_columns[0]
            for _ in range(num_colored_cells):
                if col > available_columns[-1]:
                    break
                placed_columns.append(col)
                col += 3

        # Assign colors to placed columns
        for i, col in enumerate(placed_columns):
            color = available_colors[i % len(available_colors)]
            grid[0, col] = color
        
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

