from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import retry

class Task496994bdGenerator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain (improved for clarity)
        input_reasoning_chain = [
            "Input grids contain different number of rows, but the number of columns is {vars['num_columns']}.",
            "Each input grid contains the first 2,3, or 4 rows completely filled with colored cells, forming horizontal stripes, while the rest of the grid is empty (0).",
            "Each stripe is made using a single color with the color varying across stripes.",
        ]
        
        # Transformation reasoning chain from requirements
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the colored horizontal stripes.",
            "The identified colored stripes are duplicated and appended at the bottom of the grid.",
            "When appending the duplicated stripes the order is reversed (the topmost stripe appears at the very bottom).",
            "The top horizontal stripes are always preserved in their original order at the top of the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Choose a task-level number of columns to use for examples (task variable)
        # This variable controls grid width and must be between 5 and 30.
        taskvars = {
            'num_columns': random.randint(5, 30)
        }
        
        
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        # Keep track of color pairs used to ensure variety
        used_color_sets = set()
        
        for _ in range(num_train_examples):
            # For each example choose rows independently; use task-level columns
            rows = random.randint(8, 30)
            # Use task-level column count unless overridden in gridvars
            cols = taskvars.get('num_columns')

            # Number of colored stripes (top rows) varies between 2 and 4
            num_stripes = random.randint(2, 4)

            # Number of distinct colors used in this grid varies between 2 and 4
            num_colors = random.randint(2, 4)

            # Pick distinct colors (1-9)
            colors = random.sample(list(range(1, 10)), num_colors)

            # Ensure color set hasn't been used before (variety across grids)
            colors_key = tuple(sorted(colors))
            # try a few times to get a different color set if collision occurs
            attempts = 0
            while colors_key in used_color_sets and attempts < 10:
                colors = random.sample(list(range(1, 10)), num_colors)
                colors_key = tuple(sorted(colors))
                attempts += 1
            used_color_sets.add(colors_key)

            # Prepare gridvars including per-grid size and color info
            gridvars = {
                'rows': rows,
                'cols': cols,
                'num_stripes': num_stripes,
                'colors': colors,
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example with a color set different from training examples
        test_rows = random.randint(8, 30)
        test_cols = taskvars.get('num_columns')
        test_num_stripes = random.randint(2, 4)
        test_num_colors = random.randint(2, 4)
        test_colors = random.sample(list(range(1, 10)), test_num_colors)
        attempts = 0
        while tuple(sorted(test_colors)) in used_color_sets and attempts < 20:
            test_colors = random.sample(list(range(1, 10)), test_num_colors)
            attempts += 1

        test_gridvars = {
            'rows': test_rows,
            'cols': test_cols,
            'num_stripes': test_num_stripes,
            'colors': test_colors,
        }

        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        # Read per-grid size and color configuration from gridvars
        rows = gridvars.get('rows', random.randint(6, 30))
        # Prefer explicit gridvars, otherwise use task-level `num_columns` (5..30)
        cols = gridvars.get('cols', taskvars.get('num_columns', random.randint(5, 30)))
        num_stripes = gridvars.get('num_stripes', random.randint(2, 4))
        colors = gridvars.get('colors', random.sample(list(range(1, 10)), random.randint(2, 4)))

        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Build a sequence of colors to assign to the top stripes.
        # Ensure each color in `colors` is used at least once when num_stripes >= len(colors)
        stripe_colors = []
        stripe_colors.extend(colors)
        remaining = num_stripes - len(stripe_colors)
        if remaining > 0:
            stripe_colors.extend(random.choices(colors, k=remaining))
        # If there are more colors than stripes, sample a subset
        if len(stripe_colors) > num_stripes:
            stripe_colors = random.sample(stripe_colors, num_stripes)

        # If the list is shorter than required (shouldn't be), pad with first color
        while len(stripe_colors) < num_stripes:
            stripe_colors.append(colors[0])

        # Apply the stripe colors to the top rows
        for i in range(num_stripes):
            grid[i, :] = stripe_colors[i]
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any] = None) -> np.ndarray:
        # Work from the grid itself to determine rows; taskvars is unused for size
        rows = grid.shape[0]

        # Create a copy of the input grid to modify
        output_grid = grid.copy()

        # Find the rows that contain colored cells (non-zero values)
        colored_rows = [r for r in range(rows) if np.any(grid[r] != 0)]

        # Duplicate the colored rows in reverse order at the bottom of the grid
        bottom_position = rows - len(colored_rows)
        for i, row_idx in enumerate(reversed(colored_rows)):
            output_grid[bottom_position + i] = grid[row_idx].copy()

        return output_grid

