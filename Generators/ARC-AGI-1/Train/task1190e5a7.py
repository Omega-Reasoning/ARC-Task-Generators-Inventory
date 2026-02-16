import numpy as np
import random
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, retry
from Framework.transformation_library import find_connected_objects

class Task1190e5a7Generator(ARCTaskGenerator):
    def __init__(self):
        # Updated input reasoning chain with enhanced observations about the grid.
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "The entire grid is completely filled with a single background color, that varies per example.",
            "Next, certain number of rows (a) and columns (b)are completely filled with a different color, also varying per example.",
            "The filled rows and columns create visible boundaries, effectively dividing the grid into multiple sub-regions of varying sizes."
        ]
        # Initialize the transformation reasoning chain exactly as given (with a slight adaptation to clarify the boundaries).
        transformation_reasoning_chain = [
            "The output grid has a different size than the input grid.",
            "The output grid is formed by identifying the filled rows and columns in the input grid, which act as boundaries.",
            "These boundaries divide the input grid into subgrids.",
            "The output grid is constructed such that each cell corresponds to one of these subgrids from the input grid i.e., the number of rows and columns in the output grid is one more than the number of horizontal and vertical boundaries respectively."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        # Merge gridvars into taskvars; gridvars takes precedence.
        config = {**taskvars, **gridvars}
        rows = config['rows']
        grid = np.full((rows, rows), config['color_1'], dtype=int)
        # Paint the selected rows with color_2.
        for r in config['colored_rows']:
            grid[r, :] = config['color_2']
        # Paint the selected columns with color_2.
        for c in config['colored_cols']:
            grid[:, c] = config['color_2']
        return grid

    def transform_input(self, grid: np.ndarray) -> np.ndarray:
        # Determine the background color by finding the most common value in the grid.
        unique, counts = np.unique(grid, return_counts=True)
        background_color = unique[np.argmax(counts)]
        
        # Identify the filled color (i.e. the row/column color) as the non-background color.
        filled_color = unique[unique != background_color][0]

        # Identify the rows that are entirely painted with the painted_color.
        colored_rows = [i for i in range(grid.shape[0]) if np.all(grid[i, :] == filled_color)]
        # Identify the columns that are entirely painted with the painted_color.
        colored_cols = [j for j in range(grid.shape[1]) if np.all(grid[:, j] == filled_color)]
        
        # Calculate the number of subgrids: one more than the number of dividing (painted) rows/columns.
        new_rows = len(colored_rows) + 1
        new_cols = len(colored_cols) + 1
        
        # Build the output grid (the "answer") filled completely with the background color.
        output_grid = np.full((new_rows, new_cols), background_color, dtype=int)
        return output_grid

    def create_grids(self) -> tuple:
        # Define a task-level variable 'grid_size' (range 5..30). This will be
        # available as vars['grid_size'] in reasoning strings and is used as the
        # grid dimension for every example generated in this task instance.
        taskvars = {
            'grid_size': random.randint(5, 30)
        }

        def generate_gridvars():
            # Use the task-level 'grid_size' variable (random int between 5 and 30).
            # This allows varying the input sizes via vars['grid_size'] while
            # keeping other per-example parameters randomized.
            rows = taskvars['grid_size']
            # Select color_1 (background) randomly from 1 to 9.
            color_1 = random.randint(1, 9)
            # Select color_2 (painted color) from 1 to 9, ensuring it is different from color_1.
            possible_colors = [i for i in range(1, 10) if i != color_1]
            color_2 = random.choice(possible_colors)

            # Instead of using rows//3 directly, we choose n_colored_rows and n_colored_cols such that
            # the background (color_1) cells outnumber the colored cells (color_2).
            # Painted count = (n_colored_rows + n_colored_cols)*rows - (n_colored_rows*n_colored_cols)
            # Background count = rows**2 - Painted count.
            # We want: background > painted  ⟹  rows**2 > 2 * painted
            max_lines = rows // 3
            while True:
                n_colored_rows = random.randint(1, max_lines)
                n_colored_cols = random.randint(1, max_lines)
                painted = (n_colored_rows + n_colored_cols) * rows - (n_colored_rows * n_colored_cols)
                if rows**2 > 2 * painted:
                    break

            # Helper function to choose n spaced indices from a specified range.
            # This function selects indices from start_offset to (total + start_offset - 1)
            # ensuring that the indices are spaced by at least one unit.
            def choose_spaced_indices(n, total, start_offset):
                choices = sorted(random.sample(range(total - n + 1), n))
                return [choices[i] + i + start_offset for i in range(n)]

            # Exclude the 1st and last row/column (i.e. indices 0 and rows - 1).
            colored_rows = choose_spaced_indices(n_colored_rows, rows - 2, 1)
            colored_cols = choose_spaced_indices(n_colored_cols, rows - 2, 1)

            return {
                'rows': rows,
                'color_1': color_1,
                'color_2': color_2,
                'colored_rows': colored_rows,
                'colored_cols': colored_cols
            }

        def generate_example():
            gridvars = generate_gridvars()
            # Create the input grid using taskvars and the per-example gridvars.
            inp = self.create_input(taskvars, gridvars)
            # Call transform_input with only the input grid.
            out = self.transform_input(inp)
            return {'input': inp, 'output': out}

        # Decide on the number of train and test examples.
        nr_train = random.choice([3, 4])
        nr_test = 1

        train_examples = [generate_example() for _ in range(nr_train)]
        test_examples = [generate_example() for _ in range(nr_test)]
        return taskvars, {'train': train_examples, 'test': test_examples}

