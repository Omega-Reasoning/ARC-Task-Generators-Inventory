import numpy as np
import random
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import *
from input_library import *

class Task09629e4fGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid has size {vars['rows']} X {vars['rows']}.",
            "The grid is a raster with {color('sep_color')} rows and columns as separators.",
            "The first row and column is {color('sep_color')}, i.e. a separator row/column.",
            "Subgrids have a dimension of {vars['subgrid_rows']}x{vars['subgrid_cols']} i.e. every {vars['subgrid_rows']+1} rows there is a separator row and every {vars['subgrid_cols']+1} columns there is a separator column.",
            "Some cells of each subgrid are colored (i.e. cells in them have color between 1-9), and there is exactly one subgrid which has only 4 colored cells having color(between 1-9)."
        ]
        transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Identify the subgrid which has exactly four colored cells in it.",
            "For each colored cell within this subgrid, determine its relative position specified by coordinates (i, j).",
            "Using this relative position, locate the corresponding target subgrid within the overall outputgrid.",
            "Color all cells within the identified target subgrid with the color of the originating cell.",
            "The separator rows and columns of color {color('sep_color')}, remain unchanged.",
            "The remaining cells of the input grid are empty(0)."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an input grid following the input reasoning chain.
        The grid is partitioned into subgrids separated by rows/columns filled with the separator color.
        One randomly chosen subgrid (the special subgrid) gets exactly 4 colored cells (each with a distinct color),
        while all other subgrids get 5 or 6 colored cells (again, each colored cell is unique).
        """
        rows = taskvars['rows']
        sep_color = taskvars['sep_color']
        subgrid_size = taskvars['subgrid_rows']  # note: subgrid_rows == subgrid_cols
        num_subgrids = subgrid_size  # by design, grid size is: rows = (subgrid_size^2 + subgrid_size + 1)
        
        # Create a list of valid colors for subgrid cells, ensuring sep_color is not included.
        valid_colors = [c for c in range(1, 10) if c != sep_color]

        # Initialize the grid with empty values.
        grid = np.zeros((rows, rows), dtype=int)
        
        # Fill in the separator rows and columns.
        for i in range(num_subgrids + 1):
            sep_idx = i * (subgrid_size + 1)
            grid[sep_idx, :] = sep_color
            grid[:, sep_idx] = sep_color

        # Randomly select one subgrid to be the special one.
        special_subgrid = (random.randrange(num_subgrids), random.randrange(num_subgrids))
        
        # For each subgrid, generate colored cells.
        for i in range(num_subgrids):
            for j in range(num_subgrids):
                # Compute the top-left coordinate (just after the separator row/column).
                r0 = i * (subgrid_size + 1) + 1
                c0 = j * (subgrid_size + 1) + 1
                # Initialize a blank subgrid.
                subgrid = np.zeros((subgrid_size, subgrid_size), dtype=int)
                if (i, j) == special_subgrid:
                    num_colors = 4
                else:
                    num_colors = random.choice([5, 6])
                # List all cell positions in the subgrid.
                positions = [(r, c) for r in range(subgrid_size) for c in range(subgrid_size)]
                # Randomly select positions to color.
                chosen_positions = random.sample(positions, num_colors)
                # Choose distinct colors (from valid_colors) for the colored cells.
                colors_sample = random.sample(valid_colors, num_colors)
                for (pos, col) in zip(chosen_positions, colors_sample):
                    subgrid[pos] = col
                # Insert the generated subgrid into the main grid.
                grid[r0:r0+subgrid_size, c0:c0+subgrid_size] = subgrid

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transform the input grid according to the following rules:
        
        - The output grid maintains the same dimensions as the input grid.
        - The separator rows and columns (i.e., those where the row or column index is
          a multiple of subgrid dimension + 1) remain filled with the separator color ({color('sep_color')}).
        - Identify the unique subgrid that contains exactly four colored cells (cells with a nonzero value)
          within its non-separator region.
        - For each colored cell in this special subgrid, determine its relative position (r, c) within the subgrid.
          Then, locate the target subgrid at the position (r, c) in the overall grid and fill every cell within
          that subgrid (except its separators) with the color of the instruction cell.
        - All other cells that are not part of a target subgrid or a separator remain empty (set to 0).
        """
        # Retrieve grid dimensions and subgrid properties.
        n_total = taskvars['rows']
        subgrid_rows = taskvars['subgrid_rows']
        subgrid_cols = taskvars['subgrid_cols']
        sep_color = taskvars['sep_color']

        # Initialize the output grid: set every cell to empty (0).
        output_grid = [[0 for _ in range(n_total)] for _ in range(n_total)]
        
        # Set separator rows and columns to the separator color.
        for i in range(n_total):
            for j in range(n_total):
                if i % (subgrid_rows + 1) == 0 or j % (subgrid_cols + 1) == 0:
                    output_grid[i][j] = sep_color

        # It is assumed that the number of subgrids in each direction equals the subgrid dimensions.
        n_subgrid_rows = subgrid_rows
        n_subgrid_cols = subgrid_cols

        special_subgrid_info = None  # Will store (start_row, start_col, colored_cells)

        # Iterate over all subgrids (non-separator blocks) to identify the special subgrid.
        for sub_i in range(n_subgrid_rows):
            for sub_j in range(n_subgrid_cols):
                # Compute starting indices of the current subgrid.
                start_row = sub_i * (subgrid_rows + 1) + 1
                start_col = sub_j * (subgrid_cols + 1) + 1

                colored_cells = []
                # Collect the cells (and their colors) in the current subgrid that are colored (nonzero).
                for i in range(start_row, start_row + subgrid_rows):
                    for j in range(start_col, start_col + subgrid_cols):
                        if grid[i][j] != 0:
                            colored_cells.append((i, j, grid[i][j]))

                # Identify the special subgrid as the one containing exactly four colored cells.
                if len(colored_cells) == 4:
                    special_subgrid_info = (start_row, start_col, colored_cells)
                    break
            if special_subgrid_info is not None:
                break

        # If no special subgrid is found, return the output grid where non-separator cells are empty.
        if special_subgrid_info is None:
            return output_grid

        special_start_row, special_start_col, colored_cells = special_subgrid_info

        # Process each colored cell in the special subgrid.
        for r, c, cell_color in colored_cells:
            # Determine the relative position (i, j) within the special subgrid.
            rel_r = r - special_start_row
            rel_c = c - special_start_col

            # Map the relative position to the target subgrid indices.
            # The assumption is that the overall grid is divided into subgrids arranged in a 
            # matrix with dimensions subgrid_rows x subgrid_cols.
            target_subgrid_row = rel_r
            target_subgrid_col = rel_c

            # Compute the starting indices of the target subgrid.
            target_start_row = target_subgrid_row * (subgrid_rows + 1) + 1
            target_start_col = target_subgrid_col * (subgrid_cols + 1) + 1

            # Color all cells in the target subgrid (excluding its separators) with the instruction color.
            for i in range(target_start_row, target_start_row + subgrid_rows):
                for j in range(target_start_col, target_start_col + subgrid_cols):
                    output_grid[i][j] = cell_color

        return output_grid

    def create_grids(self) -> tuple:
        """
        Generate the task variables and create a set of training and test grid pairs.
        
        - Task Variables:
            * 'rows': overall grid size (computed as k^2 + k + 1 for k subgrids per row/column).
            * 'sep_color': the separator color (random integer from 1 to 9).
            * 'subgrid_rows' and 'subgrid_cols': dimensions of each subgrid (set equal to k).
        
        - Grid Generation:
            * Create 3–4 training examples and one test example.
            * Each example is generated by first calling create_input() and then transform_input().
        """
        # Choose k from {3, 4} so that grid size is between 11 and 30.
        k = random.choice([3, 4])
        rows = k * k + k + 1  # Ensures rows = k^2 + k + 1 (e.g., 13 for k=3 or 21 for k=4)
        sep_color = random.randint(1, 9)
        taskvars = {
            'rows': rows,
            'sep_color': sep_color,
            'subgrid_rows': k,
            'subgrid_cols': k
        }

        # Randomly select 3 or 4 training examples and 1 test example.
        n_train = random.choice([3, 4])
        n_test = 1

        train_examples = []
        for _ in range(n_train):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            train_examples.append({'input': inp, 'output': out})

        test_examples = []
        for _ in range(n_test):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            test_examples.append({'input': inp, 'output': out})

        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        return taskvars, train_test_data

