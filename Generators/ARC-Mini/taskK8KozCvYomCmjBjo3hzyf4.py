from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import np, random
from Framework.transformation_library import find_connected_objects

class TaskK8KozCvYomCmjBjo3hzyf4Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialize the input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input matrix contains exactly two filled cells: {color('cell_color1')} and {color('cell_color2')}.",
            "These cells are positioned at the start and the end of a row or a column.",
            "All other cells are empty (0)."
        ]
        # 2) Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and locating the row or column that contains the {color('cell_color1')} and {color('cell_color2')} cells.",
            "The identified row or column is filled equally with {color('cell_color1')} and {color('cell_color2')} cells, preserving their original order."
        ]
        # 3) Call super().__init__()
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Returns:
            - A dict with task variables used in the template. Here, only cell_color1 and cell_color2.
            - A dict containing 'train' and 'test' keys, each holding lists of GridPair (input/output) dicts.
        """
        # Randomly choose distinct colors between 1..9
        color1 = random.randint(1, 9)
        color2 = random.randint(1, 9)
        while color2 == color1:
            color2 = random.randint(1, 9)

        taskvars = {
            'cell_color1': color1,
            'cell_color2': color2
        }

        # Number of training examples: randomly between 3 and 6
        n_train = random.randint(3, 6)

        # Generate distinct even sizes (>=4 and <=30) for each example
        all_even_sizes = [s for s in range(4, 31, 2)]  # 4,6,8,...,30
        random.shuffle(all_even_sizes)
        used_sizes = all_even_sizes[: n_train + 1]  # for train + test

        train_examples = []
        for i in range(n_train):
            grid_size = used_sizes[i]
            input_grid = self.create_input(taskvars, {'size': grid_size})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        # Create one test example, using a distinct size
        test_size = used_sizes[n_train]
        test_input = self.create_input(taskvars, {'size': test_size})
        test_output = self.transform_input(test_input, taskvars)

        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars):
        """
        Create an input grid of size NxN (where N is even and 3 <= N <= 30).
        Exactly two cells are filled with cell_color1 and cell_color2, positioned
        at the start and end of either the same row or the same column.
        """
        size = gridvars['size']  # NxN
        grid = np.zeros((size, size), dtype=int)

        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']

        # Randomly decide whether to place cells in a row or a column
        place_in_row = random.choice([True, False])

        if place_in_row:
            # Pick a row
            r = random.randrange(size)
            # Place color1 at (r, 0) and color2 at (r, size-1)
            grid[r, 0] = color1
            grid[r, size - 1] = color2
        else:
            # Pick a column
            c = random.randrange(size)
            # Place color1 at (0, c) and color2 at (size-1, c)
            grid[0, c] = color1
            grid[size - 1, c] = color2

        return grid

    def transform_input(self, grid, taskvars):
        """
        Transformation:
          1) Copy the grid.
          2) Find the row or column that has both cell_color1 and cell_color2.
          3) Fill that entire row or column equally with the two colors, preserving their order.
        """
        color1 = taskvars['cell_color1']
        color2 = taskvars['cell_color2']
        rows, cols = grid.shape

        out_grid = grid.copy()

        # Find positions of color1 and color2
        positions_color1 = np.argwhere(out_grid == color1)
        positions_color2 = np.argwhere(out_grid == color2)

        if len(positions_color1) == 0 or len(positions_color2) == 0:
            # Should never happen if input is correct, but just in case:
            return out_grid

        r1, c1 = positions_color1[0]
        r2, c2 = positions_color2[0]

        # Check if they're in the same row or same column
        if r1 == r2:
            # Same row
            row_idx = r1
            # Determine which color is left vs. right
            if c1 < c2:
                left_color, right_color = color1, color2
            else:
                left_color, right_color = color2, color1

            # Fill half with left_color, half with right_color
            # Example: if row length is L = cols, fill first L//2 with left_color
            # and the rest with right_color. For an even dimension, L//2 is exactly half.
            half = cols // 2
            out_grid[row_idx, :half] = left_color
            out_grid[row_idx, half:] = right_color

        elif c1 == c2:
            # Same column
            col_idx = c1
            # Determine which color is top vs. bottom
            if r1 < r2:
                top_color, bottom_color = color1, color2
            else:
                top_color, bottom_color = color2, color1

            # Fill half the column with top_color, half with bottom_color
            half = rows // 2
            out_grid[:half, col_idx] = top_color
            out_grid[half:, col_idx] = bottom_color

        # If not the same row or column (shouldn't happen in this task), we do nothing special.
        return out_grid


