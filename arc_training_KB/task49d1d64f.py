import numpy as np
import random

# Required imports from your framework
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData

# Optional but allowed, though we only lightly use them here
from input_library import retry, create_object, Contiguity
from transformation_library import find_connected_objects

class Task49d1d64fGenerator(ARCTaskGenerator):

    def __init__(self):
        # 1) The input reasoning chain (exactly as given)
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain a multi-colored grid that only includes empty (0) cells when the grid has more than two rows or columns.",
            "The grid consists of the following possible colors: {color('cell_color1')}, {color('cell_color2')}, {color('cell_color3')}, {color('cell_color4')}, and {color('cell_color5')}.",
            "If the grid has more than two rows and columns, all interior cells are empty, with only the border filled with multi-colored cells.",
            "Each colored cell must be 4-way connected to a differently colored cell."
        ]

        # 2) The transformation reasoning chain (exactly as given)
        transformation_reasoning_chain = [
            "The output grids are larger than the input grids.",
            "They always have two more rows and two more columns than the input grids.",
            "They are constructed by expanding the four corner cells from the input grids.",
            "Each corner cell expands into adjacent empty (0) cells: the top-left expands left and upward, the top-right expands right and upward, the bottom-left expands left and downward, and the bottom-right expands right and downward.",
            "The non-corner cells in the first row of the input grid expand one cell upward, while those in the last row expand one cell downward.",
            "Similarly, the non-corner cells in the first column expand one cell to the left, while those in the last column expand one cell to the right.",
            "If there are any interior cells in the input grid, they remain unchanged in the output grid.",
            "This expansion results in the output grid having four empty (0) corner cells."
        ]

        # 3) Call super constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the constraints:
          - The grid has 'height' and 'width' (from gridvars) between 1..15.
          - Uses exactly 5 distinct colors (cell_color1..cell_color5) in the range [1..9].
          - If height>2 and width>2, only the border is colored, and interior is 0.
          - No two 4-way adjacent colored cells share the same color.
          - If height <=2 or width <=2, all cells are colored (since there's no interior).
        """
        height = gridvars["height"]
        width = gridvars["width"]
        colors = [
            taskvars["cell_color1"],
            taskvars["cell_color2"],
            taskvars["cell_color3"],
            taskvars["cell_color4"],
            taskvars["cell_color5"]
        ]

        grid = np.zeros((height, width), dtype=int)

        # Helper to pick a color that differs from the top and left neighbors
        def pick_different_color(r, c):
            used = set()
            # up neighbor
            if r > 0:
                used.add(grid[r-1, c])
            # left neighbor
            if c > 0:
                used.add(grid[r, c-1])
            # pick from colors that differ from all 'used'
            valid = [col for col in colors if col not in used and col != 0]
            return random.choice(valid)

        if height <= 2 or width <= 2:
            # All cells are "border," so fill them all with no two adjacent the same
            for r in range(height):
                for c in range(width):
                    grid[r, c] = pick_different_color(r, c)
        else:
            # Fill only the border
            # Top row
            for c in range(width):
                grid[0, c] = pick_different_color(0, c)
            # Bottom row
            if height > 1:
                for c in range(width):
                    grid[height-1, c] = pick_different_color(height-1, c)
            # Left and right columns in the interior
            for r in range(1, height-1):
                # left column
                grid[r, 0] = pick_different_color(r, 0)
                # right column
                grid[r, width-1] = pick_different_color(r, width-1)
            # interior remains 0

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict = None) -> np.ndarray:
        """
        Transforms the input grid to the output grid following the transformation
        reasoning chain:
          - Output is (height+2) x (width+2).
          - The input cell (r,c) is normally placed at (r+1,c+1) in the output.
          - Corners expand outward/up/down, etc.
          - Edges (but not corners) expand one cell outward/up/down.
          - The four corners of the output remain empty (0).
        """
        h, w = grid.shape
        out_grid = np.zeros((h+2, w+2), dtype=int)

        for r in range(h):
            for c in range(w):
                color = grid[r, c]
                if color == 0:
                    continue
                # Base position in output: shift by (1,1)
                out_grid[r+1, c+1] = color

                # Check for corners
                top_row = (r == 0)
                bot_row = (r == h-1)
                left_col = (c == 0)
                right_col = (c == w-1)

                # Expand corners
                if top_row and left_col:
                    # Expand up => (0, c+1), left => (r+1, 0)
                    out_grid[0, c+1] = color
                    out_grid[r+1, 0] = color

                elif top_row and right_col:
                    # Expand up => (0, c+1), right => (r+1, c+2)
                    out_grid[0, c+1] = color
                    out_grid[r+1, c+2] = color

                elif bot_row and left_col:
                    # Expand down => (r+2, c+1), left => (r+1, 0)
                    out_grid[r+2, c+1] = color
                    out_grid[r+1, 0] = color

                elif bot_row and right_col:
                    # Expand down => (r+2, c+1), right => (r+1, c+2)
                    out_grid[r+2, c+1] = color
                    out_grid[r+1, c+2] = color

                # Expand edges (non-corner)
                else:
                    # top row but not corner => expand up
                    if top_row and (not left_col) and (not right_col):
                        out_grid[r, c+1] = color
                    # bottom row but not corner => expand down
                    if bot_row and (not left_col) and (not right_col):
                        out_grid[r+2, c+1] = color
                    # left column but not corner => expand left
                    if left_col and (not top_row) and (not bot_row):
                        out_grid[r+1, c] = color
                    # right column but not corner => expand right
                    if right_col and (not top_row) and (not bot_row):
                        out_grid[r+1, c+2] = color

        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        Create 3-4 train grids (we pick 3 here) plus 1 test grid, all with different sizes.
        Then instantiate the color variables to distinct values in [1..9].
        Return (taskvars, train_test_data).
        """

        # 1) Pick 5 distinct colors from [1..9]
        color_candidates = list(range(1, 10))  # 1..9
        random.shuffle(color_candidates)
        chosen_colors = color_candidates[:5]

        taskvars = {
            "cell_color1": chosen_colors[0],
            "cell_color2": chosen_colors[1],
            "cell_color3": chosen_colors[2],
            "cell_color4": chosen_colors[3],
            "cell_color5": chosen_colors[4],
        }

        # 2) Pick 4 distinct sizes for 3 training + 1 test
        #    Each size is random in [1..15], but we ensure no duplicates
        #    (We’ll generate 4 distinct (height, width) pairs).
        possible_dims = []
        while len(possible_dims) < 4:
            h = random.randint(2, 15)
            w = random.randint(2, 15)
            if (h, w) not in possible_dims:
                possible_dims.append((h, w))

        # 3) Build train/test sets
        train_pairs = []
        for i in range(3):
            (h, w) = possible_dims[i]
            input_grid = self.create_input(taskvars, {"height": h, "width": w})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({
                "input": input_grid,
                "output": output_grid
            })

        # Test example
        (h, w) = possible_dims[3]
        test_input = self.create_input(taskvars, {"height": h, "width": w})
        test_output = self.transform_input(test_input, taskvars)

        train_test_data = {
            "train": train_pairs,
            "test": [
                {
                    "input": test_input,
                    "output": test_output
                }
            ]
        }

        return taskvars, train_test_data


