from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# We can optionally use functions from the input_library and transformation_library
# inside create_input(), but not in transform_input() for input_library.
from input_library import retry  # not mandatory, but available
# from transformation_library import find_connected_objects, GridObject  # example usage if needed

class TaskMUavpxnb8di7qRxZFhNAa7Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialize the input reasoning chain:
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains a completely filled {color('object_color')} border, with empty (0) interior cells."
        ]
        
        # 2) Initialize the transformation reasoning chain:
        reasoning_chain = [
            "To construct the output grid, copy the input grid and fill in the empty (0) interior cells according to the following rule.",
            "If the number of rows in the grid is even, the empty (0) cells are filled with {color('fill_color1')} color, otherwise they are filled with {color('fill_color2')} color."
        ]
        
        # 3) Call the parent constructor
        super().__init__(observation_chain, reasoning_chain)

    def create_input(self,
                     taskvars: dict,
                     gridvars: dict) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
        1) The grid has some size (rows x cols) in [7..20], with the constraint that:
           - The number of columns is always odd.
           - The grid has a fully filled {object_color} border.
           - The interior is empty (0).
        """
        object_color = taskvars["object_color"]
        
        # We retrieve the chosen size from gridvars (which we set in create_grids()).
        rows = gridvars["rows"]
        cols = gridvars["cols"]
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Fill the border with object_color
        # top and bottom rows
        grid[0, :] = object_color
        grid[-1, :] = object_color
        # left and right columns
        grid[:, 0] = object_color
        grid[:, -1] = object_color
        
        # The interior remains zero as per requirement
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        1) Copy the input grid.
        2) If the number of rows is even, fill empty (0) interior cells with fill_color1,
           else fill with fill_color2.
        """
        fill_color1 = taskvars["fill_color1"]
        fill_color2 = taskvars["fill_color2"]
        
        out_grid = grid.copy()
        
        rows, cols = out_grid.shape
        # Decide which fill color to use
        if rows % 2 == 0:
            fill_color = fill_color1
        else:
            fill_color = fill_color2
        
        # Fill interior cells (excluding border) that are 0
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if out_grid[r, c] == 0:
                    out_grid[r, c] = fill_color
        
        return out_grid

    def create_grids(self) -> (dict, TrainTestData):
        """
        We create 4 training grids and 1 test grid with distinct sizes:
          - All columns must be odd.
          - The rows must be in [7..20], the columns in [7..20].
          - One training example must have an even # of rows, another must have an odd # of rows.
          - The border is filled with object_color, the interior is empty.
          - We fill the interior in transform_input() based on row parity with fill_color1 or fill_color2.
          - object_color, fill_color1, fill_color2 must be distinct in [1..9].

        Returns:
            A tuple:
             1) Dictionary of task variables used in the reasoning chain
             2) TrainTestData with 4 training pairs and 1 test pair
        """
        
        # 1) Randomly pick distinct colors for object_color, fill_color1, fill_color2 from 1..9
        colors = random.sample(range(1, 10), 3)
        object_color = colors[0]
        fill_color1 = colors[1]
        fill_color2 = colors[2]

        # Create a dict of task variables
        taskvars = {
            "object_color": object_color,
            "fill_color1": fill_color1,
            "fill_color2": fill_color2
        }
        
        # 2) We need 5 distinct (rows, cols) pairs:
        #    - columns must be odd
        #    - rows in [7..20], cols in [7..20]
        #    - all columns are odd
        #    - at least one grid with even # of rows, at least one with odd # of rows among the training set
        possible_rows = list(range(7, 21))
        possible_cols = [c for c in range(7, 21) if c % 2 == 1]  # columns must be odd
        random.shuffle(possible_rows)
        random.shuffle(possible_cols)
        
        # We'll pick 5 distinct size pairs
        # We ensure we have at least 1 even row in training and 1 odd row in training
        # We'll attempt a simple approach: gather valid pairs until we get 5 distinct ones
        # with the desired constraints for training sets. Then pick any leftover for test.
        
        chosen_sizes = []
        
        # We need 4 for training, 1 for test
        # We'll keep a simple strategy:
        #   - keep adding random pairs from possible_rows x possible_cols
        #   - ensure no duplicates
        #   - we stop once we have at least 4 pairs that collectively include at least 1 even row and 1 odd row
        while len(chosen_sizes) < 4:
            r = random.choice(possible_rows)
            c = random.choice(possible_cols)
            if (r, c) not in chosen_sizes:
                chosen_sizes.append((r, c))
            # Check if we have at least one even row and one odd row among chosen so far
            rows_mod = [sz[0] % 2 for sz in chosen_sizes]
            if len(chosen_sizes) >= 4 and (0 in rows_mod) and (1 in rows_mod):
                break
        
        # Now choose 1 more for test that is distinct from training
        while True:
            r = random.choice(possible_rows)
            c = random.choice(possible_cols)
            if (r, c) not in chosen_sizes:
                chosen_sizes.append((r, c))
                break
        
        # We'll store them: first 4 are train, last 1 is test
        train_sizes = chosen_sizes[:4]
        test_size = chosen_sizes[4]

        # 3) Create the grids
        train_data = []
        for (r, c) in train_sizes:
            gridvars = {"rows": r, "cols": c}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))
        
        test_data = []
        (r_test, c_test) = test_size
        gridvars_test = {"rows": r_test, "cols": c_test}
        input_grid_test = self.create_input(taskvars, gridvars_test)
        output_grid_test = self.transform_input(input_grid_test, taskvars)
        test_data.append(GridPair(input=input_grid_test, output=output_grid_test))
        
        train_test_data = TrainTestData(train=train_data, test=test_data)
        
        return taskvars, train_test_data

