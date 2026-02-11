from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# Optional but encouraged: we can use these libraries in create_input().
from input_library import retry, random_cell_coloring
from transformation_library import find_connected_objects

class Task5WKhJgUJdkVm7Nc2aU9ZMkGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They only contain {color('cell_color')} and empty cells."
        ]
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and, for each {color('cell_color')} cell, filling its adjacent (left, right, up, down) empty (0) cells with {color('fill_color')} color."
        ]
        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Create the overall task variables (colors to use) and
        produce train/test grids following the specification.
        """
        # Pick distinct colors for cell_color and fill_color
        cell_color = random.randint(1, 9)
        fill_color_options = [c for c in range(1, 10) if c != cell_color]
        fill_color = random.choice(fill_color_options)

        taskvars = {
            "cell_color": cell_color,
            "fill_color": fill_color
        }

        # Random number of train examples between 3 and 6, plus 1 test
        nr_train = random.randint(3, 6)
        nr_test = 1

        # Use the default method for generating train/test pairs
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, train_test_data

    def create_input(self,
                     taskvars,
                     gridvars):
        """
        Create a single input grid. 
        Requirements/Invariants:
          - Grid size between 5x5 and 30x30
          - Grid only has 0 (empty) or taskvars['cell_color']
          - At least 15 empty cells.
          - Must have at least one cell_color cell that has an adjacent empty cell
            so that the transformation can actually change something.
        """
        cell_color = taskvars["cell_color"]

        def generator():
            # Randomly choose grid dimensions
            rows = random.randint(5, 30)
            cols = random.randint(5, 30)
            grid = np.zeros((rows, cols), dtype=int)

            # Randomly color some cells with cell_color, with moderate density
            density = random.uniform(0.1, 0.4)
            random_cell_coloring(grid,
                                 color_palette=cell_color,
                                 density=density,
                                 background=0,
                                 overwrite=False)
            return grid

        def predicate(grid):
            # Ensure at least 15 empty cells
            num_empty = np.sum(grid == 0)
            if num_empty < 15:
                return False

            # Ensure there's at least one cell_color cell that has an adjacent empty cell
            # so the output won't be identical to input.
            # We'll check for any cell of color cell_color that has an empty neighbor.
            rows, cols = grid.shape
            cell_color_coords = np.argwhere(grid == cell_color)
            for r, c in cell_color_coords:
                # Check neighbors (4-directional)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if grid[rr, cc] == 0:
                            # Found a cell_color with an adjacent empty cell
                            return True
            return False

        # Use retry(...) to generate a suitable grid
        valid_grid = retry(generator, predicate, max_attempts=100)
        return valid_grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Copy the input grid and, for each cell_color cell, fill its adjacent
        (up, down, left, right) empty (0) cells with fill_color.
        """
        cell_color = taskvars["cell_color"]
        fill_color = taskvars["fill_color"]

        output = grid.copy()
        rows, cols = output.shape

        # Collect positions of all cell_color cells
        cell_color_positions = np.argwhere(output == cell_color)

        # For each cell_color cell, fill its 4 neighbors if they are empty
        for r, c in cell_color_positions:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and output[rr, cc] == 0:
                    output[rr, cc] = fill_color

        return output

