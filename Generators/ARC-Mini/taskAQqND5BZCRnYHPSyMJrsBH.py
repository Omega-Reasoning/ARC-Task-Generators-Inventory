import numpy as np
import random
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from typing import Dict, Any, Tuple, List

class TaskAQqND5BZCRnYHPSyMJrsBHGenerator(ARCTaskGenerator):
    """
    ARC Task Generator that creates checkerboard pattern grids of size n x n,
    with n odd and between 5 and 30, and a chosen cell_color between 1 and 9.
    The transformation is simply swapping empty (0) and cell_color cells.
    """

    def __init__(self):
        # 1) The input reasoning chain:
        input_reasoning_chain = [
            "All input grids are squares.",
            "Let n be the height and width of an input grid, where n is an odd number between 5 and 30.",
            "Each input grid has a checkerboard pattern with alternating {color('cell_color')} and empty (0) cells in each row and column."
        ]

        # 2) The transformation reasoning chain:
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid, then replace all empty (0) cells with {color('cell_color')} cells and {color('cell_color')} cells with empty (0) cells."
        ]

        # 3) Call the superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Create 3-4 train input/output pairs and 1 test input/output pair.
        
        - We choose a random cell_color in [1..9].
        - We generate distinct odd sizes n for each grid (between 5 and 30).
        - We ensure at least one grid starts with 0 in the top-left corner
          and at least one grid starts with cell_color in the top-left corner.
        """

        # Step 1: Choose the color for the checkerboard.
        cell_color = random.randint(1, 9)

        # Step 2: Decide how many train examples (3 or 4).
        nr_train = random.choice([3, 4])
        nr_test = 1
        total_grids = nr_train + nr_test

        # Step 3: Generate distinct odd sizes for each grid (5 to 30).
        possible_odd_sizes = [n for n in range(5, 31) if n % 2 == 1]
        random.shuffle(possible_odd_sizes)
        chosen_sizes = possible_odd_sizes[:total_grids]

        # We must ensure:
        #   - At least one grid has top-left cell=0
        #   - At least one grid has top-left cell=cell_color

        # Step 4: Build all train grids
        train_pairs = []

        # Force the first train grid to start with 0
        gridvars_1 = {
            "n": chosen_sizes[0],
            "start_with_color": False  # means top-left is 0
        }
        input_grid_1 = self.create_input({"cell_color": cell_color}, gridvars_1)
        output_grid_1 = self.transform_input(input_grid_1, {"cell_color": cell_color})
        train_pairs.append(GridPair(input=input_grid_1, output=output_grid_1))

        if nr_train > 1:
            # Force the second train grid to start with cell_color
            gridvars_2 = {
                "n": chosen_sizes[1],
                "start_with_color": True
            }
            input_grid_2 = self.create_input({"cell_color": cell_color}, gridvars_2)
            output_grid_2 = self.transform_input(input_grid_2, {"cell_color": cell_color})
            train_pairs.append(GridPair(input=input_grid_2, output=output_grid_2))

            # For the remaining train grids (if any), randomize start.
            for i in range(2, nr_train):
                gridvars_rest = {
                    "n": chosen_sizes[i],
                    "start_with_color": random.choice([True, False])
                }
                input_grid_rest = self.create_input({"cell_color": cell_color}, gridvars_rest)
                output_grid_rest = self.transform_input(input_grid_rest, {"cell_color": cell_color})
                train_pairs.append(GridPair(input=input_grid_rest, output=output_grid_rest))

        # Step 5: Build the test grid
        test_pairs = []
        test_gridvars = {
            "n": chosen_sizes[-1],
            "start_with_color": random.choice([True, False])
        }
        input_grid_test = self.create_input({"cell_color": cell_color}, test_gridvars)
        output_grid_test = self.transform_input(input_grid_test, {"cell_color": cell_color})
        test_pairs.append(GridPair(input=input_grid_test, output=output_grid_test))

        # Step 6: Return the final dictionary of variables plus train/test data
        taskvars = {
            "cell_color": cell_color,
        }

        train_test_data = TrainTestData(train=train_pairs, test=test_pairs)
        return taskvars, train_test_data

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> List[List[int]]:
        """
        Create a checkerboard pattern of size n x n, where n is odd and between 5 and 30.
        Returns a Python list for JSON compatibility.
        """
        cell_color = taskvars["cell_color"]
        n = gridvars["n"]
        start_with_color = gridvars["start_with_color"]

        grid = np.zeros((n, n), dtype=int)

        for r in range(n):
            for c in range(n):
                val = (cell_color if ((r + c) % 2 == 0) else 0)
                if not start_with_color:
                    val = 0 if val == cell_color else cell_color
                grid[r, c] = val

        return grid.tolist()  # Convert numpy array to Python list

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Swap cell_color and 0 in the grid. Works with Python lists.
        """
        cell_color = taskvars["cell_color"]
        grid = np.array(grid)  # Convert to numpy for easier manipulation

        mask_color = (grid == cell_color)
        mask_zero = (grid == 0)

        grid[mask_color] = 0
        grid[mask_zero] = cell_color

        return grid.tolist()  # Convert back to Python list

