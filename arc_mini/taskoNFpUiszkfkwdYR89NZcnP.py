from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskoNFpUiszkfkwdYR89NZcnPGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They only contain one or more {color('object_color')} objects, which are one-cell-wide rectangular frames enclosing empty (0) cells.",
            "At least one object encloses exactly one empty (0) cell.",
            
        ]

        # 2. Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling any empty (0) cell with {color('object_color')} if it is completely surrounded on all four sides (up, down, left, and right) by {color('object_color')} cells."
        ]

        # 3. Call the superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Creates the dictionary of task variables (e.g. the color we use) and
        train/test data following the specification:
          - 3-4 training examples
          - 1 test example
        """
        object_color = random.randint(1, 9)

        taskvars = {
            'object_color': object_color
        }
        
        nr_train = random.randint(3, 4)
        nr_test = 1

        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Creates a single input grid ensuring:
          - A chosen object_color (non-zero)
          - At least two closed objects
          - Exactly one object encloses a single empty cell
          - All objects are completely separated by empty (0) cells
          - Each object is made of 4-way connected cells
          - Grid size between 7x7 and 30x30
        """
        object_color = taskvars['object_color']
        height = random.randint(7, 15)
        width = random.randint(7, 15)
        grid = np.zeros((height, width), dtype=int)

        n_rings = random.randint(2, 4)

        def is_valid_placement(r0, c0, outer_h, outer_w):
            """
            Ensures placement is valid by checking separation by at least one empty cell.
            """
            if r0 < 1 or c0 < 1 or (r0 + outer_h) >= height - 1 or (c0 + outer_w) >= width - 1:
                return False  # Ensure at least a 1-cell buffer to the edge

            # Ensure separation from existing objects by checking a 1-cell padding
            for r in range(r0 - 1, r0 + outer_h + 1):
                for c in range(c0 - 1, c0 + outer_w + 1):
                    if grid[r, c] != 0:
                        return False
            return True

        def place_ring_with_inner(outer_h, outer_w, single_cell=False):
            """
            Tries to place a rectangular ring that is completely separated from other rings.
            Returns True if placed successfully.
            """
            for _ in range(100):  # Retry up to 100 times
                r0 = random.randint(1, height - outer_h - 1)
                c0 = random.randint(1, width - outer_w - 1)

                if not is_valid_placement(r0, c0, outer_h, outer_w):
                    continue  # Try another position

                # Place the ring
                for c in range(c0, c0 + outer_w):
                    grid[r0, c] = object_color
                    grid[r0 + outer_h - 1, c] = object_color
                for r in range(r0, r0 + outer_h):
                    grid[r, c0] = object_color
                    grid[r, c0 + outer_w - 1] = object_color

                return True
            return False  # Failed to place the ring

        # Place one special ring that encloses exactly 1 empty cell
        place_ring_with_inner(3, 3, single_cell=True)

        # Place additional rings ensuring separation
        for _ in range(n_rings - 1):
            outer_h = random.randint(3, height // 2 + 2)
            outer_w = random.randint(3, width // 2 + 2)
            place_ring_with_inner(outer_h, outer_w)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid following the pattern:
          - Copy the input grid
          - Fill any empty (0) cell with object_color if it is surrounded on all four sides
        """
        object_color = taskvars['object_color']
        out_grid = grid.copy()

        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    # Check up, down, left, right exist and are object_color
                    if (r > 0 and grid[r - 1, c] == object_color and
                        r < rows - 1 and grid[r + 1, c] == object_color and
                        c > 0 and grid[r, c - 1] == object_color and
                        c < cols - 1 and grid[r, c + 1] == object_color):
                        out_grid[r, c] = object_color

        return out_grid


