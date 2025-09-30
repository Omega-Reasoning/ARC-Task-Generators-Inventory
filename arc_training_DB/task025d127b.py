import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects

class ARCTask025d127bGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input Reasoning Chain
        observation_chain = [
            "Input matrices can have different sizes.",
            "They contain one or two objects of different colors.",
            "The objects consist of two L-shapes and a diagonal connection between the end of the L-shapes.",
            "An L-shape consists of a horizontal component having at least 3 cells and a vertical component with two cells.",
            "There is an upper L-shape and a lower L-shape.",
            "For the upper L-shape the vertical component is formed by adding a cell below the start of the horizontal component. For the lower L-shape, the vertical component is formed by adding a cell at the end going upwards.",
            "The two diagonal connectors between the L-shapes connect (i) the vertical extension of the upper L-shape with the horizontal end of the lower L-shape and (ii) the horizontal end of the upper L-shape with the vertical extension of the lower L-shape."
        ]

        # 2. Transformation Reasoning Chain
        reasoning_chain = [
            "The output matrix is constructed by using a matrix of the same size as the input matrix.",
            "First get all 8-way connected objects.",
            "Each object has a lower \"L-shape\" which consists of a set of horizontal cells in its bottom most row and a one cell vertical extension at its rightmost cell.",
            "For each object, (i) the lower L-shape is copied into the output matrix and (ii) all other cells are moved by one cell to the right."
        ]

        # 3. Call the parent constructor
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Creates 3-6 training pairs and 1 test pair.
        Returns a dictionary of variables (if needed) and a TrainTestData dict.
        """
        taskvars = {}
        return taskvars, self.create_grids_default(random.randint(3, 6), 1, taskvars)

    def create_input(self,
                    taskvars: Dict[str, Any],
                    gridvars: Dict[str, Any]) -> np.ndarray:

        # Helper to draw a diagonal line of `steps` increments (row+1, col+1),
        # starting at (r, c). This returns the final (row, col) reached.
        def draw_diagonal(r, c, steps, color, mat):
            mat[r, c] = color
            for _ in range(steps):
                r += 1
                c += 1
                mat[r, c] = color
            return (r, c)

        # Decide how many objects (1 or 2) by 50% chance
        num_objects = 1 if random.random() < 0.5 else 2

        # We'll store all parameter sets (row_upper, delta_rows, h, col_upper, color)
        # so we can compute the final matrix size afterwards.
        objects_params = []

        def create_object_params(min_row_upper=0):
            """
            Create one object's parameters:
            - row_upper in [min_row_upper.. min_row_upper+5] (capped at 5 if first object).
            - delta_rows in [2..8]
            - horizontal_len (h) in [3..7]
            - col_upper in [0..7]
            - color in [1..9]
            """
            row_upper = random.randint(min_row_upper, min_row_upper + 5)
            delta_rows = random.randint(2, 8)
            horizontal_len = random.randint(3, 7)
            col_upper = random.randint(0, 7)
            color = random.randint(1, 9)
            return {
                "row_upper": row_upper,
                "delta_rows": delta_rows,
                "h": horizontal_len,
                "col_upper": col_upper,
                "color": color
            }

        # Create the first object's parameters
        obj1 = create_object_params(0)  # 0..5 for row_upper
        objects_params.append(obj1)

        if num_objects == 2:
            # Place second object below the first.
            min_row_for_next = obj1["row_upper"] + obj1["delta_rows"] + 2
            obj2 = create_object_params(min_row_for_next)
            objects_params.append(obj2)

        # --- Determine matrix size based on all objects we need to place in it ---
        max_bottom = 0
        max_right = 0
        for obj in objects_params:
            row_u = obj["row_upper"]
            dr = obj["delta_rows"]
            h = obj["h"]
            col_u = obj["col_upper"]
            # The "lowest" row used = row_u + dr
            if row_u + dr > max_bottom:
                max_bottom = row_u + dr
            # Rightmost column:
            #   - upper shape ends at col_u + h - 1
            #   - diagonals can add up to (dr - 1) or so
            #   - then the final lower horizontal line extends from (col_u + dr - 1) to (col_u + (h - 1) + dr)
            #   So let's approximate:
            rightmost = col_u + (h - 1) + dr
            if rightmost > max_right:
                max_right = rightmost

        # Add a small random extension [0..5] to both dimensions
        rows = max_bottom + 1 + random.randint(0, 5) 
        cols = max_right + 1 + random.randint(0, 5)

        matrix = np.zeros((rows, cols), dtype=int)

        # --- Fill each object into the matrix ---
        for obj in objects_params:
            r_u = obj["row_upper"]
            dr = obj["delta_rows"]
            h = obj["h"]
            c_u = obj["col_upper"]
            color = obj["color"]

            # 1) Draw the upper L-shape:
            #    Horizontal line 
            for cc in range(c_u, c_u + h):
                matrix[r_u, cc] = color
                # Single vertical extension  
                matrix[r_u + 1, c_u] = color

            # 2) Diagonal from the vertical extension: do (dr - 1) steps of (row+1, col+1)
            #    start is (r_u+1, c_u)
            vert_end_final = draw_diagonal(r_u + 1, c_u, dr - 1, color, matrix)

            # 3) Diagonal from the upper horizontal end: (r_u, c_u + h - 1)
            horiz_end_final = draw_diagonal(r_u, c_u + h - 1, dr - 1, color, matrix)

            # 4) Draw the lower horizontal line for the L-shape at row = r_u + dr,
            #    from the left diagonal's final col to the right diagonal's final col.
            r_low = r_u + dr
            left_col = min(vert_end_final[1], horiz_end_final[1])
            right_col = max(vert_end_final[1], horiz_end_final[1])
            # Fill horizontally
            for cc in range(left_col, right_col + 1):
                if cc < cols:
                    matrix[r_low, cc] = color

        return matrix

    def transform_input(self,
                        grid: np.ndarray,
                        taskvars: dict) -> np.ndarray:
        rows, cols = grid.shape
        output = np.zeros((rows, cols), dtype=int)

        # 1) Get all 8-way connected objects
        objects = find_connected_objects(grid, True)

        for object in objects:
            obj = object.coords
            # 2) Identify the bottommost row
            max_r = max(r for (r, c) in obj)
            # Get all cells that lie in that bottommost row
            bottom_cells = [(r, c) for (r, c) in obj if r == max_r]

            # 3) The "lower L-shape" = bottom row + the one-cell vertical extension
            #    at the row's rightmost cell. Let's find the rightmost cell in bottom row:
            rightmost_cell = max(bottom_cells, key=lambda x: x[1])  # largest c in that row

            # The vertical extension cell is (row_above, col_rightmost)
            # if that cell is in the object. We check if it actually exists in obj.
            r_above = rightmost_cell[0] - 1
            c_above = rightmost_cell[1]
            lower_l_shape = set(bottom_cells)  # those are in bottom row

            if (r_above, c_above) in obj:
                lower_l_shape.add((r_above, c_above))

            # 4) Copy the lower L-shape into 'output' at the same positions
            for (r, c) in lower_l_shape:
                output[r, c] = grid[r, c]

            # 5) All other cells in the object are moved one cell to the right
            other_cells = obj - lower_l_shape
            for (r, c) in other_cells:
                output[r, c + 1] = grid[r, c]

        return output

