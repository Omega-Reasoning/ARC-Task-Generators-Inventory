# my_arc_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from input_library import create_object, retry, Contiguity
from typing import Dict, Any, Tuple, List

class TaskWzwQqNdpzqKVbhstB6bir3Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain (verbatim from the prompt)
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They only contain several objects, 4-way connected cells of the same color.",
            "The objects are randomly placed in the grid.",
            "The color of these objects can only  {color('object_color1')} and {color('object_color2')}.",
            "The remaining cells are empty (0)."
        ]
        
        # 2) The transformation reasoning chain (verbatim from the prompt)
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and  iterating through each row to find empty (0) cells.",
            "For each empty (0) cell, if it has one or more adjacent non-empty cells (up, down, left, right) of the same color, fill it with that color; otherwise, leave it unchanged."
        ]
        
        # 3) Call super().__init__()
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain.
        The grid must:
          * have a size between 8 and 30 (rows and columns).
          * contain objects in exactly two different colors:
            taskvars['object_color1'] and taskvars['object_color2'].
          * The objects are 4-way connected regions of these colors.
          * Ensure there's at least one scenario where a non-empty cell
            has an adjacent cell of the same color (to ensure the 
            transformation step does something).
        """

        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']

        # Randomly choose grid size
        rows = random.randint(8, 15)
        cols = random.randint(8, 15)

        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Decide how many objects to place
        # We'll place between 2 and 5 objects (random)
        num_objects = random.randint(2, 5)

        # We'll place objects either in color1 or color2, randomly
        for _ in range(num_objects):
            color_chosen = random.choice([object_color1, object_color2])

            # Generate a random smaller object (max dimension about rows//2 or cols//2)
            # so that there's space for empties.
            obj_height = random.randint(1, max(1, rows // 2))
            obj_width = random.randint(1, max(1, cols // 2))

            # Create a 4-way contiguous object in the chosen color
            # We use create_object() with contiguity=FOUR to ensure 4-way connectivity
            object_matrix = create_object(
                height=obj_height,
                width=obj_width,
                color_palette=color_chosen,
                contiguity=Contiguity.FOUR,
                background=0
            )

            # Random top-left position to paste this object
            r_pos = random.randint(0, rows - obj_height)
            c_pos = random.randint(0, cols - obj_width)

            # Paste it onto grid (only overwriting background=0)
            for r_local in range(obj_height):
                for c_local in range(obj_width):
                    if object_matrix[r_local, c_local] != 0:
                        grid[r_pos + r_local, c_pos + c_local] = object_matrix[r_local, c_local]

        # We must ensure there's at least one place where a non-empty cell 
        # has an adjacent cell of the same color. If not, we attempt to fix or re-generate.
        def has_same_color_adjacency(g: np.ndarray) -> bool:
            rmax, cmax = g.shape
            for rr in range(rmax):
                for cc in range(cmax):
                    if g[rr, cc] != 0:
                        color_val = g[rr, cc]
                        # check neighbors
                        for (dr, dc) in [(-1,0), (1,0), (0,-1), (0,1)]:
                            rr2, cc2 = rr + dr, cc + dc
                            if 0 <= rr2 < rmax and 0 <= cc2 < cmax:
                                if g[rr2, cc2] == color_val:
                                    return True
            return False

        # If we fail the adjacency requirement, we can do a quick fix or re-generate.
        # We'll simply do one random modification if there's no adjacency:
        if not has_same_color_adjacency(grid):
            # Let's just add a 2x1 patch in color1 somewhere, ensuring adjacency
            if rows >= 2:
                # pick random location that fits
                r_pos = random.randint(0, rows - 2)
                c_pos = random.randint(0, cols - 1)
                grid[r_pos, c_pos] = object_color1
                grid[r_pos+1, c_pos] = object_color1
            else:
                # worst case scenario: just ensure any two adjacent cells in the same row
                if cols >= 2:
                    r_pos = random.randint(0, rows - 1)
                    c_pos = random.randint(0, cols - 2)
                    grid[r_pos, c_pos] = object_color1
                    grid[r_pos, c_pos+1] = object_color1

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Implementation of the transformation reasoning chain:
        1) Copy the input grid.
        2) For each empty (0) cell, if it has one or more adjacent non-empty cells (up, down, left, right)
           of the same color, fill it with that color; otherwise, leave it unchanged.
        """
        output_grid = grid.copy()
        rows, cols = grid.shape

        
        # Step 1: Pre-check adjacency
        # We want to do this in a single pass: 
        # We gather which cells should be filled, then fill them afterwards to avoid chaining expansions.
        
        to_fill = []
        for r in range(rows):
            for c in range(cols):
                if output_grid[r, c] == 0:
                    # gather non-zero neighbors
                    neighbors = []
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            neighbor_col = grid[rr, cc]
                            if neighbor_col != 0:
                                neighbors.append(neighbor_col)
                    # check if neighbors are all the same color (and not empty)
                    if len(neighbors) > 0 and all(n == neighbors[0] for n in neighbors):
                        # fill with that color
                        to_fill.append((r, c, neighbors[0]))
        
        # Step 2: Fill the cells
        for (r, c, fill_color) in to_fill:
            output_grid[r, c] = fill_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        1) Randomly pick distinct object_color1 and object_color2 in [1..9].
        2) Generate 3-6 train pairs and 1 test pair.
        3) Return the dictionary of { 'object_color1': ..., 'object_color2': ... } 
           and the TrainTestData (train/test pairs).
        """

        # 1) Random distinct color picks
        colors = list(range(1, 10))  # from 1..9
        random.shuffle(colors)
        object_color1 = colors[0]
        object_color2 = colors[1]

        taskvars = {
            'object_color1': object_color1,
            'object_color2': object_color2
        }

        # 2) how many train pairs?
        num_train = random.randint(3, 6)
        # always 1 test pair
        num_test = 1

        # We can use the convenience method create_grids_default 
        # if we do not need special logic for different grid variables.
        # create_grids_default just calls create_input() and transform_input() 
        # for each pair. That’s sufficient for this task.
        train_test_data = self.create_grids_default(num_train, num_test, taskvars)

        return taskvars, train_test_data



