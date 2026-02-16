import numpy as np
import random

# Required imports from the framework:
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
# We may use functions from the transformation library to detect/transform objects
from Framework.transformation_library import find_connected_objects
# We use randomization utilities from the input library to ensure variety
# though here we implement custom placement logic for 2x2 squares.
# (We do not use any advanced object creation from Framework.input_library here, but we could if needed.)

class TaskdzzzsJKLGctvvGyimTeWbTGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain one {color('object_color1')} and one {color('object_color2')} 2x2 square object, while all other cells remain empty (0).",
            "The 2x2 square objects may sometimes be 4-way connected to each other."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling all empty (0) cells with {color('object_color3')} color.",
            "If the 2x2 square objects are 4-way connected to each other, change the color of the second column of the {color('object_color1')} object to {color('object_color4')}, and the first column of the {color('object_color2')} object to {color('object_color5')}."
        ]
        
        # 3) Call super().__init__ (the base class constructor)
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        We want to produce:
          - A set of 3-4 training examples
          - A set of 2 test examples
          - Ensure that at least one train example has the squares 4-way connected,
            and at least one has them separated.
          - Similarly for the test set: one connected, one separated.
          - We also ensure the color variables are distinct and chosen in [1..9].
        """
        
        # Pick distinct colors for object_color1, object_color2, object_color3, object_color4, object_color5
        distinct_colors = random.sample(range(1, 10), 5)
        taskvars = {
            'object_color1': distinct_colors[0],
            'object_color2': distinct_colors[1],
            'object_color3': distinct_colors[2],
            'object_color4': distinct_colors[3],
            'object_color5': distinct_colors[4],
        }

        # We will create 4 train examples:
        #   2 of them with connected squares, 2 with separated squares
        # Then 2 test examples:
        #   1 with connected squares, 1 with separated squares
        
        # We'll fix the pattern (connected, separated, connected, separated) for training,
        # and (connected, separated) for test, but randomize the grid sizes/positions within those constraints.
        train_connectivity = [True, False, True, False]
        test_connectivity = [True, False]
        
        train_data = []
        for conn in train_connectivity:
            inp = self.create_input(taskvars, {'connected': conn})
            outp = self.transform_input(inp, taskvars)
            train_data.append(GridPair(input=inp, output=outp))
        
        test_data = []
        for conn in test_connectivity:
            inp = self.create_input(taskvars, {'connected': conn})
            outp = self.transform_input(inp, taskvars)
            test_data.append(GridPair(input=inp, output=outp))

        return taskvars, {
            'train': train_data,
            'test': test_data
        }

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid containing exactly:
          - one 2x2 square of color object_color1
          - one 2x2 square of color object_color2
          - the rest of cells are 0 (empty)
        The squares may or may not be 4-way connected, depending on gridvars['connected'].
        We randomize the grid size in [5..10] rows/columns for variety, ensuring we remain within [5..30].
        We also randomize the positions, with constraints that:
          - If connected=True, the 2x2 squares share at least one edge (4-way adjacency).
          - If connected=False, the squares are separated so that no cell is 4-way adjacent.
        """
        connected = gridvars.get('connected', False)
        
        # Randomly choose grid size between 5 and 10
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        
        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        
        # We define a helper to place one 2x2 square in the grid at (top_r, left_c)
        def place_2x2(r, c, color):
            grid[r, c] = color
            grid[r, c+1] = color
            grid[r+1, c] = color
            grid[r+1, c+1] = color
        
        # Randomly place the first 2x2
        # possible top-left corners: row in [0..rows-2], col in [0..cols-2]
        r1 = random.randint(0, rows - 2)
        c1 = random.randint(0, cols - 2)
        place_2x2(r1, c1, color1)
        
        if connected:
            # We want the second 2x2 to share an edge with the first one.
            # The first 2x2 occupies (r1,c1), (r1,c1+1), (r1+1,c1), (r1+1,c1+1).
            # We can try placing the second one above, below, left, or right of that shape if there's space.
            
            # Collect possible positions that ensure adjacency along an edge:
            candidates = []
            # Attempt placing second square directly above the first (r1-2, c1)
            # so that (r1-1, c1..c1+1) touches (r1, c1..c1+1).
            if r1 - 2 >= 0:
                candidates.append((r1 - 2, c1))
            # Attempt placing second square directly below
            if r1 + 2 < rows - 1:
                candidates.append((r1 + 2, c1))
            # Attempt placing second square to the left
            if c1 - 2 >= 0:
                candidates.append((r1, c1 - 2))
            # Attempt placing second square to the right
            if c1 + 2 < cols - 1:
                candidates.append((r1, c1 + 2))
            
            random.shuffle(candidates)
            
            placed = False
            for (r2, c2) in candidates:
                # Place if the region is empty
                if self._can_place_2x2(grid, r2, c2):
                    place_2x2(r2, c2, color2)
                    placed = True
                    break
            
            # If none of the straightforward adjacency positions worked (rare in small grids),
            # just place second square forcibly in an overlapping-edge manner by random search
            # until we find a 4-way adjacency (fallback).
            if not placed:
                self._place_2x2_connected_fallback(grid, color2, r1, c1)
        
        else:
            # We want the second 2x2 to be placed so it is NOT 4-way adjacent to the first 2x2.
            # We'll try random positions until we find a valid separation (within some attempts).
            attempts = 100
            placed = False
            for _ in range(attempts):
                r2 = random.randint(0, rows - 2)
                c2 = random.randint(0, cols - 2)
                if self._can_place_2x2(grid, r2, c2):
                    # Check adjacency:
                    if not self._is_2x2_connected(r1, c1, r2, c2):
                        place_2x2(r2, c2, color2)
                        placed = True
                        break
            
            # If we didn't manage to place it, we forcibly place it anywhere
            # that is not overlapping. This might rarely cause 4-way adjacency
            # if the grid is too small, but we do our best to keep them separate.
            if not placed:
                for rr in range(rows - 1):
                    for cc in range(cols - 1):
                        if self._can_place_2x2(grid, rr, cc):
                            # If we can't avoid adjacency, place anyway 
                            # (the worst fallback).
                            place_2x2(rr, cc, color2)
                            placed = True
                            break
                    if placed:
                        break
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Implements the transformation:
        1) Fill all empty (0) cells with object_color3.
        2) If the 2x2 squares (color1, color2) are 4-way connected, 
           change the second column of the color1 square to color4,
           and the first column of the color2 square to color5.
        """
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        color3 = taskvars['object_color3']
        color4 = taskvars['object_color4']
        color5 = taskvars['object_color5']
        
        # 1) Copy the input and fill empty with color3
        out_grid = grid.copy()
        out_grid[out_grid == 0] = color3
        
        # 2) Detect if the squares are 4-way connected. 
        #    We'll quickly gather the positions of color1 and color2 from the input grid.
        #    Then check if any color1 cell is 4-adjacent to any color2 cell.
        
        coords_color1 = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])
                         if grid[r, c] == color1]
        coords_color2 = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])
                         if grid[r, c] == color2]
        
        # Determine adjacency:
        is_connected = False
        for (r1, c1) in coords_color1:
            for (r2, c2) in coords_color2:
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    # Found 4-way adjacency
                    is_connected = True
                    break
            if is_connected:
                break
        
        if is_connected:
            # Recolor second column of color1's 2x2 block to color4
            # Recolor first column of color2's 2x2 block to color5
            
            # We do this by bounding boxes for each 2x2 object. 
            # (We expect exactly 4 cells of color1 and 4 cells of color2.)
            # We'll find min/max row/col for color1, color2.
            
            if coords_color1:
                min_r1 = min(r for r, _ in coords_color1)
                max_r1 = max(r for r, _ in coords_color1)
                min_c1 = min(c for _, c in coords_color1)
                max_c1 = max(c for _, c in coords_color1)
                
                # For a 2x2 object, second column is the max column
                # We'll recolor all cells in that column for color1
                for rr in range(min_r1, max_r1+1):
                    for cc in range(min_c1, max_c1+1):
                        if cc == max_c1:  # second column
                            out_grid[rr, cc] = color4
            
            if coords_color2:
                min_r2 = min(r for r, _ in coords_color2)
                max_r2 = max(r for r, _ in coords_color2)
                min_c2 = min(c for _, c in coords_color2)
                max_c2 = max(c for _, c in coords_color2)
                
                # For a 2x2 object, first column is the min column
                # We'll recolor all cells in that column for color2
                for rr in range(min_r2, max_r2+1):
                    for cc in range(min_c2, max_c2+1):
                        if cc == min_c2:  # first column
                            out_grid[rr, cc] = color5
        
        return out_grid

    ############################################################################
    # Helper methods (internal) for create_input
    ############################################################################
    def _can_place_2x2(self, grid: np.ndarray, top_r: int, left_c: int) -> bool:
        """
        Check if we can place a 2x2 square at (top_r, left_c) in 'grid',
        i.e. if those 4 cells are all zero and inside the bounds.
        """
        rows, cols = grid.shape
        if top_r < 0 or top_r + 1 >= rows or left_c < 0 or left_c + 1 >= cols:
            return False
        if (grid[top_r, left_c] != 0 or
            grid[top_r, left_c+1] != 0 or
            grid[top_r+1, left_c] != 0 or
            grid[top_r+1, left_c+1] != 0):
            return False
        return True

    def _is_2x2_connected(self, r1, c1, r2, c2) -> bool:
        """
        Check if the 2x2 block at (r1,c1) is 4-way adjacent to the 2x2 block at (r2,c2).
        We'll just check if any of the 4 cells from the first is next to any from the second.
        """
        block1 = [(r1, c1), (r1, c1+1), (r1+1, c1), (r1+1, c1+1)]
        block2 = [(r2, c2), (r2, c2+1), (r2+1, c2), (r2+1, c2+1)]
        for (rr1, cc1) in block1:
            for (rr2, cc2) in block2:
                if abs(rr1 - rr2) + abs(cc1 - cc2) == 1:
                    return True
        return False

    def _place_2x2_connected_fallback(self, grid: np.ndarray, color2: int, r1: int, c1: int):
        """
        Fallback routine to forcibly place the second 2x2 so that it is
        4-way adjacent to the first 2x2 at (r1,c1). We do random tries 
        anywhere in the grid, then check if adjacency is achieved.
        """
        rows, cols = grid.shape
        placed = False
        attempts = 200
        for _ in range(attempts):
            rr = random.randint(0, rows - 2)
            cc = random.randint(0, cols - 2)
            if self._can_place_2x2(grid, rr, cc):
                if self._is_2x2_connected(r1, c1, rr, cc):
                    grid[rr, cc] = color2
                    grid[rr, cc+1] = color2
                    grid[rr+1, cc] = color2
                    grid[rr+1, cc+1] = color2
                    placed = True
                    break
        
        # If not placed, put it anywhere non-overlapping (worst-case).
        if not placed:
            for rr in range(rows - 1):
                for cc in range(cols - 1):
                    if self._can_place_2x2(grid, rr, cc):
                        grid[rr, cc] = color2
                        grid[rr, cc+1] = color2
                        grid[rr+1, cc] = color2
                        grid[rr+1, cc+1] = color2
                        return


