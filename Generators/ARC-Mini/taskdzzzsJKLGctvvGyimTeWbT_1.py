import numpy as np
import random

# --- Required imports from the prompt ---
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects, BorderBehavior, CollisionBehavior
from Framework.input_library import Contiguity
# We only need create_object, but import everything for convenience
from Framework.input_library import create_object, retry
# -----------------------------------------

class TaskdzzzsJKLGctvvGyimTeWbT_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (exactly as provided in the prompt)
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain one {color('object_color1')} and one {color('object_color2')} 2x2 square object, while all other cells remain empty (0).",
            "The 2x2 square objects may sometimes be 4-way connected to each other, but they never overlap each other."
        ]
        
        # 2) Transformation reasoning chain (exactly as provided in the prompt)
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling all empty (0) cells with {color('fill_color')} only if the two 2x2 square objects are 4-way connected to each other.",
            "If the 2x2 square objects are not 4-way connected, change the color of the second column of {color('object_color1')} to {color('object_color3')}, and the first column of {color('object_color2')} to {color('object_color4')}."
        ]
        
        # 3) Call the parent class init
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Create an input grid that contains exactly two 2x2 square objects:
        one of color object_color1, one of color object_color2, possibly 4-way connected
        or not, depending on gridvars['connected'] (True or False).
        """
        # Unpack relevant variables
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        connected = gridvars.get('connected', False)  # default False if not specified

        # Choose a random grid size from 6..30 (as per the specification)
        # (You may adjust the upper bound if you want smaller, but let's allow up to 12 or 15 
        #  to reduce visual clutter; the prompt says up to 30 is allowed. We'll do up to 12 for variety.)
        rows = random.randint(6, 30)
        cols = random.randint(6, 30)
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # We want to place exactly two 2x2 squares with distinct colors.
        # Strategy:
        #  1. Place the first 2x2 somewhere random.
        #  2. Depending on 'connected', place the second 2x2 so that:
        #       - If connected, it shares a side with the first 2x2 (but does not overlap).
        #       - If not connected, keep them at least 1 cell away so they're not 4-way adjacent.

        # 1) Place first 2x2
        #    We'll pick top-left corner for the 2x2 to be anywhere that fits.
        #    The available row range is [0..(rows-2)], col range is [0..(cols-2)].
        r1 = random.randint(0, rows - 2)
        c1 = random.randint(0, cols - 2)

        # Fill that 2x2 region with object_color1
        grid[r1:r1+2, c1:c1+2] = object_color1

        # 2) Place second 2x2
        #    We must ensure no overlap. We either want to ensure "connected" or "not connected" by 4-way adjacency.

        def can_place_2x2(r, c):
            """Check that the 2x2 at (r, c) does not overlap with the existing 2x2
               and is (or isn't) 4-way adjacent according to 'connected'."""
            # Proposed cells for the new 2x2:
            proposed_cells = [(r+dr, c+dc) for dr in range(2) for dc in range(2)]

            # Check for overlap
            for pr, pc in proposed_cells:
                if grid[pr, pc] != 0:
                    return False
            
            # Now check adjacency if 'connected' is True:
            # We want at least one cell from the new 2x2 to be side-adjacent to a cell from the old 2x2 
            # if connected==True, or none if connected==False.
            old_2x2_cells = [(r1+dr, c1+dc) for dr in range(2) for dc in range(2)]

            # 4-way adjacency set
            def neighbors_4way(rp, cp):
                return [(rp-1, cp), (rp+1, cp), (rp, cp-1), (rp, cp+1)]
            
            # Check if there's ANY side-adjacency with the old object
            adjacency_found = False
            for (nr, nc) in old_2x2_cells:
                for (rp, cp) in neighbors_4way(nr, nc):
                    if (rp, cp) in proposed_cells:
                        adjacency_found = True
                        break
                if adjacency_found:
                    break
            
            # We want adjacency_found == connected
            return adjacency_found == connected

        # We'll attempt to place the second 2x2 randomly up to some max tries.
        max_tries = 200
        placed_second = False
        for _ in range(max_tries):
            r2 = random.randint(0, rows - 2)
            c2 = random.randint(0, cols - 2)
            if can_place_2x2(r2, c2):
                grid[r2:r2+2, c2:c2+2] = object_color2
                placed_second = True
                break
        
        # If we fail to place the second object with the desired adjacency, 
        # we fallback to a simpler approach: just place it ignoring adjacency, and hope the user will see it in another example. 
        # But we do want valid tasks, so let's do a fallback that places it separated if needed:
        if not placed_second:
            # If we can't place a connected one, let's place it disconnected anyway (or vice versa).
            # For reliability, just place it anywhere not overlapping:
            for _ in range(max_tries):
                r2 = random.randint(0, rows - 2)
                c2 = random.randint(0, cols - 2)
                overlap = np.any(grid[r2:r2+2, c2:c2+2] != 0)
                if not overlap:
                    grid[r2:r2+2, c2:c2+2] = object_color2
                    break

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Implement the transformation reasoning chain:
        
        1) Copy the input grid.
        2) Find if the two 2x2 squares (in colors object_color1, object_color2) are 4-way connected.
        3) If connected, fill all empty cells (0) with fill_color.
        4) If not connected, recolor the second column of the object_color1 square to object_color3,
           and the first column of the object_color2 square to object_color4.
        """
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        object_color3 = taskvars['object_color3']
        object_color4 = taskvars['object_color4']
        fill_color   = taskvars['fill_color']

        out_grid = grid.copy()

        # 1) Identify exactly the two squares of interest.
        #    We'll find connected objects of color1 or color2. 
        #    Because they are strictly 2x2 squares, each will appear as a single "connected component" 
        #    for each color if there's no partial overlap. Let's find them:
        objects = find_connected_objects(out_grid, diagonal_connectivity=False, background=0, monochromatic=True)

        # Filter by color
        obj1_candidates = objects.with_color(object_color1)
        obj2_candidates = objects.with_color(object_color2)

        if len(obj1_candidates) == 0 or len(obj2_candidates) == 0:
            # Edge case: if something went wrong and we didn't find them, just return out_grid
            return out_grid

        # We'll assume there's only one such object of each color:
        obj1 = obj1_candidates[0]
        obj2 = obj2_candidates[0]

        # 2) Check 4-way adjacency
        connected_4way = obj1.touches(obj2, diag=False)

        if connected_4way:
            # 3) If connected, fill all empty cells with fill_color
            out_grid[out_grid == 0] = fill_color
        else:
            # 4) If not connected, recolor the second column (within the bounding box) of object_color1 to object_color3
            #    and the first column of object_color2 to object_color4.

            # Recolor object_color1's second column:
            bb1 = obj1.bounding_box  # (row_slice, col_slice)
            # The bounding box for a 2x2 is presumably 2 wide, so the second column is col_slice.start + 1
            # But let's guard for any edge case:
            col1_left  = bb1[1].start
            col1_right = bb1[1].stop - 1  # typically col1_left + 1 for a 2-wide
            if col1_left < col1_right:  # ensure we have at least 2 columns in the bounding box
                for r, c in obj1.coords:
                    if c == col1_right:  # "second column"
                        out_grid[r, c] = object_color3

            # Recolor object_color2's first column:
            bb2 = obj2.bounding_box
            col2_left  = bb2[1].start
            # For the first column, we recolor everything in col2_left
            for r, c in obj2.coords:
                if c == col2_left:
                    out_grid[r, c] = object_color4

        return out_grid

    def create_grids(self):
        """
        Create 3-5 train examples (we'll do 4 for variety) plus 2 test examples.
        We ensure at least one connected and one disconnected example in training,
        and similarly for testing.
        
        Returns: 
            (taskvars, train_test_data)
        """
        # Randomly choose 5 distinct colors from 1..9 for the task:
        colors = random.sample(range(1, 10), 5)
        taskvars = {
            'object_color1': colors[0],
            'object_color2': colors[1],
            'object_color3': colors[2],
            'object_color4': colors[3],
            'fill_color':    colors[4],
        }

        # We want 4 training examples, 2 test examples.
        # Enforce that at least one training example is connected, one is disconnected.
        # We'll fix the pattern:
        #   training[0] -> connected
        #   training[1] -> disconnected
        #   training[2], training[3] -> random choice
        #   test[0] -> connected
        #   test[1] -> disconnected
        
        def make_example(is_connected: bool) -> GridPair:
            in_grid = self.create_input(taskvars, {'connected': is_connected})
            out_grid = self.transform_input(in_grid, taskvars)
            return {'input': in_grid, 'output': out_grid}

        train = []
        # T1: connected
        train.append(make_example(True))
        # T2: disconnected
        train.append(make_example(False))

        # T3, T4: random
        for _ in range(2):
            train.append(make_example(random.choice([True, False])))

        # Test: we want exactly 2
        # T1: connected, T2: disconnected
        test = [
            make_example(True),
            make_example(False)
        ]
        
        train_test_data: TrainTestData = {
            'train': train,
            'test': test
        }

        return taskvars, train_test_data



