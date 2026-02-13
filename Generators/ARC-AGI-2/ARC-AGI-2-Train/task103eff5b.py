import random
from typing import Dict, Any, Tuple, List

import numpy as np

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import Contiguity, retry
from transformation_library import find_connected_objects, GridObject


class Task103eff5bGenerator(ARCTaskGenerator):
    """ARC‑AGI Task Generator for the "rotated‑and‑scaled twin objects" pattern.

    A small multi‑coloured object (3×3 or 4×4 bounding box) appears in the upper
    half of the grid.  Its larger twin – identical shape, rotated 90° clockwise
    and uniformly scaled (×3 for a 3×3 object, ×2 for a 4×4 object) – appears in
    the lower half in a single colour.  The solver must repaint the large copy
    so that each corresponding region inherits its colour from the small one.
    """

    # ───────────────────────────────────────── Constructor ──────────────────────────────────────────

    def __init__(self):
        # 1️⃣  Input‑observation template strings
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "Each grid contains exactly two objects: a multi-colored small object and another {color('big_object')} object; the remaining cells are empty (0).",
            "The multi-colored small object is formed by 8-way connected cells using all four colors {color('small_object1')}, {color('small_object2')}, {color('small_object3')}, and {color('small_object4')}. Its shape and size fit within either a 3x3 or 4x4 area.",
            "Each color — {color('small_object1')}, {color('small_object2')}, {color('small_object3')}, and {color('small_object4')} — forms irregular connected regions that create an asymmetric overall shape.",
            "The {color('big_object')} object has a shape that exactly matches the multi-colored object. Its size is 8x8 if the multi-colored object is 4x4; otherwise, it is 9x9.",
            "Ensure both objects are completely separated from each other by empty (0) cells. Once the {color('big_object')} object has been created, rotate it 90 degrees clockwise before placing it.",
            "The {color('big_object')} object should be placed in the bottom half, and the multi-colored small object should be placed in the upper half."
        ]

        # 2️⃣  Transformation‑reasoning template strings
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the two colored objects: one multi-colored small object, formed by 8-way connected cells using all four colors {color('small_object1')}, {color('small_object2')}, {color('small_object3')}, and {color('small_object4')}, and the {color('big_object')} object.",
            "The {color('big_object')} object is always larger than the smaller multi-colored object but has exactly the same shape. It is rotated 90 degrees clockwise before being placed.",
            "The {color('big_object')} object should be placed in the bottom half, and the multi-colored small object should be placed in the upper half.",
            "To construct the output grid, change the colors of the {color('big_object')} object to match the respective colors of the multi-colored small object.",
            "All other cells remain unchanged."
        ]
        
        # 3️⃣  Delegate to base class
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    # ───────────────────────────────────────── Utility helpers ──────────────────────────────────────

    @staticmethod
    def _generate_small_shape(size: int, colours: List[int]) -> np.ndarray:
        """Create a connected multi‑coloured irregular shape inside a *size×size* array.

        Rules
        -----
        • Exactly N colours (length of the `colours` list) – each forming irregular connected regions.
        • Overall 8‑way connectivity.
        • Shape must be irregular and asymmetric to ensure unique coloring correspondence.
        • Shape must *change* when rotated 90° clockwise (to avoid accidental symmetry).
        """

        def _is_valid(arr: np.ndarray) -> bool:
            objs = find_connected_objects(arr, diagonal_connectivity=True, background=0, monochromatic=False)
            connected = len(objs) == 1
            # Different when rotated CW (k=-1)
            rotated = np.rot90(arr, k=-1)
            different = not np.array_equal((arr != 0), (rotated != 0))
            # Each colour appears exactly once as a connected component
            unique_colours = {c for c in arr.flat if c != 0}
            colour_ok = len(unique_colours) == len(colours)
            
            # Check for irregularity - ensure no perfect rectangles or simple patterns
            irregular = _check_irregularity(arr)
            
            return connected and different and colour_ok and irregular

        def _check_irregularity(arr: np.ndarray) -> bool:
            """Check if the shape is sufficiently irregular and asymmetric."""
            # Check that it's not symmetric in any way
            flipped_h = np.fliplr(arr)
            flipped_v = np.flipud(arr)
            rot_180 = np.rot90(arr, k=2)
            
            symmetric = (np.array_equal(arr != 0, flipped_h != 0) or 
                        np.array_equal(arr != 0, flipped_v != 0) or
                        np.array_equal(arr != 0, rot_180 != 0))
            
            if symmetric:
                return False
                
            # Check that each color region is not a perfect rectangle
            for color in colours:
                color_mask = (arr == color)
                if np.sum(color_mask) == 0:
                    continue
                    
                # Find bounding box of this color
                rows, cols = np.where(color_mask)
                if len(rows) > 1:  # Only check if more than one cell
                    min_r, max_r = rows.min(), rows.max()
                    min_c, max_c = cols.min(), cols.max()
                    expected_cells = (max_r - min_r + 1) * (max_c - min_c + 1)
                    actual_cells = np.sum(color_mask)
                    
                    # If it's a perfect rectangle, reject
                    if actual_cells == expected_cells and actual_cells > 1:
                        return False
            
            return True

        def _make_once() -> np.ndarray:
            arr = np.zeros((size, size), dtype=int)
            occupied = set()

            # Create irregular patterns by growing regions organically
            for i, col in enumerate(colours):
                placed = False
                attempts = 0
                
                while not placed and attempts < 200:
                    attempts += 1
                    
                    # For first color, start anywhere; for others, start adjacent to existing shape
                    if not occupied:
                        r, c = random.randrange(size), random.randrange(size)
                        start_cells = [(r, c)]
                    else:
                        # Find all possible starting positions adjacent to existing shape
                        adjacent_cells = []
                        for (r0, c0) in occupied:
                            for dr in (-1, 0, 1):
                                for dc in (-1, 0, 1):
                                    if dr == 0 and dc == 0:
                                        continue
                                    nr, nc = r0 + dr, c0 + dc
                                    if (0 <= nr < size and 0 <= nc < size and 
                                        (nr, nc) not in occupied and (nr, nc) not in adjacent_cells):
                                        adjacent_cells.append((nr, nc))
                        
                        if not adjacent_cells:
                            continue
                            
                        start_cells = [random.choice(adjacent_cells)]
                    
                    # Grow this color region organically
                    region_cells = []
                    candidates = start_cells.copy()
                    target_size = random.randint(1, min(4, size * size // 4))  # Variable region size
                    
                    while candidates and len(region_cells) < target_size:
                        current = candidates.pop(random.randrange(len(candidates)))
                        if current in occupied:
                            continue
                            
                        region_cells.append(current)
                        
                        # Add neighbors as potential expansion candidates
                        r, c = current
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < size and 0 <= nc < size and 
                                    (nr, nc) not in occupied and (nr, nc) not in region_cells and
                                    (nr, nc) not in candidates):
                                    # Add with some probability to create irregular growth
                                    if random.random() < 0.7:  # 70% chance to add neighbor
                                        candidates.append((nr, nc))
                    
                    # Check if this region would create a valid shape
                    if len(region_cells) > 0:
                        # Temporarily place the region
                        temp_occupied = occupied.copy()
                        for r, c in region_cells:
                            arr[r, c] = col
                            temp_occupied.add((r, c))
                        
                        # Check if still connected overall
                        test_objs = find_connected_objects(arr, diagonal_connectivity=True, background=0, monochromatic=False)
                        if len(test_objs) == 1:
                            occupied = temp_occupied
                            placed = True
                        else:
                            # Revert
                            for r, c in region_cells:
                                arr[r, c] = 0
                
                if not placed:
                    # If we couldn't place this color, restart the whole process
                    return np.zeros((size, size), dtype=int)
            
            return arr

        # Retry until the constraints are met
        shape = retry(_make_once, _is_valid, max_attempts=100)
        return shape

    @staticmethod
    def _scale_and_rotate(shape: np.ndarray, scale: int, colour: int) -> np.ndarray:
        """Rotate *shape* 90° CW and scale every coloured cell to an *scale×scale* block."""
        rotated = np.rot90(shape, k=-1)
        side = rotated.shape[0] * scale
        big = np.zeros((side, side), dtype=int)
        for r in range(rotated.shape[0]):
            for c in range(rotated.shape[1]):
                if rotated[r, c] != 0:
                    big[r*scale:(r+1)*scale, c*scale:(c+1)*scale] = colour
        return big

    # ───────────────────────────────────────── Grid generation ──────────────────────────────────────

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        size_small: int = gridvars["small_size"]  # 3 or 4
        # Allow an override to use only three small colours (for test grids)
        if gridvars.get("three_colours", False):
            colours_small = [taskvars[f"small_object{i}"] for i in range(1, 4)]
        else:
            colours_small = [taskvars[f"small_object{i}"] for i in range(1, 5)]
        colour_big = taskvars["big_object"]

        # 1. Build the small object (within its own bounding box)
        small_shape = self._generate_small_shape(size_small, colours_small)

        # 2. Build the corresponding big object
        scale = 3 if size_small == 3 else 2
        big_shape = self._scale_and_rotate(small_shape, scale, colour_big)
        size_big = big_shape.shape[0]

        # 3. Determine grid dimensions ensuring halves can accommodate objects with one empty row gap
        min_height = 0
        h = 0
        while True:
            h = random.randint(size_big + size_small + 3, 30)  # upper bound 30
            top_half = h // 2
            bottom_half = h - top_half
            if top_half >= size_small + 1 and bottom_half >= size_big:
                break
        w = random.randint(max(size_big, size_small) + 2, 30)
        grid = np.zeros((h, w), dtype=int)

        # 4. Place small object in upper half
        top_half_limit = h // 2 - size_small - 1
        small_row = random.randint(0, top_half_limit)
        small_col = random.randint(0, w - size_small)
        grid[small_row:small_row + size_small, small_col:small_col + size_small] = np.where(
            small_shape != 0, small_shape, grid[small_row:small_row + size_small, small_col:small_col + size_small]
        )

        # 5. Place big object in lower half starting exactly at the mid‑row for guaranteed separation
        big_row = h // 2
        big_col = random.randint(0, w - size_big)
        grid[big_row:big_row + size_big, big_col:big_col + size_big] = np.where(
            big_shape != 0, big_shape, grid[big_row:big_row + size_big, big_col:big_col + size_big]
        )

        return grid

    # ────────────────────────────────────────── Transformation ──────────────────────────────────────

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        big_colour = taskvars["big_object"]
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=False)

        # Identify objects: the big object is the one with the single big_colour;
        # the other object is the small multi‑coloured object (may use 3 or 4 colours).
        big_obj: GridObject = next(obj for obj in objects if obj.colors == {big_colour})
        small_obj: GridObject = next(obj for obj in objects if obj is not big_obj)
        # Reference arrays and helper data
        small_arr = small_obj.to_array()
        rotated_small = np.rot90(small_arr, k=-1)
        # Scaling factor from small to big (integer; identical in both dims)
        big_bb = big_obj.bounding_box
        big_h = big_bb[0].stop - big_bb[0].start
        scale = big_h // rotated_small.shape[0]
        out = grid.copy()
        r0, c0 = big_bb[0].start, big_bb[1].start
        for r, c, _ in big_obj:
            rr = (r - r0) // scale  # row in rotated_small
            cc = (c - c0) // scale  # col in rotated_small
            new_colour = rotated_small[rr, cc]
            out[r, c] = new_colour
        return out

    # ──────────────────────────────────────────── create_grids ──────────────────────────────────────

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # 1. Task‑level colour choices (distinct, non‑zero)
        colour_choices = random.sample(range(1, 10), 5)
        taskvars = {
            "big_object": colour_choices[0],
            "small_object1": colour_choices[1],
            "small_object2": colour_choices[2],
            "small_object3": colour_choices[3],
            "small_object4": colour_choices[4]
        }

        # 2. Decide how many training examples (3 or 4) and guarantee both 3×3 and 4×4 cases
        nr_train = random.choice([3, 4])
        train_sizes = [3, 4]
        while len(train_sizes) < nr_train:
            train_sizes.append(random.choice([3, 4]))
        random.shuffle(train_sizes)

        # 3. Build train grids
        train: List[GridPair] = []
        for size in train_sizes:
            inp = self.create_input(taskvars, {"small_size": size})
            train.append({
                "input": inp,
                "output": self.transform_input(inp, taskvars)
            })

        # 4. One test grid (size randomly 3 or 4)
        test_size = random.choice([3, 4])
        # For the single test grid, use only three small colours as requested
        test_inp = self.create_input(taskvars, {"small_size": test_size, "three_colours": True})
        test = [{
            "input": test_inp,
            "output": self.transform_input(test_inp, taskvars)
        }]

        return taskvars, {"train": train, "test": test}
