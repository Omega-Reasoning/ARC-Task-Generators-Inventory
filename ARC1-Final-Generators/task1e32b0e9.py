import numpy as np
import random
from typing import Dict, Any, Tuple

# Import from the provided base class and libraries:
from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import Contiguity

class Task1e32b0e9Generator(ARCTaskGenerator):
    def __init__(self):
        # 1. Initialize the input reasoning chain
        self.input_reasoning_chain = [
                "The input grid has size {vars['rows']} X {vars['rows']}.",
                "The grid is a raster with color(between 1-9) rows and columns as separators.",
                "The first row and column are not a separator row/column.",
                "The raster divides the input matrix into subgrids of size {(vars['rows']-2)//3} x {(vars['rows']-2)//3} all of which have the same dimension.",
                "In the top left subgrid a 4-way connected object is placed at the center of the subgrid which is either a square or a plus shaped object, this is the reference object which is of color(between 1-9).",
                "In the remaining subgrid only part of the above chosen objects are placed.",
                "All the other remaining cells are empty(0)",
            ]
        # 2. Initialize the transformation reasoning chain
        self.transformation_reasoning_chain = [
            "The output grid has the same size as the input grid.",
            "Copy the input grid to the output grid.",
            "First identify the 4-way connected object in the top left subgrid.",
            "For the remaining subgrids the above identified object is placed at the center such that the already existing colored cells are not masked by the object.",
            "The existing cells with color in the subgrid remained unchanged while the remaining object cells are colored the raster row/column color.",
        ]
        # 3. Call the base class constructor
        super().__init__(self.input_reasoning_chain, self.transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Creates an input grid following these instructions:
        1) The grid size is rows x rows, with rows in {17,23,29}.
        2) The grid is divided into 3x3 subgrids separated by lines colored sep_color.
        3) The top-left subgrid has a 4-way connected object in the center (square or plus) of color obj_color.
        4) The other subgrids have a random subset of that shape placed in them.
        5) All other cells remain 0.
        """
        rows = taskvars["rows"]
        sep_color = gridvars["sep_color"]
        obj_color = gridvars["obj_color"]

        # Initialize the grid with 0.
        grid = np.zeros((rows, rows), dtype=int)

        # We define subgrid_size = (rows - 2)//3 as per instructions.
        subgrid_size = (rows - 2) // 3

        # We'll define lines to separate the subgrids.
        def clamp_line(k):
            return min(k, rows - 1)

        # Potential horizontal/vertical separator lines
        lines = [
                 clamp_line(subgrid_size),
                 clamp_line(2 * (subgrid_size ) + 1)]

        # Color the separator rows
        for r in lines:
            grid[r, :] = sep_color
        # Color the separator columns
        for c in lines:
            grid[:, c] = sep_color

        # Identify the bounding box for the top-left subgrid.
        #  That subgrid is between lines[0] and lines[1], but excluding the lines themselves.
        subgrid_top = 0
        subgrid_bottom = lines[0] - 1
        subgrid_left = 0
        subgrid_right = lines[0] - 1
        subgrid_h = subgrid_bottom - subgrid_top + 1
        subgrid_w = subgrid_right - subgrid_left + 1

        # The shape can be one of several types. We'll place it in the center.
        shape_types = ["square", "plus", "t", "l", "diamond", "line"]
        # Allow the caller to request a specific shape via gridvars (so different
        # examples can use different top-left shapes). Fall back to random.
        shape_type = gridvars.get("shape_type", random.choice(shape_types))
        # Allow caller to override shape size; otherwise choose randomly
        max_shape_dim = max(2, subgrid_h - 2)  # ensure at least small shapes
        shape_dim = gridvars.get('shape_dim') if gridvars and 'shape_dim' in gridvars else random.randint(2, max_shape_dim)
        # Clamp shape_dim so it never exceeds available subgrid dimensions
        shape_dim = min(shape_dim, subgrid_h, subgrid_w)

        subgrid_center_r = subgrid_top + subgrid_h // 2
        subgrid_center_c = subgrid_left + subgrid_w // 2

        def place_square(dim):
            # Place a dim x dim filled square centered on the subgrid center
            r0 = subgrid_center_r - (dim // 2)
            c0 = subgrid_center_c - (dim // 2)
            for rr in range(dim):
                for cc in range(dim):
                    grid[r0 + rr, c0 + cc] = obj_color

        def place_plus(dim):
            # Place a plus shape where 'dim' controls total arm length
            half = dim // 2
            for rr in range(subgrid_center_r - half, subgrid_center_r + half + 1):
                grid[rr, subgrid_center_c] = obj_color
            for cc in range(subgrid_center_c - half, subgrid_center_c + half + 1):
                grid[subgrid_center_r, cc] = obj_color

        def place_t(dim):
            # T-shape: vertical stem and a horizontal cap at the top
            half = dim // 2
            # vertical stem
            for rr in range(subgrid_center_r - half, subgrid_center_r + half + 1):
                grid[rr, subgrid_center_c] = obj_color
            # horizontal cap at the top of the stem
            cap_row = subgrid_center_r - half
            for cc in range(subgrid_center_c - half, subgrid_center_c + half + 1):
                grid[cap_row, cc] = obj_color

        def place_l(dim):
            # L-shape: vertical bar on the left of center and horizontal foot at bottom
            half = dim // 2
            # vertical bar
            for rr in range(subgrid_center_r - half, subgrid_center_r + half + 1):
                grid[rr, subgrid_center_c - half] = obj_color
            # horizontal foot at the bottom
            foot_row = subgrid_center_r + half
            for cc in range(subgrid_center_c - half, subgrid_center_c + half + 1):
                grid[foot_row, cc] = obj_color

        def place_diamond(dim):
            # Diamond in Manhattan metric: all cells where |dr|+|dc| <= radius
            radius = dim // 2
            for dr in range(-radius, radius + 1):
                span = radius - abs(dr)
                for dc in range(-span, span + 1):
                    rr = subgrid_center_r + dr
                    cc = subgrid_center_c + dc
                    grid[rr, cc] = obj_color

        def place_line(dim):
            # Straight horizontal line centered on the subgrid center
            half = dim // 2
            for cc in range(subgrid_center_c - half, subgrid_center_c + half + 1):
                grid[subgrid_center_r, cc] = obj_color

        # Map shape types to functions
        placer = {
            "square": place_square,
            "plus": place_plus,
            "t": place_t,
            "l": place_l,
            "diamond": place_diamond,
            "line": place_line,
        }

        # Place the chosen shape
        placer[shape_type](shape_dim)

        # We'll now identify the shape's coordinates, then place partial subsets in the remaining subgrids.
        # We'll do a 4-way find of connected objects, find the one containing subgrid_center.
        objects_4 = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
        shape_obj = None
        for obj in objects_4:
            if (subgrid_center_r, subgrid_center_c) in obj.coords:
                shape_obj = obj
                break
        if shape_obj is None:
            # Fallback: try to force a minimal marker at the center so the
            # detection logic always finds an object. This avoids returning an
            # empty pattern in some rare placement failures.
            grid[subgrid_center_r, subgrid_center_c] = obj_color
            objects_4 = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
            for obj in objects_4:
                if (subgrid_center_r, subgrid_center_c) in obj.coords:
                    shape_obj = obj
                    break
        if shape_obj is None:
            # As a last resort give up and return the grid with at least the
            # center marked; downstream code will still run but other
            # subgrids are ensured to receive at least one colored cell below.
            return grid

        shape_coords = list(shape_obj.coords)

        # For each of the other subgrids, we'll place a random subset of the shape's coords.
        def get_subgrid_rc(rindex, cindex):
            # Adjusted to account for first row/column being separators
            if rindex >= 3 or cindex >= 3:
                return None
            # Calculate bounds considering the first row/column are separators
            rtop = rindex * (subgrid_size + 1)
            rbot = rtop + subgrid_size - 1
            cleft = cindex * (subgrid_size + 1)
            cright = cleft + subgrid_size - 1
            if rbot >= rows or cright >= rows:
                return None
            return (rtop, rbot, cleft, cright)

        for rindex in range(3):
            for cindex in range(3):
                if (rindex, cindex) == (0, 0):
                    continue  # skip top-left subgrid
                bounds = get_subgrid_rc(rindex, cindex)
                if bounds is None:
                    continue
                (rtop, rbot, cleft, cright) = bounds
                sub_h2 = rbot - rtop + 1
                sub_w2 = cright - cleft + 1
                if sub_h2 <= 0 or sub_w2 <= 0:
                    continue
                # center of this subgrid
                sub_ctr_r = rtop + sub_h2 // 2
                sub_ctr_c = cleft + sub_w2 // 2
                offset_r = sub_ctr_r - subgrid_center_r
                offset_c = sub_ctr_c - subgrid_center_c

                # pick random subset of shape_coords
                max_subset = max(1, len(shape_coords) - 1)
                subset_size = random.randint(1, max_subset)
                subset_coords = random.sample(shape_coords, subset_size)

                for (rr, cc) in subset_coords:
                    new_r = rr + offset_r
                    new_c = cc + offset_c
                    if 0 <= new_r < rows and 0 <= new_c < rows:
                        grid[new_r, new_c] = obj_color
        # Ensure every subgrid has at least one colored cell. Some subsets or
        # placement near separators could leave a subgrid empty; guarantee at
        # least one colored cell (prefer center) so downstream logic sees a
        # colored cell in each subgrid.
        for rindex in range(3):
            for cindex in range(3):
                bounds = get_subgrid_rc(rindex, cindex)
                if bounds is None:
                    continue
                (rtop, rbot, cleft, cright) = bounds
                # slice and check for any non-zero cell
                if np.any(grid[rtop:rbot+1, cleft:cright+1] != 0):
                    continue
                # place one colored cell at the subgrid center
                sub_h2 = rbot - rtop + 1
                sub_w2 = cright - cleft + 1
                if sub_h2 <= 0 or sub_w2 <= 0:
                    continue
                sub_ctr_r = rtop + sub_h2 // 2
                sub_ctr_c = cleft + sub_w2 // 2
                grid[sub_ctr_r, sub_ctr_c] = obj_color
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transforms the input grid according to the transformation reasoning chain:
        1) Output grid same size as input.
        2) Copy input -> output.
        3) Identify the 4-way object in top-left subgrid.
        4) In all other subgrids, place that object at center, but do not overwrite existing colored cells.
        5) Instead, wherever the shape would go in an empty cell, color it with sep_color.
        """
        rows = grid.shape[0]
        #sep_color = taskvars["sep_color"]

        # 1) and 2):
        output = np.copy(grid)

        # Calculate first separator line position
        subgrid_size = (rows - 2) // 3
        # Get sep_color from the first separator line
        sep_color = grid[subgrid_size, 0]  # First horizontal separator line

        def clamp_line(k):
            return min(k, rows - 1)
        lines = [clamp_line(subgrid_size), clamp_line(2 * (subgrid_size + 1))]
        subgrid_coords =[0, clamp_line(subgrid_size), clamp_line(2 * (subgrid_size ) + 1),rows]

        # top-left subgrid bounding box
        subgrid_top = 0
        subgrid_bottom = lines[0] - 1
        subgrid_left = 0
        subgrid_right = lines[0] - 1
        subgrid_h = subgrid_bottom - subgrid_top + 1
        subgrid_w = subgrid_right - subgrid_left + 1
        subgrid_center_r = subgrid_top + subgrid_h // 2
        subgrid_center_c = subgrid_left + subgrid_w // 2

        # find the shape object.
        objects_4 = find_connected_objects(output, diagonal_connectivity=False, background=0, monochromatic=False)
        shape_obj = None
        for obj in objects_4:
            if (subgrid_center_r, subgrid_center_c) in obj.coords:
                shape_obj = obj
                break
        if shape_obj is None:
            return output

        # 4) For the remaining subgrids, place shape.
        def get_subgrid_rc(rindex, cindex):
            if rindex >= 3 or cindex >= 3:
                return None
            if rindex == 0 and cindex > 0 and cindex < 3:
                rtop = subgrid_coords[rindex] 
            else:
                rtop = subgrid_coords[rindex] + 1

            if cindex == 0 and (rindex == 1 or rindex == 2):
                cleft = subgrid_coords[cindex] 
            else:
                cleft = subgrid_coords[cindex] + 1

            rbot = subgrid_coords[rindex + 1] - 1
            cright = subgrid_coords[cindex + 1] - 1
            if rbot < rtop or cright < cleft:
                return None
            return (rtop, rbot, cleft, cright)

        for rindex in range(3):
            for cindex in range(3):
                if (rindex, cindex) == (0, 0):
                    continue
                coords = get_subgrid_rc(rindex, cindex)
                if coords is None:
                    continue
                (rtop, rbot, cleft, cright) = coords
                sub_h2 = rbot - rtop + 1
                sub_w2 = cright - cleft + 1
                if sub_h2 <= 0 or sub_w2 <= 0:
                    continue
                sub_ctr_r = rtop + sub_h2 // 2
                sub_ctr_c = cleft + sub_w2 // 2
                offset_r = sub_ctr_r - subgrid_center_r
                offset_c = sub_ctr_c - subgrid_center_c

                # place shape: existing color remains, empty cells get sep_color.
                for (rr, cc, shape_col) in shape_obj.cells:
                    new_r = rr + offset_r
                    new_c = cc + offset_c
                    if 0 <= new_r < rows and 0 <= new_c < rows:
                        if output[new_r, new_c] == 0:  # was empty
                            output[new_r, new_c] = sep_color

        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Create the final grids:
        1) Randomly pick rows from [17, 23, 29].
        2) Randomly pick distinct sep_color, obj_color from [1..9].
        3) Create 3 or 4 train pairs, plus 1 test pair.
        4) Return (taskvars, TrainTestData).
        """
        rows = random.choice([17,23,29])
        #sep_color, obj_color = random.sample(range(1, 10), 2)

        taskvars = {
            "rows": rows
        }

        nr_train = random.choice([3, 4])
        train_data = []
        # choose shape types for each example so the top-left object varies.
        shape_types = ["square", "plus", "t", "l", "diamond", "line"]
        total_needed = nr_train + 1  # train + test
        if total_needed <= len(shape_types):
            chosen_shapes = random.sample(shape_types, total_needed)
        else:
            # not enough unique shapes; allow repeats
            chosen_shapes = [random.choice(shape_types) for _ in range(total_needed)]

        for i in range(nr_train):
            sep_color, obj_color = random.sample(range(1, 10), 2)
            gridvars = {"sep_color": sep_color, "obj_color": obj_color, "shape_type": chosen_shapes[i]}
            inp = self.create_input(taskvars, gridvars)
            outp = self.transform_input(inp, taskvars)
            train_data.append({"input": inp, "output": outp})

        # Test pair
        sep_color, obj_color = random.sample(range(1, 10), 2)
        test_shape = chosen_shapes[-1]
        gridvars = {"sep_color": sep_color, "obj_color": obj_color, "shape_type": test_shape}
        test_inp = self.create_input(taskvars, gridvars)
        test_outp = self.transform_input(test_inp, taskvars)
        test_data = [{"input": test_inp, "output": test_outp}]

        return taskvars, {"train": train_data, "test": test_data}

