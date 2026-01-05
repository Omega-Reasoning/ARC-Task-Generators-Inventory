from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Utility libraries
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObject, GridObjects


class Task321b1fc6Generator(ARCTaskGenerator):
    def __init__(self):
        # Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain {vars['num_objects']} objects, each completely separated from the others by empty (0) cells.",
            "Exactly one object is multi-colored (cells 1-9) and the remaining objects are monochromatic with color {color('object_color')}",
            "All objects have the exact same shape."
        ]

        # Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids have the same size as the input grids.",
            "Identify the monochromatic objects (color {color('object_color')}) and the single multi-colored object.",
            "Change the color of each monochromatic object to exactly match the colors of the multi-colored object, while keeping their positions unchanged.",
            "Remove the original multi-colored object."
        ]

        # Call superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        """
        Generates an input grid ensuring:
        - A variable number of objects (3-6 depending on grid size), with one multi-colored and the rest monochromatic
        - No 4-way or 8-way connectivity between objects
        - Each object contains at least 2 cells
        - Objects are randomly placed with spacing constraints
        """
        object_color = taskvars['object_color']

        # 1) Choose grid size dynamically, adjusting if needed
        min_size = 7
        max_size = 30
        base_size = random.randint(min_size, max_size)

        grid = np.zeros((base_size, base_size), dtype=int)

        # 2) Generate a common object shape with at least 2 cells
        shape_h = random.randint(2, max(3, base_size // 4))
        shape_w = random.randint(2, max(3, base_size // 4))

        def valid_shape():
            shape_matrix = create_object(
                height=shape_h,
                width=shape_w,
                color_palette=1,
                contiguity=Contiguity.FOUR,
                background=0
            )
            return shape_matrix if np.count_nonzero(shape_matrix) >= 2 else None

        shape_matrix = retry(valid_shape, lambda s: s is not None, max_attempts=100)
        shape_pattern = (shape_matrix != 0)

        # 3) Generate a multi-colored object ensuring at least two distinct colors
        def generate_multicolored_shape():
            """Generate a multi-colored shape ensuring at least 2 distinct colors."""
            candidate = np.zeros((shape_h, shape_w), dtype=int)

            # Get color choices excluding the object's color
            color_choices = [c for c in range(1, 10) if c != object_color]

            # Ensure at least two distinct colors
            primary_color, secondary_color = random.sample(color_choices, 2) if len(color_choices) > 1 else (color_choices[0], color_choices[0])

            # Fill shape pattern with at least two colors
            first_half = True
            for r in range(shape_h):
                for c in range(shape_w):
                    if shape_pattern[r, c]:
                        candidate[r, c] = primary_color if first_half else secondary_color
                        first_half = not first_half  # Alternate colors

            return candidate

        multi_colored_shape = generate_multicolored_shape()

        # 4) Create monochromatic copies (we'll create N-1 of these, where N is chosen below)
        mono_shape = np.where(shape_pattern, object_color, 0)

        # Decide how many objects to place: at least 3, up to 6, limited by grid size
        max_possible = min(6, max(3, base_size // 3))
        # Use task variable 'num_objects' when provided; otherwise choose randomly
        desired_num = taskvars.get('num_objects', None)
        if desired_num is None:
            num_objects = random.randint(3, max_possible)
        else:
            # Clamp the provided task variable to allowed range for this grid
            num_objects = max(3, min(int(desired_num), max_possible))

        # 5) Ensure separated object placement
        max_attempts = 100
        placed_positions = []

        def can_place(subarr, top, left):
            """Ensure no objects are adjacent (at least one-row/column spacing)."""
            h, w = subarr.shape
            if top + h > base_size or left + w > base_size:
                return False
            # Check if the space is empty and has at least one-row/column margin
            for r in range(-1, h + 1):
                for c in range(-1, w + 1):
                    rr, cc = top + r, left + c
                    if 0 <= rr < base_size and 0 <= cc < base_size and grid[rr, cc] != 0:
                        return False
            return True

        def place_object(subarr):
            """Try to place the object with required spacing."""
            for _ in range(max_attempts):
                rr = random.randint(1, base_size - shape_h - 1)
                cc = random.randint(1, base_size - shape_w - 1)
                if can_place(subarr, rr, cc):
                    grid[rr: rr + shape_h, cc: cc + shape_w] = subarr
                    placed_positions.append((rr, cc, shape_h, shape_w))
                    return True
            return False  # If placement fails

        # Build object list: one multi-colored, rest monochromatic, then shuffle
        objects = [mono_shape] * (num_objects - 1) + [multi_colored_shape]
        random.shuffle(objects)

        for obj in objects:
            if not place_object(obj):
                base_size += 2  # Expand grid if placement fails
                grid = np.zeros((base_size, base_size), dtype=int)
                placed_positions.clear()
                return self.create_input(taskvars, gridvars)  # Retry with a larger grid

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transforms the input grid according to the given reasoning:
    - Matches the monochromatic objects' colors to the multi-colored object.
    - Removes the multi-colored object.
        """
        out_grid = np.copy(grid)
        objects = find_connected_objects(out_grid, diagonal_connectivity=False, background=0, monochromatic=False)

        multi_colored_obj = None
        mono_objs = []

        for obj in objects:
            if len(obj.colors) > 1:
                multi_colored_obj = obj
            else:
                mono_objs.append(obj)

        # Require at least one multi-colored object and at least one monochromatic object
        if multi_colored_obj is None or len(mono_objs) < 1:
            return out_grid

        multi_box = multi_colored_obj.bounding_box
        multi_arr = multi_colored_obj.to_array()

        for mono_obj in mono_objs:
            mono_obj.cut(out_grid, background=0)
            r_slice, c_slice = mono_obj.bounding_box
            for rr in range(multi_arr.shape[0]):
                for cc in range(multi_arr.shape[1]):
                    color_val = multi_arr[rr, cc]
                    if color_val != 0:
                        out_grid[r_slice.start + rr, c_slice.start + cc] = color_val

        multi_colored_obj.cut(out_grid, background=0)

        return out_grid  # Ensure we return the modified grid



    def create_grids(self) -> tuple[dict, TrainTestData]:
        """
        Generates training and test data.
        """
        # Introduce task variable for number of objects per grid
        taskvars = {
            'object_color': random.randint(1, 9),
            'num_objects': random.randint(3, 6)
        }
        nr_train = random.choice([3, 4])
        nr_test = 1

        data = self.create_grids_default(nr_train_examples=nr_train, nr_test_examples=nr_test, taskvars=taskvars)
        return taskvars, data


