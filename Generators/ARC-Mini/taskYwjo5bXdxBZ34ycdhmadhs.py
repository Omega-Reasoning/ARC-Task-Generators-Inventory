# my_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import create_object, Contiguity, retry
from typing import Dict, Any, Tuple, List
import numpy as np
import random


class TaskYwjo5bXdxBZ34ycdhmadhsGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain only two objects, where an object is a 4-way connected group of cells of the same color.",
            "The colors of the two objects can only be either {color('object_color1')} and {color('object_color2')} or {color('object_color3')} and {color('object_color4')}.",
            "The position and shape of the objects may vary between examples.",
            "All other cells are empty (0)."
        ]

        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and filling the empty (0) cells based on the colors of the two objects in the input grid.",
            "If the objects are of colors {color('object_color1')} and {color('object_color2')}, fill the empty (0) cells with {color('fill_color1')}.",
            "If the objects are of colors {color('object_color3')} and {color('object_color4')}, fill the empty (0) cells with {color('fill_color2')}."
        ]

        # 3) Call super constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars,
                     gridvars):
        """
        Create an input grid according to the input reasoning chain given
        the task and grid variables.
        """
        # We are given a choice: place two objects with either (object_color1, object_color2)
        # or (object_color3, object_color4). gridvars['color_choice'] = 1 or 2 indicates which pair.
        # We will generate a random size grid. Then place the two objects in it, 
        # ensuring constraints:
        #   - Each object has size >= 2 and <= 25% of total cells
        #   - The two objects must not be connected (4-way adjacency)

        color_pair = (taskvars['object_color1'], taskvars['object_color2']) \
            if gridvars['color_choice'] == 1 else (taskvars['object_color3'], taskvars['object_color4'])

        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        total_cells = rows * cols

        # For convenience:
        # The object size must be at least 2 and at most total_cells//4
        min_size = 2
        max_size = max(total_cells // 4, 2)  # ensure it's at least 2

        # We will place a background=0 grid
        grid = np.zeros((rows, cols), dtype=int)

        # We define a function to generate a single 4-connected object of random shape
        # with a chosen color from color_pair, ensuring object size in [min_size, max_size].
        # We'll use create_object from Framework.input_library, then place it randomly if possible.
        def generate_single_object(color) -> np.ndarray:
            # We'll generate an object in a bounding box that is somewhat smaller
            # than the entire grid. Then we place that bounding box somewhere inside the grid.
            # We attempt random bounding box sizes up to the entire grid dimension,
            # but we rely on the internal random fill for shape.
            # We'll keep retrying until the object meets size constraints.
            # We do contiguity=FOUR to ensure 4-way connectivity.

            # We'll do a simple approach: pick a bounding box up to 1/2 of rows, 1/2 of cols 
            # to keep it from dominating. This is not mandatory, just for variety.
            # Then let create_object fill it, and check the actual object size.
            # If it's within [min_size, max_size], we accept it. Otherwise, retry.
            def attempt_object():
                box_height = random.randint(1, rows // 2 if rows >= 2 else 1)
                box_width = random.randint(1, cols // 2 if cols >= 2 else 1)

                obj_mat = create_object(
                    height=box_height,
                    width=box_width,
                    color_palette=color,
                    contiguity=Contiguity.FOUR,
                    background=0
                )
                return obj_mat

            valid_obj_mat = retry(
                attempt_object,
                predicate=lambda mat: min_size <= (mat != 0).sum() <= max_size,
                max_attempts=200
            )
            return valid_obj_mat

        # We'll place objects so that they do NOT overlap or connect (4-way).
        # We'll attempt a few times, each time randomly placing object2, if it overlaps or connects to object1, we retry.

        def place_two_objects():
            # Generate object 1
            color1 = color_pair[0]
            obj1_mat = generate_single_object(color1)

            # Random top-left for obj1
            top1 = random.randint(0, rows - obj1_mat.shape[0])
            left1 = random.randint(0, cols - obj1_mat.shape[1])

            # Place it into a copy of the grid to see result
            candidate_grid = np.copy(grid)
            for r in range(obj1_mat.shape[0]):
                for c in range(obj1_mat.shape[1]):
                    if obj1_mat[r, c] != 0:
                        candidate_grid[top1 + r, left1 + c] = obj1_mat[r, c]

            # Generate object 2
            color2 = color_pair[1]
            obj2_mat = generate_single_object(color2)

            # Attempt random positions for object2, check no overlap & no connection
            # We'll do up to X attempts
            for _ in range(200):
                top2 = random.randint(0, rows - obj2_mat.shape[0])
                left2 = random.randint(0, cols - obj2_mat.shape[1])

                # Check if we can place it
                # 1) no overlap
                # 2) no 4-way adjacency to object1
                overlap = False
                adjacency = False

                # Temporarily place in a test grid
                test_grid = np.copy(candidate_grid)
                for rr in range(obj2_mat.shape[0]):
                    for cc in range(obj2_mat.shape[1]):
                        if obj2_mat[rr, cc] != 0:
                            if test_grid[top2 + rr, left2 + cc] != 0:
                                # overlap
                                overlap = True
                                break
                    if overlap:
                        break

                if overlap:
                    continue

                # If no overlap, place it
                for rr in range(obj2_mat.shape[0]):
                    for cc in range(obj2_mat.shape[1]):
                        if obj2_mat[rr, cc] != 0:
                            test_grid[top2 + rr, left2 + cc] = obj2_mat[rr, cc]

                # Now check adjacency
                # Find objects in test_grid
                objs = find_connected_objects(test_grid, diagonal_connectivity=False, background=0, monochromatic=True)
                # We expect exactly 2 objects if there's no adjacency => we want them separate.
                # If they connected into one big object, we'd have 1 object. If we have more than 2,
                # it means something else got formed, but that shouldn't happen here.
                if len(objs) == 2:
                    # success, we can accept test_grid
                    return test_grid
            # If we got here, we can't place the second object in a valid way -> fail
            raise ValueError("Failed to place second object without overlap/connectivity")

        final_grid = retry(
            place_two_objects,
            predicate=lambda x: True,  # if we got it, it's good
            max_attempts=50
        )
        return final_grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain.
        """
        # We'll do the transformation:
        # 1) If the object colors are object_color1 and object_color2, fill all zeros with fill_color1
        # 2) If the object colors are object_color3 and object_color4, fill all zeros with fill_color2

        # First, find unique non-zero colors in the grid. There should be exactly two, each forming a 4-connected object.
        # We interpret them and fill accordingly.

        out_grid = np.copy(grid)

        # Extract set of non-zero colors
        colors = set(out_grid.flatten()) - {0}

        # We expect exactly two colors
        if colors == {taskvars['object_color1'], taskvars['object_color2']}:
            # fill empty with fill_color1
            out_grid[out_grid == 0] = taskvars['fill_color1']
        elif colors == {taskvars['object_color3'], taskvars['object_color4']}:
            # fill empty with fill_color2
            out_grid[out_grid == 0] = taskvars['fill_color2']
        else:
            # If for some reason we got something else, we do nothing 
            # (in principle shouldn't happen if the generator is correct).
            pass

        return out_grid

    def create_grids(self):
        """
        1) Create random distinct values for object_color1, object_color2, object_color3, 
           object_color4, fill_color1, fill_color2 (each between 1..9).
        2) Create 3-4 train pairs, ensuring at least one uses color1+color2, at least one uses color3+color4.
        3) Create 1 test pair.
        4) Return (taskvars, TrainTestData).
        """

        # Step 1: generate distinct colors
        # We'll pick 6 distinct numbers from 1..9
        all_candidates = list(range(1, 10))
        random.shuffle(all_candidates)
        chosen = all_candidates[:6]
        object_color1, object_color2, object_color3, object_color4, fill_color1, fill_color2 = chosen

        taskvars = {
            'object_color1': object_color1,
            'object_color2': object_color2,
            'object_color3': object_color3,
            'object_color4': object_color4,
            'fill_color1': fill_color1,
            'fill_color2': fill_color2
        }

        # We want 3-4 train pairs. We'll randomly pick either 3 or 4
        n_train = random.choice([3, 4])

        # We must ensure at least one train uses (obj_col1, obj_col2) and one uses (obj_col3, obj_col4).
        # Let's plan: force #1 uses pair1, #2 uses pair2, then the rest random.

        def make_example(color_choice: int):
            # color_choice == 1 => (obj_col1, obj_col2)
            # color_choice == 2 => (obj_col3, obj_col4)
            input_grid = self.create_input(taskvars, {'color_choice': color_choice})
            output_grid = self.transform_input(input_grid, taskvars)
            return {'input': input_grid, 'output': output_grid}

        # Create train data
        train_data = []
        # Force first pair
        train_data.append(make_example(1))
        # Force second pair
        train_data.append(make_example(2))
        # For the remaining n_train-2, pick randomly
        for _ in range(n_train - 2):
            choice = random.choice([1, 2])
            train_data.append(make_example(choice))

        # Now create 2 test pairs, one with color_choice 1 and the other with color_choice 2
        test_data = [make_example(1), make_example(2)]

        all_data = TrainTestData(train=train_data, test=test_data)
        return taskvars, all_data



