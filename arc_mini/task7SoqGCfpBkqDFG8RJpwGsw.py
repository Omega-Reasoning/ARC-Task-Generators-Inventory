from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class Task7SoqGCfpBkqDFG8RJpwGswGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains only one {color('object_color1')} and one {color('object_color2')} object, with the {color('object_color1')} object specifically being a single-cell-wide ectangular block.",
            "The objects are 4-way connected to each other, with the remaining cells being empty (0)."
        ]
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and changing all {color('object_color1')} cells to {color('object_color3')} that share at least one side (not diagonal) with a {color('object_color2')} cell."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]

        height = random.randint(5, 30)
        width = random.randint(5, 30)
        grid = np.zeros((height, width), dtype=int)

        # Create a vertical column for object1
        obj1_size = random.randint(3, min(height, 6))  # At least 3 cells, max 6
        r1 = random.randint(0, height - obj1_size)
        c1 = random.randint(0, width - 1)

        for i in range(obj1_size):
            grid[r1 + i, c1] = color1  # Vertical placement

        # Try placing object2 next to object1 with at least one 4-way connection
        MAX_ATTEMPTS = 100
        found_valid = False

        for _ in range(MAX_ATTEMPTS):
            obj2_h = random.randint(3, min(5, height))
            obj2_w = random.randint(3, min(5, width))

            obj2_array = create_object(
                height=obj2_h,
                width=obj2_w,
                color_palette=color2,
                contiguity=Contiguity.FOUR,
                background=0
            )

            if np.count_nonzero(obj2_array) < 3:
                continue  

            for _inner in range(MAX_ATTEMPTS):
                r2 = random.randint(0, height - obj2_h)
                c2 = random.randint(0, width - obj2_w)

                grid_candidate = grid.copy()
                overlap = False
                for rr in range(obj2_h):
                    for cc in range(obj2_w):
                        if obj2_array[rr, cc] != 0 and grid_candidate[r2 + rr, c2 + cc] != 0:
                            overlap = True
                            break
                    if overlap:
                        break
                if overlap:
                    continue

                for rr in range(obj2_h):
                    for cc in range(obj2_w):
                        if obj2_array[rr, cc] != 0:
                            grid_candidate[r2 + rr, c2 + cc] = color2

                # Check for 4-way adjacency between color1 and color2
                adj_found = False
                for r in range(height):
                    for c in range(width):
                        if grid_candidate[r, c] == color1:
                            for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                rr, cc = r + dr, c + dc
                                if 0 <= rr < height and 0 <= cc < width and grid_candidate[rr, cc] == color2:
                                    adj_found = True
                                    break
                            if adj_found:
                                break
                    if adj_found:
                        break

                if adj_found:
                    grid = grid_candidate
                    found_valid = True
                    break

            if found_valid:
                return grid

        raise ValueError("Could not generate a valid grid with a vertical color1 object, one color2 object, and adjacency.")

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        color1 = taskvars["object_color1"]
        color2 = taskvars["object_color2"]
        color3 = taskvars["object_color3"]

        out = grid.copy()
        height, width = out.shape
        for r in range(height):
            for c in range(width):
                if out[r, c] == color1:
                    neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
                    for nr, nc in neighbors:
                        if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == color2:
                            out[r, c] = color3
                            break
        return out

    def create_grids(self):
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        c1, c2, c3 = available_colors[:3]

        taskvars = {
            "object_color1": c1,
            "object_color2": c2,
            "object_color3": c3,
        }

        nr_train = random.randint(3, 6)
        nr_test = 1

        train_test_data = self.create_grids_default(nr_train_examples=nr_train,
                                                    nr_test_examples=nr_test,
                                                    taskvars=taskvars)

        return taskvars, train_test_data



