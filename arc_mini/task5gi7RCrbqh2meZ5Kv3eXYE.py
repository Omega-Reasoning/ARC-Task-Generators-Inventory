from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optional imports from the libraries you provided
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects, GridObjects

class Task5gi7RCrbqh2meZ5Kv3eXYEGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input (observation) reasoning chain
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains three objects, where an object is a 4-way connected group of cells, of the same color.",
            f"The three objects are rectangular shaped and each object is of a different color. The colors are {{color('object_color1')}}, {{color('object_color2')}} and {{color('object_color3')}}.",
            "The remaining cells are empty (0)."
        ]

        # 2. Transformation reasoning chain
        reasoning_chain = [
            "To construct the output grid, copy the input grid.",
            "Transform each object to {color('object_color4')} color, except the {color('object_color2')} object, which remains unchanged."
        ]

        # 3. Call superclass constructor
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        Create 3 train pairs and 1 test pair, each with 3 rectangular objects
        of distinct colors (object_color1, object_color2, object_color3).
        In the transformation, all but the object_color2 are recolored to object_color4.
        
        Returns:
            Tuple[Dict[str, Any], TrainTestData]
        """
        # --- 1) Pick distinct colors in [1..9] for the 4 required color variables ---
        available_colors = list(range(1, 10))  # from 1 to 9
        random.shuffle(available_colors)
        taskvars = {
            "object_color1": available_colors[0],
            "object_color2": available_colors[1],
            "object_color3": available_colors[2],
            "object_color4": available_colors[3]
        }

        # --- 2) Create multiple train examples and 1 test example ---
        #     We want some variety in grid sizes, so let's pick them randomly.
        #     You can adjust the logic (e.g., ensure the test size differs from train sizes, etc.)
        num_train_examples = 3
        num_test_examples = 1

        train_data = []
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, gridvars={})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})

        test_data = []
        for _ in range(num_test_examples):
            input_grid = self.create_input(taskvars, gridvars={})
            output_grid = self.transform_input(input_grid, taskvars)
            test_data.append({'input': input_grid, 'output': output_grid})

        # Return the variables and the train/test data
        return taskvars, {"train": train_data, "test": test_data}

    def create_input(self, taskvars, gridvars):
        """
        Creates a single input grid containing exactly 3 rectangular objects
        in different colors: object_color1, object_color2, object_color3.
        
        The objects do not overlap and are strictly 4-way connected rectangles.
        The rest of the grid cells are 0.
        """
        # 1) Randomly choose grid size between 7 and 15
        height = random.randint(7, 30)
        width = random.randint(7, 30)

        # 2) Initialize empty grid
        grid = np.zeros((height, width), dtype=int)

        # 3) Place 3 non-overlapping rectangles of random sizes
        #    that fit entirely in the grid.
        #    We'll do this by:
        #    - picking a random row/col start
        #    - picking a random rectangle height/width
        #    - filling with the color
        colors = [
            taskvars['object_color1'],
            taskvars['object_color2'],
            taskvars['object_color3']
        ]
        random.shuffle(colors)  # random order of placing

        placed_rects = []
        for color in colors:
            grid = self._place_random_rectangle(grid, color, placed_rects)

        return grid

    def _place_random_rectangle(self, grid, color, placed_rects, max_attempts=100):
        """
        Place one rectangular object of 'color' in the empty grid in a random position.
        Make sure it doesn't overlap previously placed rectangles in placed_rects.
        'placed_rects' is a list of (r0, r1, c0, c1) bounding boxes of already placed rectangles.
        """
        rows, cols = grid.shape

        for _ in range(max_attempts):
            # random rectangle size (at least 1 cell, up to half or so for variety)
            rect_height = random.randint(1, max(1, rows // 2))
            rect_width = random.randint(1, max(1, cols // 2))

            # random top-left corner
            r0 = random.randint(0, rows - rect_height)
            c0 = random.randint(0, cols - rect_width)
            r1 = r0 + rect_height - 1
            c1 = c0 + rect_width - 1

            # check overlap with existing rectangles
            overlap = any(not (r1 < pr0 or r0 > pr1 or c1 < pc0 or c0 > pc1) 
                          for (pr0, pr1, pc0, pc1) in placed_rects)

            if not overlap:
                # Place the rectangle
                grid[r0:r1 + 1, c0:c1 + 1] = color
                placed_rects.append((r0, r1, c0, c1))
                return grid

        # If we fail to place after max_attempts, just return the grid unchanged
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        1. Copy the grid
        2. Recolor each object to object_color4 except the object whose color is object_color2.
        """
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        color3 = taskvars['object_color3']
        color4 = taskvars['object_color4']

        output_grid = grid.copy()

        # We can either do a simple recoloring (cell by cell),
        # or find connected objects and recolor them.  
        # Because we know the input has exactly 3 rectangular objects
        # with distinct colors, a cell-by-cell approach is straightforward.
        # 
        # We'll do a cell-based approach:
        #   If cell color is color1 or color3 => recolor to color4
        #   If cell color is color2 => keep color2 (unchanged)
        #   If cell color is 0 => keep 0
        # 
        # Or, if you like object detection, you can do it with the library:
        #
        #   objects = find_connected_objects(output_grid, diagonal_connectivity=False, background=0, monochromatic=True)
        #   for obj in objects:
        #       # check object color (since these are rectangles, each obj is uniform color)
        #       c = next(iter(obj.colors))
        #       if c == color2:
        #           # unchanged
        #           continue
        #       else:
        #           # recolor
        #           for (r,co,_) in obj.cells:
        #               output_grid[r, co] = color4
        #
        # But let's do cell-based for brevity:
        rows, cols = output_grid.shape
        for r in range(rows):
            for c in range(cols):
                if output_grid[r, c] in (color1, color3):
                    output_grid[r, c] = color4
                # color2 => unchanged
                # 0 => unchanged

        return output_grid


