# my_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# We will use a few helper methods from the transformation library (optional).
from transformation_library import (
    find_connected_objects,
    GridObject,
    GridObjects
)

class Task5heLsQobb7VFGnV7RYJDciGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Initialize the input reasoning chain (list of strings)
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain a single 4-way connected object, that is a 2x2 square block.",
            "The object can only be either of {color('object_color1')} color or {color('object_color2')} color.",
            "All other cells are empty (0)."
        ]

        # 2. Initialize the transformation reasoning chain (list of strings)
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and placing a duplicate copy of the original object exactly next to it on the right.",
            "The duplicated copy has a different color than the original and this color depends on the original color of the object.",
            "If the original object is of  {color('object_color1')} color, the duplicated object is of {color('object_color3')} color.",
            "If the original object is of  {color('object_color2')} color, the duplicated object is of {color('object_color4')} color."
        ]

        # 3. Call super().__init__ with an empty dictionary for 'taskvars_definitions'
        #    (We don't provide a separate definitions structure here; we will define them on the fly in create_grids).
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain given the task and grid variables.
        We must place exactly one 4-way connected 2x2 square block of either object_color1 or object_color2
        at a random valid location in the grid, ensuring at least 2 empty columns on the right side of the object.
        """

        object_color1 = taskvars["object_color1"]
        object_color2 = taskvars["object_color2"]

        # Grid dimensions (between 5 and 30)
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Which color are we using for this grid? (Either color1 or color2)
        # If the user wants specific control, we can read it from gridvars. 
        # But here let's just choose from [object_color1, object_color2] if not already set.
        if "block_color" in gridvars:
            block_color = gridvars["block_color"]
        else:
            block_color = random.choice([object_color1, object_color2])

        # Place the 2x2 block so that there are at least 2 empty columns to the right
        # i.e. top-left corner can be at most col = (cols - 2 - 2)
        # Because the block itself has width 2, plus 2 columns must remain empty.
        max_col_for_block = cols - 2 - 2
        if max_col_for_block < 0:
            # If for some reason the random dims are too small, just bail out or re-generate
            # but given constraints, we'll keep it simple and let it pass for demonstration.
            max_col_for_block = 0

        r_start = random.randint(0, rows - 2)   # block has height = 2
        c_start = random.randint(0, max_col_for_block)

        # Paint the 2x2 block
        grid[r_start, c_start] = block_color
        grid[r_start, c_start + 1] = block_color
        grid[r_start + 1, c_start] = block_color
        grid[r_start + 1, c_start + 1] = block_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain, producing an output grid.
        Steps:
          1. Copy the input grid.
          2. Find the 2x2 block (it will be a single connected object).
          3. Duplicate it immediately to the right.
          4. Recolor the duplicate depending on the original block's color:
             - If original is object_color1 -> duplicate is object_color3
             - If original is object_color2 -> duplicate is object_color4
        """

        object_color1 = taskvars["object_color1"]
        object_color2 = taskvars["object_color2"]
        object_color3 = taskvars["object_color3"]
        object_color4 = taskvars["object_color4"]

        # 1. Copy the input grid
        output_grid = np.copy(grid)

        # 2. Detect the single 2x2 object
        #    We assume there is exactly one such object, 4-way connected, non-zero.
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=False)
        # There should be exactly 1 non-background object. 
        # If we want to be robust, we'd check or pick the largest if multiple, etc.
        if len(objects) == 0:
            # Unexpected edge case: no objects found
            return output_grid

        # We'll assume the first found object is the one 2x2 block
        original_obj = objects[0]

        # 3. We want to place the duplicate copy immediately to the right (i.e. shift by +2 columns from bounding box)
        #    Let's figure out the bounding box of the original
        (row_slice, col_slice) = original_obj.bounding_box
        block_height = row_slice.stop - row_slice.start
        block_width = col_slice.stop - col_slice.start

        # We'll compute how many columns to shift. 
        # We want it "exactly next to it on the right", meaning a shift of 'block_width' columns from the original bounding box.
        # But the puzzle statement says "exactly next to it on the right" – that generally means shift by exactly the block width (2).
        col_shift = block_width

        # 4. Determine the color to recolor the duplicate
        #    If the original block color is object_color1 -> object_color3
        #    If the original block color is object_color2 -> object_color4
        # We find the block's color by checking any cell in the object (since it's presumably uniform).
        # But the instructions mention it can be object_color1 OR object_color2. Let's do a quick check:
        unique_colors = list(original_obj.colors)
        if len(unique_colors) == 1:
            old_color = unique_colors[0]
        else:
            # In rare cases if it wasn't truly uniform, pick the first color or do some fallback.
            old_color = unique_colors[0]

        if old_color == object_color1:
            new_color = object_color3
        else:
            new_color = object_color4

        # We'll create the duplicated object by shifting the coordinates and recoloring
        duplicated_obj = GridObject(cells={
            (r, c + col_shift, new_color)
            for (r, c, col) in original_obj.cells
        })

        # Paste the duplicated object on the output grid
        duplicated_obj.paste(output_grid, overwrite=True, background=0)

        return output_grid

    def create_grids(self):
        """
        Create the dictionary of variables (object_color1, object_color2, object_color3, object_color4)
        and the train/test data. We need 3-6 train examples and 2 test examples total.
        
        Constraints from instructions:
        * object_color1, object_color2, object_color3, object_color4 are distinct and in [1..9].
        * We ensure at least one training input with a block of color1, and one with color2.
        * We also ensure the test set has one color1 block, one color2 block.
        * The 2x2 block must have at least 2 empty columns to its right in each grid.
        """

        # 1) Choose distinct colors for the 4 color variables
        all_colors = list(range(1, 10))  # 1..9
        random.shuffle(all_colors)
        object_color1, object_color2, object_color3, object_color4 = all_colors[:4]

        taskvars = {
            "object_color1": object_color1,
            "object_color2": object_color2,
            "object_color3": object_color3,
            "object_color4": object_color4
        }

        # We plan to produce 4 training examples and 2 test examples (total 6).
        # If you want a different distribution, adjust accordingly.
        # We'll ensure we get at least the coverage: 
        #   - 2 train with color1 block, 2 train with color2 block
        #   - 1 test with color1 block, 1 test with color2 block

        train_pairs = []
        test_pairs = []

        # Helper function to generate a single pair given a chosen block color
        def generate_pair(block_color):
            # create_input can take gridvars to force the block color
            inp = self.create_input(taskvars, {"block_color": block_color})
            out = self.transform_input(inp, taskvars)
            return {"input": inp, "output": out}

        # 2) Build train data
        # For variety, let's create 4 training grids:
        #   2 with color1, 2 with color2
        for _ in range(2):
            train_pairs.append(generate_pair(object_color1))
        for _ in range(2):
            train_pairs.append(generate_pair(object_color2))

        # 3) Build test data
        #   1 with color1, 1 with color2
        test_pairs.append(generate_pair(object_color1))
        test_pairs.append(generate_pair(object_color2))

        # Pack them into the TrainTestData structure
        train_test_data = {
            "train": train_pairs,
            "test": test_pairs
        }

        return taskvars, train_test_data


