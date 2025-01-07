from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optionally import from the provided libraries
from input_library import create_object, retry, Contiguity
from transformation_library import find_connected_objects

class Task7DfE84dHB2yXPndoSGQdA4Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) The "Input Reasoning Chain" from your instructions
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input matrix contains a single object, 4-way connected cells of the {color('object_color1')} color, with the rest of the cells being empty (0)."
        ]
        
        # 2) The "Transformation Reasoning Chain" from your instructions
        transformation_reasoning_chain = [
            "The output matrix is constructed by moving the object one column to the right, if there is an empty (0) column on the right and changing the color of the object from {color('object_color1')} to {color('object_color2')}."
        ]
        
        # 3) Call the superclass constructor with these two reasoning chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Randomly initializes the dictionary of task variables used in the template strings,
        and creates the train/test pairs.
        """
        # 1) Randomly pick two different colors (1..9)
        object_color1 = random.randint(1, 9)
        object_color2 = random.randint(1, 9)
        while object_color2 == object_color1:
            object_color2 = random.randint(1, 9)
        
        # This dictionary will be used in the template strings, e.g. {color('object_color1')}, {vars['object_color2']}, etc.
        taskvars = {
            "object_color1": object_color1,
            "object_color2": object_color2
        }
        
        # 2) Randomly decide how many train examples (3–6) and we make exactly 1 test example
        nr_train = random.randint(3, 6)
        nr_test = 1

        # 3) We can create the train/test data using a helper from the parent class
        #    that calls create_input() and transform_input() for each example.
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        
        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create a random input grid containing exactly one object of color taskvars['object_color1'].
        The rest of the cells are 0 (empty). We ensure that at least one column to the right is all 0s,
        so that the transformation can move the object by 1 column.
        """
        object_color1 = taskvars["object_color1"]
        
        # Randomly choose grid dimensions (in 5..10 for demonstration; up to 30..30 is also allowed)
        rows = random.randint(5, 10)
        cols = random.randint(5, 10)
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Strategy for guaranteeing we have at least one empty column on the right:
        #  - We pick a random boundary "split" so that everything to the right of that boundary is all zero
        #  - We only place our object to the left side of that boundary

        # We ensure that boundary is at least 1 column from the right
        # so that we do have a fully-empty column to the right
        if cols > 1:
            boundary = random.randint(0, cols - 2)
        else:
            boundary = 0  # trivial fallback if cols=1 (though typically we keep cols >= 5)

        width_left_side = boundary + 1  # columns that we may fill with the object
        height = rows

        # Now generate a single connected object of color object_color1 within that left side
        def object_generator() -> np.ndarray:
            # create_object from input_library can randomly fill a matrix with one or more color cells
            # but we want to ensure it is 4-way connected
            return create_object(
                height=height,
                width=width_left_side,
                color_palette=object_color1,
                contiguity=Contiguity.FOUR,
                background=0
            )
        
        # We'll just keep the generated object in a top-left subregion, then paste it back into grid
        obj_matrix = retry(
            object_generator,
            predicate=lambda mat: np.any(mat != 0)  # ensure at least one cell is colored
        )
        
        # Paste that object_matrix into the top-left corner of `grid`
        # or at some random offset within that left side
        row_offset = 0
        col_offset = 0
        # Optionally randomize row_offset if you want more vertical variety
        # But we must ensure the object won't accidentally cross beyond boundary
        # because we only generated `obj_matrix` up to width_left_side.
        
        for r in range(obj_matrix.shape[0]):
            for c in range(obj_matrix.shape[1]):
                if obj_matrix[r, c] != 0:
                    grid[r + row_offset, c + col_offset] = obj_matrix[r, c]

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid by:
          1) Finding the single object of color object_color1.
          2) If the next column to the right for that object is entirely 0's, move the object right by 1.
          3) Change the object color from object_color1 to object_color2.
        """
        object_color1 = taskvars["object_color1"]
        object_color2 = taskvars["object_color2"]

        # Copy the grid so we don't modify the original
        output = grid.copy()

        # 1) Find the single object that has color object_color1
        objects = find_connected_objects(output, diagonal_connectivity=False, background=0, monochromatic=True)
        # Filter out all objects that do NOT contain color object_color1
        matching_objects = objects.with_color(object_color1)
        
        if len(matching_objects) == 0:
            # No object found; just return the same grid. (Edge case)
            return output
        
        # We expect exactly one object that is color1, but let's just take the first
        obj = matching_objects[0]
        
        # 2) Check if the object can move one column to the right (i.e. bounding box + 1 is empty).
        #    To do that, we look at the bounding box of the object:
        box = obj.bounding_box
        row_slice, col_slice = box
        min_col = col_slice.start
        max_col = col_slice.stop - 1  # inclusive last col of bounding box
        rows, cols = output.shape

        # We'll see if the next column to the right of the bounding box is within the grid
        # and also if it's free (i.e. all 0) for those same row extents.
        can_move = False
        if max_col + 1 < cols:
            # check that for all rows in [row_slice], col = max_col+1 is 0
            # strictly speaking, if object only occupies certain rows in row_slice, we should check only those.
            # We'll just check all rows in the bounding box slice to keep it simple.
            col_to_check = max_col + 1
            rows_range = range(row_slice.start, row_slice.stop)
            can_move = all(output[r, col_to_check] == 0 for r in rows_range)

        # If we can move, we do so:
        if can_move:
            # We "cut" the object from the grid, shift it, and "paste" it back
            obj.cut(output, background=0)
            obj = obj.translate(dx=0, dy=1, grid_shape=(rows, cols))
            # 3) Change color to object_color2 in all its cells
            new_cells = set()
            for (r, c, _) in obj.cells:
                new_cells.add((r, c, object_color2))
            obj.cells = new_cells
            # Paste back
            obj.paste(output, overwrite=True, background=0)
        else:
            # We cannot move. We only recolor the object in place to object_color2
            obj.cut(output, background=0)
            new_cells = set()
            for (r, c, _) in obj.cells:
                new_cells.add((r, c, object_color2))
            obj.cells = new_cells
            obj.paste(output, overwrite=True, background=0)
        
        return output

