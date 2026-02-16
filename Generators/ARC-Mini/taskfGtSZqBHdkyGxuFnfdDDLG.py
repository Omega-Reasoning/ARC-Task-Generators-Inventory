from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Importing from input and transformation libraries
from Framework.input_library import create_object, retry, Contiguity
from Framework.input_library import enforce_object_width, enforce_object_height
from Framework.input_library import random_cell_coloring
from Framework.transformation_library import find_connected_objects

class TaskfGtSZqBHdkyGxuFnfdDDLGGenerator(ARCTaskGenerator):
    def __init__(self):
        """
        Initializes the ARC-AGI task generator with pre-defined reasoning chains.
        """
        # Define the input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain exactly two colored (1-9) objects, where an object is 4-way connected group of cells of the same color.",
            "Each object has a different color, with one located in the top half and the other in the bottom half of the grid, while the middle row remains empty (0)."
        ]
        
        # Define the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid, removing the object in the bottom half, and reflecting the object in the top half onto the bottom half, using the middle row as the line of reflection.",
            "The reflected object should have the same color as the object in the top half."
        ]

        # Call the parent class constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        Generate task variables and training/testing grid pairs.
        """
        # Randomly select an odd number of rows (5 to 30) and columns (5 to 30)
        rows = random.randrange(5, 31, 2)  # Ensures rows is odd
        cols = random.randint(5, 30)

        # Store the chosen values as task variables
        taskvars = {
            "rows": rows,
            "cols": cols
        }

        # Generate between 3 to 6 training pairs and 1 test pair
        nr_train = random.randint(3, 6)
        nr_test = 1

        # Use the default train/test grid creation method
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid following the specified constraints.
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        mid = rows // 2  # The middle row index

        # Choose two different colors
        color_top = random.randint(1, 9)
        color_bottom = random.choice([c for c in range(1, 10) if c != color_top])

        # Initialize an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # Create a 4-way connected object for the top half
        top_half = create_object(mid, cols, color_top, contiguity=Contiguity.FOUR)
        grid[:mid, :] = top_half

        # Create a 4-way connected object for the bottom half
        bottom_half = create_object(rows - mid - 1, cols, color_bottom, contiguity=Contiguity.FOUR)
        grid[mid + 1:, :] = bottom_half

        # Ensure middle row remains empty
        grid[mid, :] = 0

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid into the output grid based on the reasoning chain.
        """
        rows = taskvars["rows"]
        cols = taskvars["cols"]
        mid = rows // 2  # Middle row index

        # Copy the input grid
        out_grid = grid.copy()

        # Find objects in the grid
        objects = find_connected_objects(out_grid, diagonal_connectivity=False, background=0, monochromatic=True)

        # Expect exactly two objects (top and bottom)
        if len(objects) != 2:
            return out_grid  # Return unchanged grid if unexpected number of objects found

        # Sort objects by their vertical position
        objects_sorted = sorted(objects, key=lambda obj: min(r for (r, c, col) in obj.cells))
        top_obj = objects_sorted[0]
        bottom_obj = objects_sorted[1]

        # Remove bottom object
        for (r, c, col) in bottom_obj.cells:
            out_grid[r, c] = 0

        # Reflect the top object onto the bottom half
        for (r, c, col) in top_obj.cells:
            new_r = 2 * mid - r  # Reflect row
            out_grid[new_r, c] = col

        return out_grid

