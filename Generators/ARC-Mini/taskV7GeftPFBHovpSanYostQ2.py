# my_arc_task_generator.py
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optionally import from Framework.transformation_library or input_library if desired:
# from Framework.transformation_library import ...
# from Framework.input_library import ...

class TaskV7GeftPFBHovpSanYostQ2Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each row has a pair of single-colored cells, while the rest of the cells in the row are empty (0).",
            "The two colored cells in each row are always disconnected.",
            "The color of the two cells varies across different rows."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid.",
            "In each row, the empty (0) cells between the two colored cells are filled with the same color as the two cells."
        ]
        
        # 3) Call parent constructor (you can pass an empty dict or None if not using taskvars_definitions explicitly)
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create a grid where each row has exactly two disconnected colored cells,
        i.e. there is at least one empty cell between them. The color of these
        two cells is the same within each row, but may differ across rows.
        """
        # Random grid dimensions (within ARC specs)
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)
        
        # Initialize a rows x cols grid of zeros
        grid = np.zeros((rows, cols), dtype=int)
        
        # For each row, place exactly two colored cells of the same color
        for r in range(rows):
            color = random.randint(1, 9)
            # Ensure c2 >= c1 + 2 so cells are "disconnected" (i.e. not adjacent)
            c1 = random.randint(0, cols - 3)
            c2 = random.randint(c1 + 2, cols - 1)
            
            grid[r, c1] = color
            grid[r, c2] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Creates the output grid by copying the input grid. In each row,
        fill any empty cells between the two colored cells with that color.
        """
        output_grid = grid.copy()
        rows, cols = output_grid.shape
        
        # For each row, find exactly two colored cells and fill in-between
        for r in range(rows):
            # Identify all non-zero columns in this row
            nonzero_cols = np.where(output_grid[r] != 0)[0]
            # Expect exactly two colored cells
            if len(nonzero_cols) == 2:
                c1, c2 = nonzero_cols
                color = output_grid[r, c1]  # same color on both
                # Fill cells between c1 and c2 with color
                output_grid[r, c1+1:c2] = color
        
        return output_grid

    def create_grids(self):
        """
        Creates and returns the task variables (if any) and the train/test data.
        We produce 3-6 training grids and 1 test grid. For simplicity,
        we'll pick a random number of train examples between 3 and 6.
        """
        # No special cross-example task variables needed
        taskvars = {}
        
        # Choose how many training examples to generate
        nr_train = random.randint(3, 6)
        nr_test = 1  # Usually 1 test example
        
        # Use the default helper to produce train/test data
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        return taskvars, train_test_data



