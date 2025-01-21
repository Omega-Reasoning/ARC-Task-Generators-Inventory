# filename: my_arc_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# We may use the transformation library in transform_input()
from transformation_library import find_connected_objects

class TaskXSzztcED8SwAgX3V7iqTbPGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) The input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They contain only a single {color('cell_color1')} or {color('cell_color2')} cell in the first row, while all other cells remain empty.",
            "The position of the colored cell can vary across examples."
        ]
        
        # 2) The transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and adding new colored cells.",
            "The new cells are connected diagonally, starting from the colored cell in the first row, with each new cell connecting to the bottom-right edge of the previous one, extending until the matrix boundary is reached.",
            "The color of the new cells is determined by the color of the original cell in the input grid.",
            "If the original cell is {color('cell_color1')}, the newly added cells are {color('cell_color3')}; otherwise, they are {color('cell_color4')}."
        ]
        
        # 3) Call the superclass constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self):
        """
        We create the dictionary of task-level variables cell_color1, cell_color2, cell_color3, cell_color4,
        ensuring they are distinct. Then we generate 3-6 training examples and 2 test examples.
        
        We guarantee that:
        1) At least one training grid has the colored cell with cell_color1,
           and at least one training grid has the colored cell with cell_color2.
        2) At least one test grid has the colored cell with cell_color1,
           and at least one test grid has the colored cell with cell_color2.
        """

        # Choose 4 distinct colors between 1..9
        colors = random.sample(range(1, 10), 4)  # distinct
        cell_color1, cell_color2, cell_color3, cell_color4 = colors

        # Create our taskvars dictionary
        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2,
            "cell_color3": cell_color3,
            "cell_color4": cell_color4
        }

        # Decide how many train examples we want: between 3 and 6
        nr_train = random.randint(3, 6)
        nr_test = 2  # we want two test examples

        train_pairs = []
        used_color1 = False
        used_color2 = False

        # We'll keep generating examples until we have nr_train total,
        # but we ensure at least one uses cell_color1 and one uses cell_color2.
        while len(train_pairs) < nr_train:
            # If we still haven't used color1 or color2, force that color.
            if not used_color1:
                chosen_color = cell_color1
                used_color1 = True
            elif not used_color2:
                chosen_color = cell_color2
                used_color2 = True
            else:
                # Random choice for the rest
                chosen_color = random.choice([cell_color1, cell_color2])

            # Create an input grid with chosen_color in the first row
            inp = self.create_input(taskvars, {"chosen_color": chosen_color})
            outp = self.transform_input(inp, taskvars)
            train_pairs.append({"input": inp, "output": outp})

        # If for some reason we still haven't used color1 or color2, fix that
        # (very unlikely with the logic above, but just to be safe).
        if not used_color1:
            inp = self.create_input(taskvars, {"chosen_color": cell_color1})
            outp = self.transform_input(inp, taskvars)
            train_pairs[0] = {"input": inp, "output": outp}  # forcibly place
            used_color1 = True

        if not used_color2:
            inp = self.create_input(taskvars, {"chosen_color": cell_color2})
            outp = self.transform_input(inp, taskvars)
            # If we replaced index 0 above, let's replace index 1 here.
            # If that doesn't exist, we append (for safety).
            if len(train_pairs) > 1:
                train_pairs[1] = {"input": inp, "output": outp}
            else:
                train_pairs.append({"input": inp, "output": outp})
            used_color2 = True

        # Now create test pairs (2 examples),
        # guaranteeing one with color1, one with color2.
        test_pairs = []
        for test_color in [cell_color1, cell_color2]:
            inp = self.create_input(taskvars, {"chosen_color": test_color})
            outp = self.transform_input(inp, taskvars)
            test_pairs.append({"input": inp, "output": outp})

        data = {
            "train": train_pairs,
            "test": test_pairs
        }

        return taskvars, data

    def create_input(self, taskvars, gridvars):
        """
        Creates a single input grid. According to the puzzle spec:
         1. The grid can have 5..30 rows and 5..30 columns.
         2. There's exactly one colored cell in the first row, 
            color is either cell_color1 or cell_color2.
         3. The position of the colored cell is between column 0 and (width - 3).
         4. All other cells remain 0.
        """
        chosen_color = gridvars["chosen_color"]
        
        # random rows, random columns
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        # position is in the first row, col in [0 .. cols-3]
        # (To ensure there's at least some horizontal space to demonstrate the diagonal effect)
        max_col_for_placement = max(cols - 3, 0)
        colpos = random.randint(0, max_col_for_placement)

        # Create the grid
        grid = np.zeros((rows, cols), dtype=int)
        grid[0, colpos] = chosen_color

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        """
        Transform logic:
         1. Copy the input grid.
         2. Identify the single non-zero cell in the first row (it will be color1 or color2).
         3. Starting from that cell, move diagonally down-right until the boundary is reached.
         4. Fill each diagonal cell (beyond the first) with color3 if the original cell was color1,
            or color4 if the original cell was color2.
        """
        cell_color1 = taskvars['cell_color1']
        cell_color2 = taskvars['cell_color2']
        cell_color3 = taskvars['cell_color3']
        cell_color4 = taskvars['cell_color4']

        # Make a copy of the grid
        output_grid = grid.copy()

        rows, cols = output_grid.shape

        # Locate the single colored cell in the first row
        # We know there's exactly one.
        row0 = output_grid[0]
        colored_indices = np.nonzero(row0)[0]
        if len(colored_indices) == 0:
            # Fallback if something unexpected, just return the same grid
            return output_grid

        start_col = colored_indices[0]
        original_color = row0[start_col]

        # Determine the new color
        if original_color == cell_color1:
            new_color = cell_color3
        else:
            new_color = cell_color4

        # Fill diagonal cells
        # We already have the first cell as original_color, so we start from row=1, col=start_col+1
        r, c = 1, start_col + 1
        while r < rows and c < cols:
            # Add new colored cell
            output_grid[r, c] = new_color
            r += 1
            c += 1

        return output_grid

