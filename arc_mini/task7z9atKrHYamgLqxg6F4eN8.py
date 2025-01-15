from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random


class Task7z9atKrHYamgLqxg6F4eN8Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Define the input reasoning chain exactly as given
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "The input grids have a checkerboard pattern with two alternating colors across rows and columns, forming a structured layout.",
            "The two colors used in the pattern are either {color('object_color1')} and {color('object_color2')} or {color('object_color3')} and {color('object_color4')}."
        ]
        
        # 2) Define the transformation reasoning chain exactly as given
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and coloring certain cells {color('object_color5')} based on a specific criteria.",
            "If the checkerboard pattern in the input grid uses {color('object_color1')} and {color('object_color2')} colors, cells in odd-numbered columns and even-numbered rows of the output grid are changed to {color('object_color5')}.",
            "If the checkerboard pattern in the input grid uses {color('object_color3')} and {color('object_color4')} colors, cells in even-numbered columns and odd-numbered rows of the output grid are changed to {color('object_color5')}."
        ]

        # 3) Call superclass constructor. 
        #    (Note: the instructions mention a 3rd parameter 'taskvars_definitions', but the provided base class does not have it.)
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid with a checkerboard pattern, using either (object_color1, object_color2)
        or (object_color3, object_color4). The choice of pair and the grid size come from gridvars.
        """
        rows = gridvars["rows"]
        cols = gridvars["cols"]
        # if is_pair12 is True, use object_color1 & object_color2, else use object_color3 & object_color4
        is_pair12 = gridvars["is_pair12"]
        
        color1 = taskvars["object_color1"] if is_pair12 else taskvars["object_color3"]
        color2 = taskvars["object_color2"] if is_pair12 else taskvars["object_color4"]

        grid = np.zeros((rows, cols), dtype=int)
        
        # Build a checkerboard pattern: 
        # at position (r,c), if (r+c)%2==0 -> color1, else color2
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    grid[r, c] = color1
                else:
                    grid[r, c] = color2

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
          * If the checkerboard uses (object_color1, object_color2), then color certain cells with object_color5
            in odd-numbered columns and even-numbered rows (1-based).
          * If the checkerboard uses (object_color3, object_color4), then color certain cells with object_color5
            in even-numbered columns and odd-numbered rows (1-based).
        """
        # We'll detect which pair of colors is in the grid by checking a corner cell, for instance (0,0).
        # If (0,0) is object_color1 or object_color2, we assume it's the "pair12" pattern. Otherwise the "pair34" pattern.
        # This is safe for our generation approach, which only uses exactly two colors in the grid.
        c1, c2, c3, c4, c5 = (
            taskvars["object_color1"],
            taskvars["object_color2"],
            taskvars["object_color3"],
            taskvars["object_color4"],
            taskvars["object_color5"],
        )
        
        top_left_color = grid[0, 0]
        # Heuristic: if the top-left color is c1 or c2, we assume "pair12"
        uses_pair12 = (top_left_color == c1 or top_left_color == c2)
        
        # Make a copy for the output
        output = np.copy(grid)
        rows, cols = output.shape

        if uses_pair12:
            # "cells in odd-numbered columns and even-numbered rows" => 1-based indexing
            # => row+1 is even => row is odd in 0-based; col+1 is odd => col is even in 0-based
            for r in range(rows):
                for c in range(cols):
                    if (r % 2 == 1) and (c % 2 == 0):
                        output[r, c] = c5
        else:
            # "cells in even-numbered columns and odd-numbered rows" => 1-based indexing
            # => row+1 is odd => row is even in 0-based; col+1 is even => col is odd in 0-based
            for r in range(rows):
                for c in range(cols):
                    if (r % 2 == 0) and (c % 2 == 1):
                        output[r, c] = c5

        return output

    def create_grids(self):
        """
        Create 3-6 training pairs and 2 test pairs, each with distinct sizes. 
        Also ensure at least one training example with the pair (object_color1, object_color2) 
        and one with (object_color3, object_color4), 
        and the same for the two test pairs. 
        The colors object_color1..5 must be distinct and between 1..9.
        """
        # First: pick distinct colors 1..9 for the 5 color variables
        all_possible_colors = list(range(1, 10))  # 1..9
        random.shuffle(all_possible_colors)
        c1, c2, c3, c4, c5 = all_possible_colors[:5]

        taskvars = {
            "object_color1": c1,
            "object_color2": c2,
            "object_color3": c3,
            "object_color4": c4,
            "object_color5": c5,
        }

        # We'll produce exactly 4 training examples and 2 test examples for clarity,
        # abiding by the constraints. Sizes must all be distinct.
        # Let's pick 6 distinct sizes from [5..15], for instance.

        possible_sizes = []
        for r in range(5, 16):
            for c in range(5, 16):
                possible_sizes.append((r, c))
        random.shuffle(possible_sizes)

        used_sizes = []
        
        # We want 4 training pairs:
        #   2 with (object_color1, object_color2), 2 with (object_color3, object_color4).
        # Then 2 test pairs: 
        #   1 with (object_color1, object_color2), 1 with (object_color3, object_color4).

        train = []
        test = []

        def pick_unique_size():
            sz = possible_sizes.pop()
            used_sizes.append(sz)
            return sz

        # Create 2 training inputs for c1,c2
        for _ in range(2):
            rows, cols = pick_unique_size()
            gridvars = {
                "rows": rows,
                "cols": cols,
                "is_pair12": True
            }
            inp = self.create_input(taskvars, gridvars)
            outp = self.transform_input(inp, taskvars)
            train.append(GridPair(input=inp, output=outp))

        # Create 2 training inputs for c3,c4
        for _ in range(2):
            rows, cols = pick_unique_size()
            gridvars = {
                "rows": rows,
                "cols": cols,
                "is_pair12": False
            }
            inp = self.create_input(taskvars, gridvars)
            outp = self.transform_input(inp, taskvars)
            train.append(GridPair(input=inp, output=outp))

        # Create 1 test input for c1,c2
        rows, cols = pick_unique_size()
        gridvars = {
            "rows": rows,
            "cols": cols,
            "is_pair12": True
        }
        inp = self.create_input(taskvars, gridvars)
        outp = self.transform_input(inp, taskvars)  # We'll store the correct output 
        test.append(GridPair(input=inp, output=outp))

        # Create 1 test input for c3,c4
        rows, cols = pick_unique_size()
        gridvars = {
            "rows": rows,
            "cols": cols,
            "is_pair12": False
        }
        inp = self.create_input(taskvars, gridvars)
        outp = self.transform_input(inp, taskvars)
        test.append(GridPair(input=inp, output=outp))

        train_test_data: TrainTestData = {
            "train": train,
            "test": test
        }

        return taskvars, train_test_data



