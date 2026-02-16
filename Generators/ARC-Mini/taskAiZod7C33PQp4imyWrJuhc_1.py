from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# Optional (but encouraged) imports from Framework.transformation_library and input_library:
# from Framework.transformation_library import ...
# from Framework.input_library import ...

class TaskAiZod7C33PQp4imyWrJuhc_1Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "All input grids are squares.",
            "Let n be the height and width of an input grid, where n is an odd number.",
            "Each input grid contains (n+1)/2 same-colored cells, evenly spaced along the main diagonal (top-left to bottom-right), starting from (0,0) and ending at (n,n).",
            "The color of these cells can only be {color('cell_color1')}, {color('cell_color2')}, or {color('cell_color3')}."
        ]
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid, and for each colored cell in the main diagonal, all adjacent empty (0) cells (up, down, left, and right) are filled.",
            "The fill color is determined based on the color of the main diagonal cells.",
            "If the diagonal cells are {color('cell_color1')}, adjacent empty cells are filled with {color('fill_color1')}; if {color('cell_color2')}, they are filled with {color('fill_color2')}; if {color('cell_color3')}, they are filled with {color('fill_color3')}."
        ]
        # 3) Call super constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
          * n is an odd integer (the grid is n x n).
          * We place (n+1)//2 same-colored cells along the main diagonal, 
            from (0,0) to (n-1,n-1), skipping every other diagonal position (step=2).
        """
        n = gridvars["n"]
        diag_color = gridvars["diag_color"]
        grid = np.zeros((n, n), dtype=int)
        # Place (n+1)//2 evenly spaced diagonal cells
        for i in range(0, n, 2):
            grid[i, i] = diag_color
        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid by filling adjacent empty cells around each main-diagonal colored cell.
        If diagonal cell = cell_color1 => fill with fill_color1
                          cell_color2 => fill with fill_color2
                          cell_color3 => fill with fill_color3
        """
        out = grid.copy()
        n = out.shape[0]

        for i in range(n):
            color_here = out[i, i]
            # Determine fill color based on diagonal cell color
            if color_here == taskvars["cell_color1"]:
                fill_color = taskvars["fill_color1"]
            elif color_here == taskvars["cell_color2"]:
                fill_color = taskvars["fill_color2"]
            elif color_here == taskvars["cell_color3"]:
                fill_color = taskvars["fill_color3"]
            else:
                # If it's not one of the three diagonal colors, skip
                continue

            # Fill adjacent empty cells (up, down, left, right)
            for (dr, dc) in [(0,1), (1,0), (-1,0), (0,-1)]:
                r, c = i + dr, i + dc
                if 0 <= r < n and 0 <= c < n and out[r, c] == 0:
                    out[r, c] = fill_color

        return out

    def create_grids(self):
        """
        Build the final train/test data for the ARC task.
         * We pick 6 distinct colors out of [1..9] to fill the variables:
             cell_color1, cell_color2, cell_color3, fill_color1, fill_color2, fill_color3
         * We pick 6 distinct odd numbers for n (each <= 29).
         * We create exactly 3 train pairs and 3 test pairs, each with a different n.
         * One train/test pair each uses diagonal color = cell_color1, cell_color2, cell_color3.
        """
        # Pick 6 distinct colors from 1..9
        all_colors = random.sample(range(1, 10), 6)
        cell_color1, cell_color2, cell_color3, fill_color1, fill_color2, fill_color3 = all_colors

        # Collect them in a dictionary to be used in transformation templates
        taskvars = {
            "cell_color1": cell_color1,
            "cell_color2": cell_color2,
            "cell_color3": cell_color3,
            "fill_color1": fill_color1,
            "fill_color2": fill_color2,
            "fill_color3": fill_color3
        }

        # Pick 6 distinct odd n in [3..29]
        odd_candidates = [x for x in range(3, 30, 2)]
        chosen_ns = random.sample(odd_candidates, 6)
        
        # Helper to create a single example
        def make_example(n, diag_color):
            gridvars = {"n": n, "diag_color": diag_color}
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)
            return {"input": inp, "output": out}

        # We want:
        #   Train examples (3 total):
        #     1) diagonal color = cell_color1,  n = chosen_ns[0]
        #     2) diagonal color = cell_color2,  n = chosen_ns[1]
        #     3) diagonal color = cell_color3,  n = chosen_ns[2]
        #   Test examples (3 total):
        #     1) diagonal color = cell_color1,  n = chosen_ns[3]
        #     2) diagonal color = cell_color2,  n = chosen_ns[4]
        #     3) diagonal color = cell_color3,  n = chosen_ns[5]
        train_data = [
            make_example(chosen_ns[0], cell_color1),
            make_example(chosen_ns[1], cell_color2),
            make_example(chosen_ns[2], cell_color3),
        ]
        test_data = [
            make_example(chosen_ns[3], cell_color1),
            make_example(chosen_ns[4], cell_color2),
            make_example(chosen_ns[5], cell_color3),
        ]

        return taskvars, {"train": train_data, "test": test_data}


