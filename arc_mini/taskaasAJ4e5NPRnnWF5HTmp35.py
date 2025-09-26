from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# Optional libraries (encouraged but not mandatory to use):
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class TaskaasAJ4e5NPRnnWF5HTmp35Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (as provided)
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a single L-shaped object, i.e., an object of the form [[c,0],[c,0],[c,c]] for a color c, while all remaining cells are empty (0).",
            "The L-shaped object is colored either {color('object_color1')} or {color('object_color2')}."
        ]
        # 2) Transformation reasoning chain (as provided)
        transformation_reasoning_chain = [
            "Output grids are of size {2*vars['rows']}x{2*vars['cols']}.",
            "They are created by transforming each cell in the input into a 2x2 block in the output, where empty cells become empty 2x2 blocks and colored cells become colored 2x2 blocks.",
            "All colored 2x2 blocks are identical, consisting of two colored cells that alternate across rows and columns.",
            "If the input contains {color('object_color1')} color, the two colors are {color('object_color1')} and {color('object_color3')}; otherwise, they are {color('object_color2')} and {color('object_color4')}.",
            "The pattern in each 2x2 block always starts with {color('object_color1')} or {color('object_color2')}."
        ]
        # 3) Call super().__init__()
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars,
                     gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain, given the task variables.
        Each grid has exactly one L-shaped object of color c, where c is either object_color1 or object_color2.
        The rest of the cells are 0 (empty). The L shape is:
          [c, 0]
          [c, 0]
          [c, c]
        and is placed at a random valid position in the grid.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        # color_choice is either object_color1 or object_color2
        color_choice = gridvars['object_color_choice']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # The L shape is fixed size 3x2:
        # shape:
        #   row0: [c, 0]
        #   row1: [c, 0]
        #   row2: [c, c]
        L_height, L_width = 3, 2

        # Compute a random top-left position where the L-shape will fit
        # ensure there's space for 3 rows and 2 columns
        r_start = random.randint(0, rows - L_height)
        c_start = random.randint(0, cols - L_width)

        # Place the L shape
        grid[r_start + 0, c_start + 0] = color_choice
        grid[r_start + 0, c_start + 1] = 0
        grid[r_start + 1, c_start + 0] = color_choice
        grid[r_start + 1, c_start + 1] = 0
        grid[r_start + 2, c_start + 0] = color_choice
        grid[r_start + 2, c_start + 1] = color_choice

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid into the output grid by:
          1) Doubling the dimensions (2*rows, 2*cols).
          2) For each non-zero cell of color c in input:
             - If c == object_color1, fill the corresponding 2x2 block with the pattern
               [[object_color1, object_color3],
                [object_color3, object_color1]]
             - If c == object_color2, fill the corresponding 2x2 block with the pattern
               [[object_color2, object_color4],
                [object_color4, object_color2]]
          3) Zeros remain empty (0) in the corresponding 2x2 block.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        object_color1 = taskvars['object_color1']
        object_color2 = taskvars['object_color2']
        object_color3 = taskvars['object_color3']
        object_color4 = taskvars['object_color4']
        
        out_rows = 2 * rows
        out_cols = 2 * cols
        out_grid = np.zeros((out_rows, out_cols), dtype=int)

        for r in range(rows):
            for c in range(cols):
                color_in = grid[r, c]
                # The 2x2 block in the output
                R = 2 * r
                C = 2 * c

                if color_in == 0:
                    # Remain empty
                    pass
                elif color_in == object_color1:
                    # 2x2 pattern: color1, color3 / color3, color1
                    out_grid[R,   C]   = object_color1
                    out_grid[R,   C+1] = object_color3
                    out_grid[R+1, C]   = object_color3
                    out_grid[R+1, C+1] = object_color1
                elif color_in == object_color2:
                    # 2x2 pattern: color2, color4 / color4, color2
                    out_grid[R,   C]   = object_color2
                    out_grid[R,   C+1] = object_color4
                    out_grid[R+1, C]   = object_color4
                    out_grid[R+1, C+1] = object_color2
                else:
                    # If ever there's some unexpected color, do nothing or skip
                    pass
        
        return out_grid

    def create_grids(self):
        """
        This method creates the dictionary of task variables (vars) and the
        train/test pairs. The constraints from the general instructions are:
          * rows and cols between 5 and 30
          * object_color1, object_color2, object_color3, object_color4 ∈ {1..9}, all distinct
          * We produce multiple training inputs, each with a single L-shaped object, color either object_color1 or object_color2.
          * We produce test inputs similarly. 
          * At least one train and one test must have color1; 
            and at least one train and one test must have color2.
          * Each input grid must place the L shape in a different position to encourage variety.
        """
        # 1) Choose distinct colors
        all_colors = list(range(1, 10))  # 1..9
        random.shuffle(all_colors)
        object_color1, object_color2, object_color3, object_color4 = all_colors[:4]

        # 2) Choose rows, cols
        rows = random.randint(5, 30)   # smaller range to see results easily
        cols = random.randint(5, 30)

        # We store them in the taskvars dictionary
        taskvars = {
            'rows': rows,
            'cols': cols,
            'object_color1': object_color1,
            'object_color2': object_color2,
            'object_color3': object_color3,
            'object_color4': object_color4
        }

        # We want at least 2 or 3 training examples, ensuring:
        #  - at least one uses object_color1
        #  - at least one uses object_color2
        # We'll produce exactly 3 training examples for variety. 
        # Among them:
        #   Example 1 -> color1
        #   Example 2 -> color2
        #   Example 3 -> color1 or color2 (random choice)
        train_colors = [
            object_color1,
            object_color2,
            random.choice([object_color1, object_color2])
        ]

        train_data = []
        for c in train_colors:
            gridvars = {'object_color_choice': c}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append(GridPair(input=input_grid, output=output_grid))

        # For tests, we want at least one test with color1 and one test with color2
        # We'll produce 2 test examples:
        #   Test 1 -> color1
        #   Test 2 -> color2
        test_colors = [object_color1, object_color2]
        test_data = []
        for c in test_colors:
            gridvars = {'object_color_choice': c}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            test_data.append(GridPair(input=input_grid, output=output_grid))

        train_test_data = TrainTestData(train=train_data, test=test_data)
        return taskvars, train_test_data


