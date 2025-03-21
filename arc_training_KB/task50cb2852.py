from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

# We can import from input_library for the creation process:
from input_library import Contiguity, create_object, retry
# We can import from transformation_library for the transform (optional here, but possible)
from transformation_library import find_connected_objects

class Task50cb2852Generator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain (copy exactly from your input)
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain rectangular blocks, each made of same-colored cells, with a width and length greater than two.",
            "The colors of the blocks are {color('object_color1')}, {color('object_color2')}, and {color('object_color3')}.",
            "The number of rectangular blocks can range from two to four, depending on the grid size.",
            "If there are two or three blocks, each block has a unique color; if there are four blocks, one color is used for two blocks.",
            "Each colored block is completely separated from the others.",
            "The size and position of the colored blocks vary across examples.",
            "All other cells are empty (0)."
        ]

        # 2) Transformation reasoning chain (copy exactly from your input)
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all colored blocks.",
            "Once identified, change the colors of all interior cells in each colored block to {color('object_color4')}.",
            "This transformation results in rectangular blocks with a differently colored frame and interior."
        ]

        # 3) Call the super constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create a single input grid according to the input reasoning chain:
          - The grid is rows x cols with 2–4 rectangular blocks of min size 3x3.
          - Blocks are separated by at least 1 row or col of empty cells.
          - If block_count <= 3, use distinct colors from object_color1,2,3.
            If block_count=4, pick one color to repeat among object_color1,2,3.
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        color1 = taskvars['object_color1']
        color2 = taskvars['object_color2']
        color3 = taskvars['object_color3']

        # Decide how many blocks to place: 2..4
        # (We do not strictly tie block count to grid size here, but you can adapt as needed.)
        block_count = random.randint(2, 4)

        # If block_count < 4 => use distinct colors
        # If block_count == 4 => pick one color to duplicate
        if block_count < 4:
            block_colors = random.sample([color1, color2, color3], k=block_count)
        else:
            # pick one color to repeat
            chosen_for_repeat = random.choice([color1, color2, color3])
            others = [c for c in [color1, color2, color3] if c != chosen_for_repeat]
            # We want total of 4 blocks => 2 blocks of chosen_for_repeat, plus 2 distinct from others
            block_colors = [chosen_for_repeat, chosen_for_repeat] + others
            # Shuffle the final list
            random.shuffle(block_colors)

        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)

        # For each block, attempt to place a rectangle of at least 3x3 size
        # separated by at least 1 empty row/column from the others
        for color in block_colors:
            # We'll do random until we find a free region big enough
            def try_place_block() -> bool:
                # random block size:
                # ensure at least 3 in each dimension
                block_height = random.randint(3, rows // 2 + 2)
                block_width = random.randint(3, cols // 2 + 2)

                if block_height > rows or block_width > cols:
                    return False

                # random top-left
                max_row_start = rows - block_height
                max_col_start = cols - block_width

                if max_row_start < 0 or max_col_start < 0:
                    return False

                row_start = random.randint(0, max_row_start)
                col_start = random.randint(0, max_col_start)

                # Check if it is separated from existing blocks by 1 row/col
                # We'll define the bounding region (row_start-1 ... row_start+block_height+1) etc.
                # but first clamp them to valid indices.
                check_top = max(row_start - 1, 0)
                check_bot = min(row_start + block_height + 1, rows)
                check_left = max(col_start - 1, 0)
                check_right = min(col_start + block_width + 1, cols)

                region = grid[check_top:check_bot, check_left:check_right]
                if np.any(region != 0):
                    return False

                # If free, fill the rectangle
                grid[row_start:row_start+block_height, col_start:col_start+block_width] = color
                return True

            # We allow multiple tries for placing each block
            placed = False
            for _ in range(100):
                if try_place_block():
                    placed = True
                    break
            if not placed:
                # If we fail to place it, just skip – we might end up with fewer blocks
                break

        return grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid by:
          1) Copying it
          2) Identifying all colored blocks
          3) Re-coloring the interior cells of each block to object_color4
        """
        object_color4 = taskvars['object_color4']
        out_grid = grid.copy()

        # Identify blocks (connected objects). Since they are separated, each rectangle is a single object.
        objects = find_connected_objects(out_grid, diagonal_connectivity=False, background=0, monochromatic=False)

        for block in objects:
            # bounding box
            (row_slice, col_slice) = block.bounding_box
            r0, r1 = row_slice.start, row_slice.stop  # top, bottom
            c0, c1 = col_slice.start, col_slice.stop  # left, right

            # Recolor every "interior" cell: r in [r0+1..r1-2], c in [c0+1..c1-2], if it matches the block’s color
            # Actually the block could be multi-color if not strictly enforced,
            # but here we treat any cell in that bounding box with the same color as the block's frame
            # as part of the interior.
            # We want to do this for each color region that belongs to this block.
            # However, in this puzzle, each rectangle is uniform color, so we can just do:
            if (r1 - r0) >= 3 and (c1 - c0) >= 3:
                for rr in range(r0 + 1, r1 - 1):
                    for cc in range(c0 + 1, c1 - 1):
                        # Recolor if it's not background and belongs to the same block
                        # (We assume uniform color blocks, so check out_grid[rr, cc] != 0)
                        if out_grid[rr, cc] != 0:
                            out_grid[rr, cc] = object_color4

        return out_grid

    def create_grids(self):
        """
        Creates 3–4 training examples and 2 test examples. 
        All share the same grid size and color choices but differ in block placements.
        """
        # 1) Decide random grid size
        rows = random.randint(8, 15)
        cols = random.randint(8, 15)

        # 2) Choose color variables
        # we want object_color1,2,3,4 distinct (except for the repeated color case for a 4-block example).
        # We'll just ensure 4 distinct for the sake of the variables.
        distinct_colors = random.sample(range(1, 10), 4)

        # Fill the dictionary of variables used in the template
        taskvars = {
            'rows': rows,
            'cols': cols,
            'object_color1': distinct_colors[0],
            'object_color2': distinct_colors[1],
            'object_color3': distinct_colors[2],
            'object_color4': distinct_colors[3]
        }

        # 3) Make train and test data
        # We produce 3 or 4 train examples randomly, and exactly 2 test examples
        nr_train = random.choice([3, 4])
        nr_test = 2

        def generate_examples(n):
            result = []
            for _ in range(n):
                inp = self.create_input(taskvars, {})
                outp = self.transform_input(inp, taskvars)
                result.append(GridPair(input=inp, output=outp))
            return result

        train_pairs = generate_examples(nr_train)
        test_pairs  = generate_examples(nr_test)

        return taskvars, TrainTestData(train=train_pairs, test=test_pairs)


