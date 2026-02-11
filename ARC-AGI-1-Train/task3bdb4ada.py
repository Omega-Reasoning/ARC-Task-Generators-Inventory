from arc_task_generator import ARCTaskGenerator, TrainTestData
import numpy as np
import random
from input_library import create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects

class Task3bdb4adaGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids have {vars['rows']} rows, but the number of columns varies from grid to grid.",
            "They contain several colored (1-9) rectangular blocks, each fully separated by empty (0) cells.",
            "Each rectangular block has a unique color, which varies across examples.",
            "The length of each rectangular block is fixed at 3, while the width is an odd number, always more than the length, and varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and removing specific cells from each colored block.",
            "Every second cell in the second row of each colored block, starting from the second cell, is removed."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Generate task variables
        # Choose rows but ensure there is room for at least two different block-count choices
        rows = random.randint(8, 30)

        # Compute how many blocks can fit for this height
        usable_rows = rows - 1
        max_blocks = usable_rows // 4
        if usable_rows % 4 == 3:
            max_blocks += 1

        # If there aren't at least two feasible block counts (2 and 3), force rows to 12
        # so we can create train vs test with different block counts. This keeps rows in the
        # original allowed range while ensuring we can satisfy the requirement.
        if max_blocks < 3:
            rows = 12
            usable_rows = rows - 1
            max_blocks = usable_rows // 4
            if usable_rows % 4 == 3:
                max_blocks += 1

        # We expose only rows as a task variable; columns will vary per example
        taskvars = {'rows': rows}

        # Define a reasonable per-example column range so columns vary from grid to grid
        min_cols = max(8, rows + 5)
        max_cols = rows * 2 + 2

        # Generate 3-5 training examples
        n_train = random.randint(3, 5)
        train_data = []

        # Determine feasible block counts (cap at 5 as requested)
        max_blocks_for_task = min(max_blocks, 5)
        feasible_counts = list(range(2, max_blocks_for_task + 1))

        # Pick a fixed number of blocks for all training examples
        train_num_blocks = random.choice(feasible_counts)

        for _ in range(n_train):
            cols_example = random.randint(min_cols, max_cols)
            gridvars = {'rows': rows, 'cols': cols_example, 'num_blocks': train_num_blocks}  # Ensure consistent size and block count
            input_grid = self.create_input(gridvars, {})
            output_grid = self.transform_input(input_grid, gridvars)

            train_data.append({'input': input_grid, 'output': output_grid})

        # Generate 1 test example with a different number of blocks than the training grids
        test_feasible = [c for c in feasible_counts if c != train_num_blocks]
        # If for some reason there's no alternative (shouldn't happen because we forced rows),
        # fall back to the nearest different valid count.
        if not test_feasible:
            # Attempt to pick a different count within 2..5
            alt = 2 if train_num_blocks != 2 else min(5, max_blocks_for_task)
            test_num_blocks = alt
        else:
            test_num_blocks = random.choice(test_feasible)

        test_cols = random.randint(min_cols, max_cols)
        test_gridvars = {'rows': rows, 'cols': test_cols, 'num_blocks': test_num_blocks}  # Match train grid sizes but different block count
        test_input = self.create_input(test_gridvars, {})
        test_output = self.transform_input(test_input, test_gridvars)

        test_data = [{'input': test_input, 'output': test_output}]

        return taskvars, {'train': train_data, 'test': test_data}

    def create_input(self, gridvars: dict[str, any], unused_vars: dict[str, any]) -> np.ndarray:
        # Use provided height and width
        height = gridvars['rows']
        width = gridvars['cols']

        # Create empty grid
        grid = np.zeros((height, width), dtype=int)

        # Calculate max blocks that can fit (each block is 3 rows + 1 empty row)
        usable_rows = height - 1  # Start placing from row index 1
        max_blocks = usable_rows // 4
        if usable_rows % 4 == 3:
            max_blocks += 1

        # Cap at 5 as per requirement; allow caller to request a specific number via gridvars['num_blocks']
        max_blocks = min(max_blocks, 5)

        if 'num_blocks' in gridvars:
            requested = int(gridvars['num_blocks'])
            # Ensure requested is within feasible bounds [2, max_blocks]
            num_blocks = max(2, min(requested, max_blocks))
        else:
            # Default behavior: choose at least 2 blocks but not more than max_blocks
            num_blocks = max(2, min(max_blocks, 5))

        # Choose different colors for blocks
        colors = random.sample(range(1, 10), num_blocks)

        # Start placing blocks from the second row
        current_row = 1
        blocks_placed = 0
        wide_block_placed = False

        # Calculate minimum width for a "wide" block (at least 70% of grid width)
        min_wide_width = int(width * 0.7)
        if min_wide_width % 2 == 0:
            min_wide_width += 1
        min_wide_width = min(min_wide_width, width - 2)

        while current_row + 2 < height and blocks_placed < num_blocks:
            if not wide_block_placed:
                # Create a wide block
                block_width = min_wide_width
                offset = random.randint(-2, 2)
                col_start = max(1, (width - block_width) // 2 + offset)
                col_start = min(col_start, width - block_width - 1)

                grid[current_row:current_row+3, col_start:col_start+block_width] = colors[blocks_placed]
                blocks_placed += 1
                wide_block_placed = True
            else:
                # Regular blocks
                min_block_width = 5  # 3 (length) + 2 ensures width > length
                max_block_width = min(width - 2, 15)

                possible_widths = [w for w in range(min_block_width, max_block_width + 1) if w % 2 == 1]
                block_width = random.choice(possible_widths) if possible_widths else min_block_width + 1

                col_start = random.randint(1, width - block_width - 1)
                grid[current_row:current_row+3, col_start:col_start+block_width] = colors[blocks_placed]
                blocks_placed += 1

            # Move to next row with spacing
            current_row += 4  

            # Special case for last block placement
            remaining_rows = height - current_row
            if blocks_placed == num_blocks - 1 and remaining_rows == 3:
                current_row = height - 3

        # Ensure we have at least 2 blocks
        if blocks_placed < 2 and height >= 8:
            if current_row + 2 >= height:
                current_row = 5  
            
            min_block_width = 5
            max_block_width = min(width - 2, 15)

            possible_widths = [w for w in range(min_block_width, max_block_width + 1) if w % 2 == 1]
            block_width = random.choice(possible_widths) if possible_widths else min_block_width + 1
            col_start = random.randint(1, width - block_width - 1)
            grid[current_row:current_row+3, col_start:col_start+block_width] = colors[1 % len(colors)]

        return grid
    
    def transform_input(self, grid: np.ndarray, gridvars: dict[str, any]) -> np.ndarray:
        # Create a copy of the input grid
        output_grid = grid.copy()

        # Find all connected colored blocks
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)

        for obj in objects:
            rows, cols = obj.bounding_box
            min_row, min_col = rows.start, cols.start

            height = rows.stop - rows.start
            width = cols.stop - cols.start

            if height == 3 and width > 3:
                for col in range(min_col + 1, cols.stop, 2):
                    output_grid[min_row + 1, col] = 0

        return output_grid
