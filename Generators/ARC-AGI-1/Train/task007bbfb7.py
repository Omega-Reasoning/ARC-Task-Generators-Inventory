import numpy as np
from typing import Dict, List, Any, Tuple
from Framework.arc_task_generator import ARCTaskGenerator, TrainTestData

class Task007bbfb7Generator(ARCTaskGenerator):

    def __init__(self):
       
        input_reasoning_chain = [
            "Input grids are of size {vars['row_blocks']}x{vars['column_blocks']}.",
            "Each grid contains several single-colored (1-9) and empty cells (0), randomly distributed across the grid.",
            "The color of the single-colored cells varies per example."
        ]
        transformation_reasoning_chain = [
            "The output grid is of size {vars['row_blocks']*vars['row_blocks']}x{vars['column_blocks']*vars['column_blocks']}.",
            "It can be subdivided into {vars['row_blocks']}x{vars['column_blocks']} blocks.",
            "Each block corresponds to a cell in the input grid.",
            "If the corresponding input cell is non-empty (1-9), the entire block is filled with that color.",
            "If the corresponding input cell is empty (0), the entire block remains empty (0)."
        ]
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        taskvars = {}
        taskvars['row_blocks'] = np.random.randint(2, 6)
        taskvars['column_blocks'] = np.random.randint(2, 6)

        # Number of training examples
        num_train = np.random.randint(3, 6)

        # Choose distinct colors for all train + test examples
        colors = np.random.choice(range(1, 10), size=num_train + 1, replace=False)

        train = []

        # Create training examples (each with a unique color)
        for i in range(num_train):
            color = colors[i]
            input_matrix = self.create_input(taskvars, {'color': color})
            output_matrix = self.transform_input(input_matrix, taskvars)
            train.append({
                "input": input_matrix,
                "output": output_matrix
            })

        # Create test example with a different color
        test_color = colors[-1]
        test_input = self.create_input(taskvars, {'color': test_color})
        test_output = self.transform_input(test_input, taskvars)
        test = [{
            "input": test_input,
            "output": test_output
        }]

        return (taskvars, {"train": train, "test": test})


    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['row_blocks']
        columns = taskvars['column_blocks']
        color = gridvars['color']

        # Create empty matrix
        matrix = np.zeros((rows, columns), dtype=int)

        total_cells = rows * columns

        # Ensure at least 2 colored cells and at least 1 empty cell
        min_filled = 2
        max_filled = total_cells - 1

        # Safety check (in case of very small grids, though ARC usually avoids 1x1)
        if max_filled < min_filled:
            min_filled = max_filled

        num_filled = np.random.randint(min_filled, max_filled + 1)

        # Generate random unique positions
        positions = np.random.choice(total_cells, size=num_filled, replace=False)

        # Convert positions to 2D indices and fill matrix
        row_indices = positions // columns
        col_indices = positions % columns
        matrix[row_indices, col_indices] = color

        return matrix


    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        row_blocks = taskvars['row_blocks']
        column_blocks = taskvars['column_blocks']

        # Create output matrix of appropriate size
        output = np.zeros((row_blocks * row_blocks, column_blocks * column_blocks), dtype=int)

        # Iterate through each position in the input matrix
        for i in range(row_blocks):
            for j in range(column_blocks):
                if grid[i, j] != 0:
                    # Calculate the starting position of the corresponding block
                    block_start_row = row_blocks * i
                    block_start_col = column_blocks * j

                    # Copy the input matrix into the corresponding block
                    output[block_start_row:block_start_row+row_blocks,
                           block_start_col:block_start_col+column_blocks] = grid

        return output
