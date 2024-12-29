from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class ARCTask017c7c7bGenerator(ARCTaskGenerator):
    def __init__(self):
        observation_chain = [
            "The input matrices have {vars['input_rows']} rows and {vars['columns']} columns.",
            "All non-empty cells are colored {color('input_color')}.",
            "The arrangement of these colored cells repeats in the vertical direction, i.e. the n-th row and n+m-th row is equal."
        ]
        transformation_chain = [
            "The grid is extended to {vars['output_rows']} rows, keeping the same {vars['columns']} columns.",
            "First, the vertical repetition patterns needs to be detected in the input matrix and then continued to fill all rows in the output matrix.",
            "Finally, the color is changed from {color('input_color')} to {color('output_color')}."
        ]
        super().__init__(observation_chain, transformation_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Initialise the task variables and create the train and test matrices.
        We randomise the input rows, columns, output rows, as well as the input and output colors.
        """
        # Randomly pick the input dimensions
        input_rows = random.randint(6, 9)
        columns = random.randint(3, 6)
        # Ensure output rows > input_rows to highlight the extension
        output_rows = random.randint(input_rows + 3, input_rows + 5)

        # Choose distinct colors for input and output
        input_color = random.choice(range(1,9))
        remaining_colors = [c for c in range(1,9) if c != input_color]
        output_color = random.choice(remaining_colors)

        # Put these into the dictionary of task variables
        taskvars = {
            "input_rows": input_rows,
            "columns": columns,
            "output_rows": output_rows,
            "input_color": input_color,
            "output_color": output_color,
        }

        train_test_data = self.create_grids_default(nr_train_examples=random.randint(3, 6),
                                                       nr_test_examples=1,
                                                       taskvars=taskvars)
        return taskvars, train_test_data

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input matrix with shape (input_rows, columns), repeated along
        the vertical dimension, and filled with {vars['input_color']} as the
        non-empty color. We ensure at least two non-empty and at least two empty cells.
        """
        input_rows = taskvars["input_rows"]
        columns = taskvars["columns"]
        color = taskvars["input_color"]

        # Create an empty grid
        grid = np.zeros((input_rows, columns), dtype=int)

        # Generate a small 'sprite' that will be tiled vertically:
        sprite_height = random.randint(2, 4)
        sprite = np.zeros((sprite_height, columns), dtype=int)

        # Fill random positions in the sprite with 'color' until we have at least 2 cells filled
        # while ensuring at least 2 empty cells remain.
        total_cells = sprite_height * columns
        max_filled = total_cells - 2  
        fill_count = random.randint(2, max_filled)
        # Randomly pick positions to fill
        fill_positions = random.sample(range(total_cells), fill_count)
        for pos in fill_positions:
            r = pos // columns
            c = pos % columns
            sprite[r, c] = color

        # Now tile the sprite vertically until reaching at least the number of input rows
        repeated_times = taskvars['input_rows'] // sprite_height + 1
        big_pattern = np.tile(sprite, (repeated_times, 1))  # shape => (sprite_height*repeated_times, columns)

        # Crop to the desired input_rows
        cropped = big_pattern[:input_rows, :]

        # Place it into grid
        grid[:, :] = cropped
        return grid

    def transform_input(self,
                    grid: np.ndarray,
                    taskvars: Dict[str, Any]) -> np.ndarray:
        input_color = taskvars["input_color"]
        output_color = taskvars["output_color"]
        output_rows = taskvars["output_rows"]
        rows_in, cols_in = grid.shape
        
        # Find the repetition period by comparing rows
        period = None
        for m in range(1, rows_in):
            is_period = True
            for i in range(rows_in - m):
                if i + m >= rows_in:
                    break
                if not np.array_equal(grid[i], grid[i + m]):
                    is_period = False
                    break
            if is_period:
                period = m
                break
                
        if period is None:
            period = rows_in
        
        # Create output grid
        output_grid = np.zeros((output_rows, cols_in), dtype=int)
        
        # Fill the output grid by repeating the pattern
        pattern = grid[:period]  # Take the first period of rows as the pattern
        for i in range(0, output_rows, period):
            # Calculate how many rows we can copy (might be partial at the end)
            rows_to_copy = min(period, output_rows - i)
            output_grid[i:i + rows_to_copy] = pattern[:rows_to_copy]
        
        # Change color
        output_grid[output_grid == input_color] = output_color
        
        return output_grid
