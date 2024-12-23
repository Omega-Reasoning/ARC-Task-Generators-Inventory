from arc_task_generator import ARCTaskGenerator, MatrixPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List

from transformation_library import detect_translational_symmetry, orbit

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

    def create_matrices(self) -> Tuple[Dict[str, Any], TrainTestData]:
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

        train_test_data = self.create_matrices_default(nr_train_examples=random.randint(3, 6),
                                                       nr_test_examples=1,
                                                       taskvars=taskvars)
        return taskvars, train_test_data

    def create_input(self,
                     taskvars: Dict[str, Any],
                     matrixvars: Dict[str, Any]) -> np.ndarray:
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
                        matrix: np.ndarray,
                        taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input matrix to the output matrix by:
          1) Detecting translational symmetry (vertical repetition).
          2) Creating a new grid with {vars['output_rows']} rows and the same columns.
          3) Replicating the input pattern according to the discovered symmetry.
          4) Changing color from {vars['input_color']} to {vars['output_color']}.
        """
        input_color = taskvars["input_color"]
        output_color = taskvars["output_color"]
        output_rows = taskvars["output_rows"]

        # The new output has 'output_rows' rows, same # of columns as input
        rows_in, cols_in = matrix.shape
        output_grid = np.full((output_rows, cols_in), 0, dtype=int)

        # Detect translational symmetry
        symmetries = detect_translational_symmetry(matrix,
                                                   ignore_colors=[],
                                                   background=0)
        
        # if no symmetry was found, then our task generation is incorrect 
        assert len(symmetries) > 0, "No translational symmetry found in the input matrix."

        # Apply the discovered translational symmetry to replicate the pattern
        for (x, y) in np.argwhere(matrix != 0):
            # The orbit() function finds all symmetrical positions in the new shape
            # but we have to ensure they're in-bounds (which orbit does by ignoring out-of-bounds).
            for x2, y2 in orbit(output_grid, x, y, symmetries):
                # If it's within the new shape, fill it with the original color
                if 0 <= x2 < output_rows and 0 <= y2 < cols_in:
                    output_grid[x2, y2] = matrix[x, y]

        # Color change: input_color -> output_color
        output_grid[output_grid == input_color] = output_color
        return output_grid
