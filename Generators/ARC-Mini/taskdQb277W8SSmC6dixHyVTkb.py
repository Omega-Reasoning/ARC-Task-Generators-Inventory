from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# We do not actually need these libraries for the straightforward approach, but we show how you might import them:
# from Framework.input_library import create_object, retry, Contiguity
# from Framework.transformation_library import find_connected_objects, GridObject, GridObjects, BorderBehavior, CollisionBehavior

class TaskdQb277W8SSmC6dixHyVTkbGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each input grid has exactly two cells of the same color located either in the top-left and bottom-right corners or top-right and bottom-left corners of the grid.",
            "The remaining cells are empty (0)."
        ]
        
        # 2) Transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and modifying specific rows and columns based on the placement of filled cells.",
            "If the two colored cells are located in the top-left and bottom-right corners, the entire first column and last row are filled with the same color as the two cells. Otherwise, the entire first column and first row are filled with the same color."]
        # 3) Call superclass __init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars,
                     gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
         - Two same-colored corner cells.
         - Rest are empty.
         - The two corners are either (top-left, bottom-right) or (top-right, bottom-left).
        """
        rows = taskvars['rows']
        cols = taskvars['cols']
        color = gridvars['color']
        pattern = gridvars['pattern']  # 'LR' or 'RL'

        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)

        if pattern == 'LR':
            # Top-left & bottom-right corners
            grid[0, 0] = color
            grid[rows - 1, cols - 1] = color
        else:
            # Top-right & bottom-left corners
            grid[0, cols - 1] = color
            grid[rows - 1, 0] = color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain.
        - Detect which corners are filled.
        - If top-left & bottom-right => fill entire first column & last row.
        - If top-right & bottom-left => fill entire first column & first row.
        """
        # Make a copy to avoid in-place changes
        output_grid = grid.copy()
        rows, cols = output_grid.shape

        # Determine if the corners are (top-left, bottom-right) or (top-right, bottom-left)
        # Whichever corner is non-zero, that color is used
        # We assume exactly two corners have the same color (per instructions).
        if output_grid[0, 0] != 0 and output_grid[rows-1, cols-1] != 0:
            # top-left / bottom-right
            color = output_grid[0, 0]
            # Fill entire first column
            output_grid[:, 0] = color
            # Fill entire last row
            output_grid[rows - 1, :] = color
        else:
            # top-right / bottom-left
            # Could check output_grid[0, cols-1], for instance
            color = output_grid[0, cols - 1] if output_grid[0, cols - 1] != 0 else output_grid[rows - 1, 0]
            # Fill entire first column
            output_grid[:, 0] = color
            # Fill entire first row
            output_grid[0, :] = color
        
        return output_grid

    def create_grids(self):
        """
        Creates 4 training examples and 2 test examples total, ensuring:
          * Distinct corner colors for each example
          * At least one training example with corners in top-left/bottom-right
          * At least one training example with corners in top-right/bottom-left
          * One test with top-left/bottom-right, one with top-right/bottom-left
          * rows, cols in [5..30], chosen once for all examples
        """
        # 1) Pick random rows and cols for all examples in this task
        rows = random.randint(5, 30)
        cols = random.randint(5, 30)

        # 2) Prepare task variables dictionary
        taskvars = {
            'rows': rows,
            'cols': cols
        }

        # We want a total of 6 examples: 4 training and 2 testing
        # Distinct colors in [1..9] for each example
        all_colors = random.sample(range(1, 10), 6)

        # Force at least one training example to have LR corners, one training to have RL
        # Force test set to have one LR, one RL
        # We can fill the rest randomly or systematically.

        # Define patterns for train examples (4 total)
        # We ensure the first is LR, second is RL, then the third and fourth are random
        train_patterns = ['LR', 'RL']
        for _ in range(2):  # Make 2 more training patterns, randomly chosen
            train_patterns.append(random.choice(['LR', 'RL']))

        # Define patterns for test examples (2 total)
        test_patterns = ['LR', 'RL']  # as requested by constraints

        # 3) Build the training examples
        train_examples = []
        for i in range(4):
            gridvars = {
                'color': all_colors[i],
                'pattern': train_patterns[i]
            }
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)
            train_examples.append(GridPair(input=inp, output=out))

        # 4) Build the test examples
        test_examples = []
        for i in range(2):
            gridvars = {
                'color': all_colors[4 + i],
                'pattern': test_patterns[i]
            }
            inp = self.create_input(taskvars, gridvars)
            out = self.transform_input(inp, taskvars)
            test_examples.append(GridPair(input=inp, output=out))

        train_test_data = TrainTestData(train=train_examples, test=test_examples)
        return taskvars, train_test_data


