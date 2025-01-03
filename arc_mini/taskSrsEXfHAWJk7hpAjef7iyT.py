from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

from input_library import retry
from transformation_library import find_connected_objects  # Not strictly needed here, but example usage

class TasktaskSrsEXfHAWJk7hpAjef7iyTGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input reasoning chain
        observation_chain = [
            "Input grids are of size {vars['rows']} x {vars['columns']}.",
            "They contain multi-colored cells with values between 1 and 9."
        ]
        
        # 2) Transformation reasoning chain
        reasoning_chain = [
            "The output grid has dimensions {vars['rows']} x 1 and is is constructed by copying the first column of the input grid."
        ]

        # 3) Call super().__init__ with these chains
        super().__init__(observation_chain, reasoning_chain)

    def create_grids(self):
        """
        Create a dictionary of task variables and
        the corresponding train/test data grids.
        """
        # Randomly select the size of the input grids
        rows = random.randint(5, 30)
        columns = random.randint(5, 30)
        taskvars = {
            'rows': rows,
            'columns': columns,
        }

        # Randomly select how many training examples (3 to 6) and fix 1 test example
        nr_train = random.randint(3, 6)
        nr_test = 1

        # Use the provided helper to generate the default set of train/test grids
        # where each call to create_input() and transform_input() uses the same rows/columns
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)

        return taskvars, train_test_data

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid of size (rows x columns),
        filling it randomly with values in [1..9] while ensuring
        at least two distinct colors are present.
        """
        rows = taskvars['rows']
        cols = taskvars['columns']

        def generator():
            # Generate a random grid with values in [1..9]
            return np.random.randint(1, 10, size=(rows, cols))

        def is_multicolored(grid: np.ndarray):
            # Ensure at least two distinct values appear
            return len(np.unique(grid)) > 1

        # Repeatedly generate until at least two distinct colors exist
        input_grid = retry(generator, is_multicolored, max_attempts=100)
        return input_grid

    def transform_input(self, grid: np.ndarray, taskvars) -> np.ndarray:
        """
        Transform the input grid by copying its first column
        to produce an output grid of shape (rows x 1).
        """
        rows = taskvars['rows']
        # We ignore columns here because the output is forced to 1 column

        # Copy the first column
        first_column = grid[:, 0].reshape(rows, 1)

        return first_column

