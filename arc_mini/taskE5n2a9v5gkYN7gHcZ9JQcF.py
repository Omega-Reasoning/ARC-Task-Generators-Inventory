from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple
# Optional but encouraged: we can import from the provided libraries
from input_library import Contiguity, create_object, retry  # etc. if needed
from transformation_library import find_connected_objects  # etc. if needed

class TaskE5n2a9v5gkYN7gHcZ9JQcFGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Input Reasoning Chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each input grid has a completely filled main diagonal (top-left to bottom-right) with same colored (1-9) cells, while all other cells are empty (0).",
            "The color of the main diagonal cells changes across examples."
        ]
        
        # 2) Transformation Reasoning Chain
        transformation_reasoning_chain = [
            "The output grids are of size {vars['grid_size']}x{2*vars['grid_size']}.",
            "The output grid is created by duplicating the input grid, resulting in the diagonal pattern appearing twice, once starting from (0,0) and again from (0,{vars['grid_size']})."
        ]

        # 3) Call super().__init__
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars,
                     gridvars) -> np.ndarray:
        """
        Create an input grid according to:
        - A square grid of size grid_size.
        - Fill the main diagonal with a unique color.
        - All other cells are 0 (empty).
        """
        grid_size = taskvars['grid_size']
        color_for_diagonal = gridvars['color_for_diagonal']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)

        # Fill main diagonal with the specified color
        np.fill_diagonal(grid, color_for_diagonal)

        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid by duplicating it side by side:
        - The output grid size is (grid_size, 2 * grid_size).
        - The input pattern appears at columns [0..grid_size-1] and again
          at columns [grid_size..2*grid_size-1].
        """
        grid_size = taskvars['grid_size']
        
        # Create output grid
        output_grid = np.zeros((grid_size, 2 * grid_size), dtype=int)
        
        # Copy the input grid to the left half
        output_grid[:, :grid_size] = grid
        
        # Copy the same input grid to the right half
        output_grid[:, grid_size:] = grid

        return output_grid

    def create_grids(self):
        """
        Randomly:
        - Choose a grid_size in [6..30].
        - Choose how many train examples we want (3..6).
        - For each example (train + test), pick a distinct diagonal color in [1..9].
        - Return (taskvars, train_test_data).
        """
        # 1) Decide on a random grid_size
        grid_size = random.randint(6, 30)

        # 2) Decide the number of training examples
        nr_train_examples = random.randint(3, 6)

        # We'll create exactly 1 test example per the instructions
        nr_test_examples = 1

        # We need distinct colors for each example
        # total distinct = nr_train_examples + nr_test_examples
        total_needed_colors = nr_train_examples + nr_test_examples
        if total_needed_colors > 9:
            # Should not happen given we max at 7, but let's be safe
            total_needed_colors = 9
        
        # Sample distinct colors from 1..9
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        chosen_colors = available_colors[:total_needed_colors]

        # Setup task variables
        taskvars = {
            'grid_size': grid_size
        }

        # Build the train/test data
        train_pairs = []
        for i in range(nr_train_examples):
            gridvars = {'color_for_diagonal': chosen_colors[i]}
            inp = self.create_input(taskvars, gridvars)
            outp = self.transform_input(inp, taskvars)
            train_pairs.append(GridPair(input=inp, output=outp))

        # Test example
        test_pairs = []
        gridvars_test = {'color_for_diagonal': chosen_colors[-1]}
        inp_test = self.create_input(taskvars, gridvars_test)
        outp_test = self.transform_input(inp_test, taskvars)
        test_pairs.append(GridPair(input=inp_test, output=outp_test))

        # Package into the final data structure
        train_test_data: TrainTestData = {
            'train': train_pairs,
            'test': test_pairs
        }

        return taskvars, train_test_data



