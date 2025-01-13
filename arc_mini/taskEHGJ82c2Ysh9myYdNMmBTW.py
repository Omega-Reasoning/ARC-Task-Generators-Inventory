# corner_diagonal_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry  # (Optional) you can import more if desired
from transformation_library import find_connected_objects  # (Optional) you can import more if desired
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskEHGJ82c2Ysh9myYdNMmBTWGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1) Initialize the input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "Each input grid contains four same-colored (1-9) cells located in the four corners of the grid.",
            "All other cells are empty (0)."
        ]
        
        # 2) Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input grid.",
            "Extend the corner cells diagonally using the same color to form the main diagonal (top-left to bottom-right) and the inverse diagonal (top-right to bottom-left).",
            "All other cells remain unchanged."
        ]
        
        # 3) Call the parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain:
        - Grid size is taskvars['grid_size'] x taskvars['grid_size']
        - Four same-colored corner cells (color = gridvars['corner_color'])
        - All other cells are 0.
        """
        size = taskvars['grid_size']
        color = gridvars['corner_color']
        grid = np.zeros((size, size), dtype=int)

        # Set the four corners to the same color
        grid[0, 0] = color
        grid[0, size - 1] = color
        grid[size - 1, 0] = color
        grid[size - 1, size - 1] = color
        
        return grid

    def transform_input(self,
                        grid: np.ndarray,
                        taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        1. Copy the input grid.
        2. Extend the corner color diagonally on both main and inverse diagonal.
        """
        output_grid = grid.copy()
        n = output_grid.shape[0]
        
        # The corner color is the same at the four corners, so just read the top-left corner
        corner_color = output_grid[0, 0]
        
        # Fill the main diagonal and the inverse diagonal
        for i in range(n):
            output_grid[i, i] = corner_color
            output_grid[i, n - 1 - i] = corner_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Creates the dictionary of task variables and the corresponding train/test data.
        
        - Randomly choose an integer grid_size between 7 and 30.
        - Randomly choose how many train examples (3-5).
        - For each train example and for the single test example, assign a unique corner color.
        - Generate input and output grids using create_input() and transform_input().
        """
        # 1. Pick grid_size
        grid_size = random.randint(7, 30)
        
        # 2. Pick how many training examples
        nr_train_examples = random.randint(3, 5)
        nr_test_examples = 1
        
        # 3. Pick distinct colors (one for each train example + 1 test example)
        #    We have 9 possible non-zero colors (1..9).
        needed_colors = nr_train_examples + nr_test_examples
        all_possible_colors = list(range(1, 10))
        random.shuffle(all_possible_colors)
        chosen_colors = all_possible_colors[:needed_colors]
        
        # 4. Prepare train/test data lists
        train_data = []
        test_data = []
        
        # 5. For each training example, build a grid with a distinct corner color
        for i in range(nr_train_examples):
            corner_color = chosen_colors[i]
            input_grid = self.create_input(
                taskvars={'grid_size': grid_size},
                gridvars={'corner_color': corner_color}
            )
            output_grid = self.transform_input(input_grid, {'grid_size': grid_size})
            train_data.append(GridPair(input=input_grid, output=output_grid))
        
        # 6. For the test example, use the final color
        test_color = chosen_colors[-1]
        test_input_grid = self.create_input(
            taskvars={'grid_size': grid_size},
            gridvars={'corner_color': test_color}
        )
        test_output_grid = self.transform_input(test_input_grid, {'grid_size': grid_size})
        test_data.append(GridPair(input=test_input_grid, output=test_output_grid))
        
        # 7. The dictionary of task variables we place in the final ARC task
        #    We'll store just the grid_size as a top-level variable.
        #    The corner colors differ per example, so we didn't store them all in 'taskvars'.
        #    That's fine, because the input reasoning chain is phrased with {vars['grid_size']}.
        taskvars = {
            'grid_size': grid_size
        }
        
        train_test_data = TrainTestData(train=train_data, test=test_data)
        return taskvars, train_test_data


