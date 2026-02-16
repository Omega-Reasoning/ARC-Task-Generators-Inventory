from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import random, np
from typing import Dict, Any, Tuple, List


class TaskWbXtZUNwj39hLKgftdRuUSGenerator(ARCTaskGenerator):
    def __init__(self):
        # 1. Input reasoning chain (as is, from the prompt)
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "They only contain several {color('cell_color')} cells.",
            "All other cells are empty (0)."
        ]
        
        # 2. Transformation reasoning chain (as is, from the prompt)
        transformation_reasoning_chain = [
            "The output grid is created by initializing a zero-filled grid, then filling the middle row and middle column with cells of the same color as in the input grid."
        ]
        
        # 3. Call super().__init__ with a dictionary for taskvars definitions 
        #    (Here we only have one variable 'cell_color', so we just show an empty dict. 
        #     We will populate 'cell_color' at runtime in create_grids().)
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain.
        The grid is an odd-sized matrix with mostly zeros and a few cells
        colored with `taskvars['cell_color']`.
        
        We pick the size from gridvars and place the color randomly.
        """
        cell_color = taskvars['cell_color']
        rows = gridvars['rows']
        cols = gridvars['cols']
        
        grid = np.zeros((rows, cols), dtype=int)
        
        # Randomly place the colored cells. For example, fill up to ~20% of cells:
        num_cells_to_color = random.randint(1, max(1, rows * cols // 10))
        for _ in range(num_cells_to_color):
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            grid[r, c] = cell_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain:
        * Initialize a zero-filled grid of the same shape.
        * Then fill the middle row and middle column with the same color as in the input grid.
        """
        cell_color = taskvars['cell_color']
        rows, cols = grid.shape
        
        # Create a zero-filled output grid
        output_grid = np.zeros((rows, cols), dtype=int)
        
        mid_row = rows // 2
        mid_col = cols // 2
        
        # Fill middle row and column with cell_color
        output_grid[mid_row, :] = cell_color
        output_grid[:, mid_col] = cell_color
        
        return output_grid

    def create_grids(self):
        """
        Randomly create 3-6 train grids and 1 test grid. 
        Each grid has a distinct odd size from 5 to 30 (inclusive).
        The color cell_color is chosen once for the entire task.
        
        Returns:
            Tuple of:
            - A dict of task variables including 'cell_color'.
            - The TrainTestData with lists of train and test grid pairs.
        """
        # 1) Choose color for this entire ARC task:
        cell_color = random.randint(1, 9)  # from 1..9
        
        # 2) Randomly determine how many training examples
        nr_train_examples = random.randint(3, 6)
        nr_test_examples = 1
        
        # 3) Pick unique odd sizes (between 5 and 30) for each example
        #    so that each input grid has a different size
        possible_sizes = [s for s in range(5, 31) if s % 2 == 1]  # 5,7,9,...,29
        chosen_sizes = random.sample(possible_sizes, nr_train_examples + nr_test_examples)
        
        # 4) Prepare the train examples
        train_pairs = []
        for i in range(nr_train_examples):
            size = chosen_sizes[i]
            gridvars = {'rows': size, 'cols': size}
            input_grid = self.create_input({'cell_color': cell_color}, gridvars)
            output_grid = self.transform_input(input_grid, {'cell_color': cell_color})
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # 5) Prepare the test example
        test_pairs = []
        test_size = chosen_sizes[-1]
        gridvars_test = {'rows': test_size, 'cols': test_size}
        test_input = self.create_input({'cell_color': cell_color}, gridvars_test)
        test_output = self.transform_input(test_input, {'cell_color': cell_color})
        test_pairs.append({
            'input': test_input,
            'output': test_output
        })
        
        # 6) Return the task variables and the dictionary containing the train/test data
        taskvars = {
            'cell_color': cell_color
        }
        
        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }
        
        return taskvars, train_test_data

