# checkerboard_task_generator.py

import numpy as np
import random
from typing import Dict, List, Any, Tuple
from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, retry, random_cell_coloring
from Framework.transformation_library import find_connected_objects

class TaskSCzQu58iCqZNi46VnxRUFeGenerator(ARCTaskGenerator):
    def __init__(self):
        # Initialize the input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains only one {color('object_color')} rectangular object, 4-way connected cells of the same color, and a single {color('cell_color')} cell.",
            "The remaining cells are empty (0)."
        ]
        
        # Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "In the output grid, only the {color('object_color')} rectangular object is replaced with a checkerboard pattern that alternates between {color('checker_color1')} and {color('checker_color2')} cells in each row and column.",
            "The pattern always starts with {color('checker_color1')} in the top-left cell of the pattern.",
            "The {color('cell_color')} cell remains unchanged."
        ]
        
        # Define task variables definitions if needed (optional)
        taskvars_definitions = {
            'object_color': 'int',
            'cell_color': 'int',
            'checker_color1': 'int',
            'checker_color2': 'int'
        }
        
        # Call the superclass initializer
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self,
                    taskvars: Dict[str, Any],
                    gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain given the task and grid variables.
        """
        # Extract task variables
        object_color = taskvars['object_color']
        cell_color = taskvars['cell_color']
        
        # Extract grid variables or set defaults
        rows = gridvars.get('rows', random.randint(5, 30))
        cols = gridvars.get('cols', random.randint(5, 30))
        
        # Initialize empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Define rectangle size (minimum size 2x2 to ensure it's a rectangle)
        rect_height = random.randint(2, min(6, rows // 2))
        rect_width = random.randint(2, min(6, cols // 2))
        
        # Define top-left corner of the rectangle
        max_row_start = rows - rect_height
        max_col_start = cols - rect_width
        rect_row_start = random.randint(0, max_row_start)
        rect_col_start = random.randint(0, max_col_start)
        
        # Place the rectangular object
        grid[rect_row_start:rect_row_start + rect_height, rect_col_start:rect_col_start + rect_width] = object_color
        
        # Place a single cell with cell_color, ensuring it does not overlap the object
        empty_cells = np.argwhere(grid == 0)
        if empty_cells.size == 0:
            raise ValueError("No empty cells available to place the single cell.")
        single_cell_idx = tuple(empty_cells[random.randint(0, len(empty_cells) - 1)])
        grid[single_cell_idx] = cell_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars=None) -> np.ndarray:

        output_grid = grid.copy()

        object_color = taskvars['object_color']
        checker_color1 = taskvars['checker_color1']
        checker_color2 = taskvars['checker_color2']

        rows, cols = grid.shape

        # Find bounding box of the rectangle
        top = rows
        left = cols
        bottom = -1
        right = -1

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == object_color:
                    if r < top:
                        top = r
                    if r > bottom:
                        bottom = r
                    if c < left:
                        left = c
                    if c > right:
                        right = c

        # Dimensions of the rectangle
        obj_rows = bottom - top + 1
        obj_cols = right - left + 1

        # Create checkerboard pattern
        for r in range(obj_rows):
            for c in range(obj_cols):

                if (r + c) % 2 == 0:
                    output_grid[top + r, left + c] = checker_color1
                else:
                    output_grid[top + r, left + c] = checker_color2

        return output_grid
    
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Initialize task variables and create train and test data grids.
        """
        # Define task variables with constraints
        taskvars = {}
        taskvars['object_color'] = random.randint(1, 9)
        
        # Ensure cell_color is different from object_color
        taskvars['cell_color'] = retry(
            lambda: random.randint(1, 9),
            lambda x: x != taskvars['object_color']
        )
        
        # Ensure checker_color1 and checker_color2 are different
        taskvars['checker_color1'] = random.randint(1, 9)
        taskvars['checker_color2'] = retry(
            lambda: random.randint(1, 9),
            lambda x: x != taskvars['checker_color1']
        )
        
        # Number of training examples
        nr_train = random.randint(3, 6)
        nr_test = 1
        
        # Generate training grids
        train = []
        for _ in range(nr_train):
            gridvars = {
                'rows': random.randint(5, 20),
                'cols': random.randint(5, 20)
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train.append({'input': input_grid, 'output': output_grid})
        
        # Generate test grid
        gridvars = {
            'rows': random.randint(5, 20),
            'cols': random.randint(5, 20)
        }
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test = [{'input': test_input, 'output': test_output}]
        
        # Compile train and test data
        train_test_data = {
            'train': train,
            'test': test
        }
        
        return taskvars, train_test_data



