# frame_fill_task_generator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, GridObjects, GridObject
import numpy as np
from typing import Dict, List, Any, Tuple
import random

class TaskVYET79k3tEpBZ96z5t6gxnGenerator(ARCTaskGenerator):
    def __init__(self):
        # Initialize the input reasoning chain
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid has only one 4-way connected object, which is a {color('object_color')} rectangular-shaped border, with empty (0) interior cells."
        ]
        
        # Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "To construct the output grid, copy the input matrix and color all empty (0) cells inside the frame with {color('fill_color')} color."
        ]
        
        # Define task variables definitions if needed (empty in this case)
        taskvars_definitions = {}
        
        # Initialize the superclass
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Create an input grid according to the input reasoning chain.
        """
        # Extract object color from task variables
        object_color = taskvars['object_color']
        
        # Randomly select grid size between 5 and 30
        grid_size = random.randint(5, 30)
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Define frame thickness (1 cell thick)
        thickness = 1
        
        # Create a square frame
        for r in range(grid_size):
            for c in range(grid_size):
                if (r < thickness or r >= grid_size - thickness or
                    c < thickness or c >= grid_size - thickness):
                    grid[r, c] = object_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transform the input grid according to the transformation reasoning chain.
        """
        # Extract fill color from task variables
        fill_color = taskvars['fill_color']
        
        # Copy the input grid to create the output grid
        output_grid = grid.copy()
        
        # Find all connected objects in the grid
        objects = find_connected_objects(output_grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Assume there's only one object (the frame)
        if len(objects) != 1:
            raise ValueError("Input grid should contain exactly one 4-way connected frame object.")
        
        frame = objects.objects[0]
        
        # Get the bounding box of the frame
        row_slice, col_slice = frame.bounding_box
        inner_rows = range(row_slice.start + 1, row_slice.stop - 1)
        inner_cols = range(col_slice.start + 1, col_slice.stop - 1)
        
        # Fill the interior with fill_color
        for r in inner_rows:
            for c in inner_cols:
                if output_grid[r, c] == 0:
                    output_grid[r, c] = fill_color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Initialize task variables and create train/test data grids.
        """
        # Define possible colors excluding 0 (empty)
        possible_colors = list(range(1, 10))
        
        # Randomly select object_color and fill_color ensuring they are different
        object_color, fill_color = random.sample(possible_colors, 2)
        
        # Define task variables
        taskvars = {
            'object_color': object_color,
            'fill_color': fill_color
        }
        
        # Create train and test data using the default grid creation method
        nr_train = random.randint(3, 6)
        nr_test = 1
        train_test_data = self.create_grids_default(nr_train, nr_test, taskvars)
        
        return taskvars, train_test_data
