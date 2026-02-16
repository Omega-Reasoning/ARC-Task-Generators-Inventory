# striping_task_generator.py

from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import create_object, random_cell_coloring, Contiguity
from Framework.transformation_library import find_connected_objects, GridObject
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class TaskMJduwptw9f9usLotC9GnNhGenerator(ARCTaskGenerator):
    def __init__(self):
        """
        Initializes the StripingTaskGenerator with the input and transformation reasoning chains.
        """
        # Input Reasoning Chain
        observation_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains either a {color('object_color1')} or a {color('object_color2')} rectangular object, 4-way connected cells of the same color.",
            "The remaining cells are empty (0)."
        ]
        
        # Transformation Reasoning Chain
        reasoning_chain = [
            "In the output grid the rectangular object is colored with a striped column pattern, alternating between {color('object_color1')} and {color('object_color2')}.",
            "If the object in the input grid is of {color('object_color1')} color, the striped column pattern starts with a {color('object_color1')} column as the left-most column of the output grid.",
            "If the object in the input grid is of {color('object_color2')} color, the pattern starts with a {color('object_color2')} column as the left-most column of the output grid.",
            "All empty (0) cells remain empty (0)."
        ]
        
        # Task Variable Definitions (if any)
        taskvars_definitions = {}
        
        # Initialize the superclass with the reasoning chains
        super().__init__(observation_chain, reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """
        Generates the task variables and the training/testing grid pairs.
        
        Returns:
            Tuple containing:
                - Dictionary of task variables.
                - Dictionary with 'train' and 'test' GridPair lists.
        """
        # Initialize task variables
        taskvars = self._initialize_taskvars()
        
        # Determine the number of training examples (3-6)
        nr_train = random.randint(3, 6)
        
        # Generate training examples
        train = []
        for _ in range(nr_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train.append({'input': input_grid, 'output': output_grid})
        
        # Generate one test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train, 'test': test}
    
    def _initialize_taskvars(self) -> Dict[str, Any]:
        """
        Initializes the task variables based on the general instructions.
        
        Returns:
            Dictionary of task variables.
        """
        # Choose two distinct colors between 1 and 9
        object_colors = random.sample(range(1, 10), 2)
        object_color1, object_color2 = object_colors
        
        # Choose grid size between 5 and 20 for both rows and columns
        grid_height = random.randint(5, 30)
        grid_width = random.randint(5, 30)
        
        return {
            'object_color1': object_color1,
            'object_color2': object_color2,
            'grid_height': grid_height,
            'grid_width': grid_width
        }
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """
        Creates an input grid based on the input reasoning chain.
        
        Args:
            taskvars: Task-level variables.
            gridvars: Grid-specific variables (unused in this generator).
        
        Returns:
            A numpy ndarray representing the input grid.
        """
        height = taskvars['grid_height']
        width = taskvars['grid_width']
        grid = np.zeros((height, width), dtype=int)
        
        # Randomly select the object's color
        object_color = random.choice([taskvars['object_color1'], taskvars['object_color2']])
        
        # Randomly determine the size of the rectangular object
        obj_height = random.randint(2, height // 2)
        obj_width = random.randint(2, width // 2)
        
        # Randomly position the object within the grid boundaries
        max_row = height - obj_height
        max_col = width - obj_width
        start_row = random.randint(0, max_row)
        start_col = random.randint(0, max_col)
        
        # Place the rectangular object
        grid[start_row:start_row + obj_height, start_col:start_col + obj_width] = object_color
        
        # Ensure there are some empty cells
        if np.all(grid != 0):
            num_empty = random.randint(1, max(1, (height * width) // 10))
            empty_indices = random.sample([(r, c) for r in range(height) for c in range(width)], num_empty)
            for r, c in empty_indices:
                grid[r, c] = 0
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """
        Transforms the input grid into the output grid based on the transformation reasoning chain.
        
        Args:
            grid: The input grid.
            taskvars: Task-level variables.
        
        Returns:
            A numpy ndarray representing the output grid.
        """
        # Find connected objects in the input grid
        objects = find_connected_objects(
            grid,
            diagonal_connectivity=False,
            background=0,
            monochromatic=True
        )
        
        # Ensure there's exactly one object
        if len(objects.objects) != 1:
            raise ValueError("Input grid must contain exactly one rectangular object.")
        
        obj = objects.objects[0]
        
        # Extract the bounding box of the object
        box = obj.bounding_box
        obj_grid = grid[box[0], box[1]].copy()
        
        # Determine the starting color for the stripes
        input_color = next(iter(obj.colors))
        if input_color == taskvars['object_color1']:
            start_color = taskvars['object_color1']
            alternate_color = taskvars['object_color2']
        else:
            start_color = taskvars['object_color2']
            alternate_color = taskvars['object_color1']
        
        # Apply striped column pattern
        height, width = obj_grid.shape
        for c in range(width):
            color = start_color if c % 2 == 0 else alternate_color
            obj_grid[:, c] = color
        
        # Create the output grid by placing the striped object
        output_grid = np.zeros_like(grid)
        output_grid[box[0], box[1]] = obj_grid
        
        return output_grid


