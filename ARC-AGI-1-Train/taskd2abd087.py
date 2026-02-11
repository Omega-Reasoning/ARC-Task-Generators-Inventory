from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry, create_object, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskd2abd087Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each input grid contains {vars['m']} objects of size {vars['k']} and a random number of objects each with a size in the range ({vars['k']}-4, {vars['k']}+4) — provided this range is valid.",
            "All objects are colored with {color('color_1')}.",
            "All objects are composed of 4-adjacent cells.",
            "All objects are fully isolated from each other, with no shared edges or corners."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all distinct objects and their respective sizes.",
            "All objects of size {vars['k']} are recolored with {color('color_2')}, while all objects of any other size are recolored with {color('color_3')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'n': random.randint(8, 20),  # Grid size
            'k': random.randint(3, 8),   # Target object size
            'm': random.randint(1, 3),   # Number of target-sized objects
            'color_1': random.randint(1, 9),  # Initial color
            'color_2': random.randint(1, 9),  # Color for target-sized objects
            'color_3': random.randint(1, 9),  # Color for other objects
        }
        
        # Ensure colors are different
        while taskvars['color_2'] == taskvars['color_1']:
            taskvars['color_2'] = random.randint(1, 9)
        while taskvars['color_3'] == taskvars['color_1'] or taskvars['color_3'] == taskvars['color_2']:
            taskvars['color_3'] = random.randint(1, 9)

        # Create train and test examples
        num_train = random.randint(3, 6)
        train_examples = []
        test_examples = []
        
        for i in range(num_train + 1):
            # For each example, vary the number of non-target objects
            gridvars = {
                'num_other_objects': random.randint(1, 4)
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            example = {'input': input_grid, 'output': output_grid}
            
            if i < num_train:
                train_examples.append(example)
            else:
                test_examples.append(example)
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        k = taskvars['k']
        m = taskvars['m']
        color_1 = taskvars['color_1']
        num_other_objects = gridvars['num_other_objects']
        
        # Calculate valid size range for other objects
        min_other_size = max(1, k - 4)
        max_other_size = k + 4
        
        def attempt_create_grid():
            grid = np.zeros((n, n), dtype=int)
            placed_objects = []
            
            # Create target-sized objects
            for _ in range(m):
                obj = self._create_isolated_object(grid, k, color_1, placed_objects)
                if obj is None:
                    return None
                placed_objects.append(obj)
            
            # Create other-sized objects
            for _ in range(num_other_objects):
                # Choose a size different from k
                other_sizes = [s for s in range(min_other_size, max_other_size + 1) if s != k]
                if not other_sizes:
                    continue
                    
                other_size = random.choice(other_sizes)
                obj = self._create_isolated_object(grid, other_size, color_1, placed_objects)
                if obj is None:
                    continue
                placed_objects.append(obj)
            
            return grid
        
        return retry(attempt_create_grid, lambda g: g is not None, max_attempts=50)

    def _create_isolated_object(self, grid: np.ndarray, target_size: int, color: int, existing_objects: List) -> object:
        """Create a single isolated object of the specified size."""
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Try different object sizes around the target
            obj_height = random.randint(1, min(6, grid.shape[0] // 2))
            obj_width = random.randint(1, min(6, grid.shape[1] // 2))
            
            # Create object using input_library
            obj_grid = create_object(obj_height, obj_width, color, Contiguity.FOUR, background=0)
            
            # Check if object has the right size
            if np.sum(obj_grid != 0) != target_size:
                continue
            
            # Try to place it in the grid
            max_row = grid.shape[0] - obj_height
            max_col = grid.shape[1] - obj_width
            
            if max_row < 0 or max_col < 0:
                continue
                
            for _ in range(20):  # Try different positions
                start_row = random.randint(0, max_row)
                start_col = random.randint(0, max_col)
                
                # Check if this position would overlap or be adjacent to existing objects
                if self._can_place_object(grid, obj_grid, start_row, start_col):
                    # Place the object
                    for r in range(obj_height):
                        for c in range(obj_width):
                            if obj_grid[r, c] != 0:
                                grid[start_row + r, start_col + c] = obj_grid[r, c]
                    
                    # Return the coordinates of the placed object
                    coords = set()
                    for r in range(obj_height):
                        for c in range(obj_width):
                            if obj_grid[r, c] != 0:
                                coords.add((start_row + r, start_col + c))
                    return coords
        
        return None

    def _can_place_object(self, grid: np.ndarray, obj_grid: np.ndarray, start_row: int, start_col: int) -> bool:
        """Check if object can be placed at given position without overlapping or touching existing objects."""
        obj_height, obj_width = obj_grid.shape
        
        # Check extended area including 1-cell buffer for isolation
        for r in range(max(0, start_row - 1), min(grid.shape[0], start_row + obj_height + 1)):
            for c in range(max(0, start_col - 1), min(grid.shape[1], start_col + obj_width + 1)):
                if grid[r, c] != 0:
                    return False
        
        return True

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        k = taskvars['k']
        color_2 = taskvars['color_2']
        color_3 = taskvars['color_3']
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Recolor objects based on size
        for obj in objects:
            if len(obj) == k:
                # Target size - use color_2
                obj.cut(output_grid, background=0)
                for r, c, _ in obj.cells:
                    output_grid[r, c] = color_2
            else:
                # Other size - use color_3
                obj.cut(output_grid, background=0)
                for r, c, _ in obj.cells:
                    output_grid[r, c] = color_3
        
        return output_grid

