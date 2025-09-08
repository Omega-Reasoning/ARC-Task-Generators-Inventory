from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry, enforce_object_width, enforce_object_height, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task88a62173Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid has a completely empty (0) middle row and middle column.",
            "The remaining 4 subgrids, each of size {(vars['grid_size']-1)//2} × {(vars['grid_size']-1)//2}, are located in the 4 corners of the grid.",
            "Each subgrid contains one single-colored object.",
            "Within a single input grid, all four objects share the same color; this color may vary across different examples.",
            "Each colored object is sized so that it fits within its {(vars['grid_size']-1)//2} × {(vars['grid_size']-1)//2} subgrid, such that every row and column of the subgrid contains at least one filled cell.",
            "Three out of the four objects have the exact same shape, and one is shaped differently."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {(vars['grid_size']-1)//2} × {(vars['grid_size']-1)//2}.",
            "They are constructed by identifying the four colored objects located in the {(vars['grid_size']-1)//2} × {(vars['grid_size']-1)//2} subgrids at each corner of the input grid.",
            "Three out of the four objects have the exact same shape and size, while one object has a different shape.",
            "The output grid is constructed by copying the {(vars['grid_size']-1)//2} × {(vars['grid_size']-1)//2} subgrid that contains the differently shaped object and pasting it as the output."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        grid_size = random.choice([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])  # odd sizes
        
        taskvars = {
            'grid_size': grid_size
        }
        
        # Generate training examples (3-5)
        num_train = random.randint(3, 5)
        
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        subgrid_size = (grid_size - 1) // 2
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Generate the template shape (used for 3 objects)
        template_shape = self._generate_template_shape(subgrid_size)
        
        # Generate different shape
        different_shape = self._generate_different_shape(subgrid_size, template_shape)
        
        # Choose which corner gets the different shape
        different_corner = random.randint(0, 3)
        
        # Define corner positions
        corners = [
            (0, 0),  # top-left
            (0, subgrid_size + 1),  # top-right
            (subgrid_size + 1, 0),  # bottom-left
            (subgrid_size + 1, subgrid_size + 1)  # bottom-right
        ]
        
        # --- CHANGE: choose a single color for all 4 objects in this grid
        common_color = random.choice(list(range(1, 10)))
        
        # Place objects in corners
        for i, (start_row, start_col) in enumerate(corners):
            shape = different_shape if i == different_corner else template_shape
            colored_shape = np.where(shape > 0, common_color, 0)
            
            grid[start_row:start_row + subgrid_size, 
                 start_col:start_col + subgrid_size] = colored_shape
        
        return grid
    
    def _generate_template_shape(self, size: int) -> np.ndarray:
        """Generate a shape that fills every row and column."""
        def shape_generator():
            return create_object(size, size, [1], Contiguity.EIGHT, background=0)
        
        # Ensure every row and column has at least one cell
        shape = retry(
            lambda: enforce_object_height(lambda: enforce_object_width(shape_generator)),
            lambda x: np.all(np.any(x != 0, axis=1)) and np.all(np.any(x != 0, axis=0))
        )
        return shape
    
    def _generate_different_shape(self, size: int, template_shape: np.ndarray) -> np.ndarray:
        """Generate a shape different from the template."""
        def different_generator():
            shape = self._generate_template_shape(size)
            # Ensure it's actually different from template
            return shape if not np.array_equal(shape > 0, template_shape > 0) else None
        
        return retry(
            different_generator,
            lambda x: x is not None and not np.array_equal(x > 0, template_shape > 0),
            max_attempts=50
        )
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        subgrid_size = (grid_size - 1) // 2
        
        # Extract the 4 corner subgrids
        corners = [
            grid[0:subgrid_size, 0:subgrid_size],  # top-left
            grid[0:subgrid_size, subgrid_size + 1:grid_size],  # top-right
            grid[subgrid_size + 1:grid_size, 0:subgrid_size],  # bottom-left
            grid[subgrid_size + 1:grid_size, subgrid_size + 1:grid_size]  # bottom-right
        ]
        
        # Get the shape patterns (ignoring color)
        shapes = [(corner > 0).astype(int) for corner in corners]
        
        # Find which shape is different
        for i, shape_i in enumerate(shapes):
            is_different = True
            for j, shape_j in enumerate(shapes):
                if i != j and np.array_equal(shape_i, shape_j):
                    is_different = False
                    break
            
            if is_different:
                # Ensure the others are the same
                other_shapes = [shapes[k] for k in range(4) if k != i]
                if len(set(tuple(map(tuple, s)) for s in other_shapes)) == 1:
                    return corners[i].copy()
        
        # Fallback: if we can't determine, return the first corner
        return corners[0].copy()
