from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from input_library import create_object, retry
from transformation_library import find_connected_objects

class taskRjaY9uXLUcpG3oCNroeKb9Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grids have different sizes.",
            "Each input grid contains exactly one rectangular object of a single color (1-9), with a size of at least 3×3; the remaining cells are empty (0).",
            "The color of the rectangular object changes across examples.",
            "The sizes of all rectangular objects vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and creating a checkerboard pattern within the colored rectangular object.",
            "The checkerboard pattern is created by alternating two different colors across rows and columns: the original rectangle color and {color('fix_colour')} color.",
            "The checkerboard pattern always starts with {color('fix_colour')} color.",
            "All empty (0) cells remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
        # Track rectangle colors used within a single task generation
        self.used_rect_colors: set[int] = set()
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid with a rectangular object."""
        grid_height = gridvars['grid_height']
        grid_width = gridvars['grid_width']
        rect_color = gridvars['rect_color']
        rect_height = gridvars['rect_height']
        rect_width = gridvars['rect_width']
        
        # Create empty grid
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Calculate position to center the rectangle
        start_row = (grid_height - rect_height) // 2
        start_col = (grid_width - rect_width) // 2
        
        # Place the rectangle
        grid[start_row:start_row + rect_height, start_col:start_col + rect_width] = rect_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by creating checkerboard pattern in rectangular object."""
        output_grid = grid.copy()
        fix_colour = taskvars['fix_colour']
        
        # Find the rectangular object
        objects = find_connected_objects(output_grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        if len(objects) > 0:
            # Get the rectangular object (should be only one)
            rect_object = objects[0]
            original_color = list(rect_object.colors)[0]  # Get the original color
            
            # Get bounding box of the rectangle
            bounding_box = rect_object.bounding_box
            rect_rows = range(bounding_box[0].start, bounding_box[0].stop)
            rect_cols = range(bounding_box[1].start, bounding_box[1].stop)
            
            # Create checkerboard pattern starting with fix_colour
            for i, row in enumerate(rect_rows):
                for j, col in enumerate(rect_cols):
                    if (i + j) % 2 == 0:
                        output_grid[row, col] = fix_colour
                    else:
                        output_grid[row, col] = original_color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and generate train/test grids."""
        
        # Task variables (fixed across all examples)
        fix_colour = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        taskvars = {'fix_colour': fix_colour}
        
        # Reset used rectangle colors for this task generation
        self.used_rect_colors.clear()
        
        # Generate train examples
        num_train = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train):
            gridvars = self._generate_gridvars(taskvars, self.used_rect_colors)
            # Reserve the chosen color to keep it unique across examples
            self.used_rect_colors.add(gridvars['rect_color'])
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        gridvars = self._generate_gridvars(taskvars, self.used_rect_colors)
        self.used_rect_colors.add(gridvars['rect_color'])
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def _generate_gridvars(self, taskvars: Dict[str, Any], used_rect_colors: set[int]) -> Dict[str, Any]:
        """Generate grid-specific variables ensuring constraints are met and colors are unique."""
        fix_colour = taskvars['fix_colour']
        
        def generate_valid_setup():
            # Grid size between 5 and 30
            grid_height = random.randint(5, 30)
            grid_width = random.randint(5, 30)
            
            # Rectangle must be at least 3x3 and occupy >2/3 of grid
            min_rect_area = max(9, int((grid_height * grid_width * 2) / 3) + 1)
            
            # Colors: 1..9 excluding fix_colour and already used rectangle colors
            palette = [c for c in range(1, 10) if c != fix_colour and c not in used_rect_colors]
            if not palette:
                # No unique color left → fail this attempt so retry() tries again (or you may raise)
                return None
            
            # Try different rectangle sizes
            max_attempts = 50
            for _ in range(max_attempts):
                rect_height = random.randint(3, grid_height)
                rect_width = random.randint(3, grid_width)
                rect_area = rect_height * rect_width
                
                if rect_area >= min_rect_area:
                    rect_color = random.choice(palette)
                    return {
                        'grid_height': grid_height,
                        'grid_width': grid_width,
                        'rect_height': rect_height,
                        'rect_width': rect_width,
                        'rect_color': rect_color
                    }
            return None
        
        # Use retry to ensure we get valid parameters
        return retry(generate_valid_setup, lambda x: x is not None, max_attempts=100)
