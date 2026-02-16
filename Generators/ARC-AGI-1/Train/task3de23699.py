from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects
from Framework.input_library import create_object, Contiguity, random_cell_coloring

class Task3de23699Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain four same-colored cells positioned at the four corners of an imaginary rectangle or square block.",
            "The imaginary block is at least 4x4 in size and is positioned within the grid, ensuring at least a one-cell wide empty (0) border around it.",
            "One or more 8-way connected objects are placed inside the imaginary rectangle.",
            "Two different colors are used in each grid; first for the four corner cells and second for the 8-way connected object.",
            "The colors vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are always smaller than the input grids.",
            "They are constructed by first identifying the four same-colored corner cells and then copying their entire interior and pasting it in output."
        ]
        
        taskvars_definitions = {}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
        
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        rows = random.randint(8, 30)
        cols = random.randint(8, 30)
        
        taskvars = {'rows': rows, 'cols': cols}
        
        # Create 3-5 training examples and 1 test example
        num_train_examples = random.randint(3, 5)
        
        train_examples = []
        for _ in range(num_train_examples):
            # For each example, create input grid and transform to output
            gridvars = {
                'corner_color': random.randint(1, 9),
                'object_color': random.randint(1, 9),
                'num_objects': random.randint(1, 3)  # Generate 1-3 objects
            }
            # Make sure corner color and object color are different
            while gridvars['corner_color'] == gridvars['object_color']:
                gridvars['object_color'] = random.randint(1, 9)
                
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_gridvars = {
            'corner_color': random.randint(1, 9),
            'object_color': random.randint(1, 9),
            'num_objects': random.randint(1, 3)  # Generate 1-3 objects
        }
        while test_gridvars['corner_color'] == test_gridvars['object_color']:
            test_gridvars['object_color'] = random.randint(1, 9)
            
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        corner_color = gridvars['corner_color']
        object_color = gridvars['object_color']
        num_objects = gridvars.get('num_objects', random.randint(1, 3))  # Default to random if not specified
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Define rectangle parameters (ensuring at least 1 cell border)
        min_row = random.randint(1, rows - 6)  # Ensure at least 4x4 rectangle + border
        min_col = random.randint(1, cols - 6)
        max_row = random.randint(min_row + 3, rows - 2)  # At least 4x4
        max_col = random.randint(min_col + 3, cols - 2)
        
        # Place corner cells
        corner_positions = [
            (min_row, min_col),
            (min_row, max_col),
            (max_row, min_col),
            (max_row, max_col)
        ]
        
        for r, c in corner_positions:
            grid[r, c] = corner_color
        
        # Define interior dimensions
        interior_height = max_row - min_row - 1
        interior_width = max_col - min_col - 1
        
        if interior_height <= 0 or interior_width <= 0:
            # Fallback if rectangle is too small
            interior_height = max(1, interior_height)
            interior_width = max(1, interior_width)
        
        # Create multiple 8-way connected objects inside the rectangle
        object_canvas = np.zeros((interior_height, interior_width), dtype=int)
        
        # Divide the interior into regions based on number of objects
        if num_objects == 1:
            # For a single object, use the whole interior
            regions = [(0, 0, interior_height, interior_width)]
        elif num_objects == 2:
            # For two objects, split either horizontally or vertically
            if random.choice([True, False]) or interior_width < 4:
                # Split horizontally
                mid_height = interior_height // 2
                regions = [
                    (0, 0, mid_height, interior_width),
                    (mid_height, 0, interior_height, interior_width)
                ]
            else:
                # Split vertically
                mid_width = interior_width // 2
                regions = [
                    (0, 0, interior_height, mid_width),
                    (0, mid_width, interior_height, interior_width)
                ]
        else:  # num_objects == 3
            # For three objects, split into thirds or create a 2x2 grid with one quadrant empty
            if random.choice([True, False]) or min(interior_height, interior_width) < 6:
                # Split into thirds (either horizontally or vertically)
                if interior_height > interior_width:
                    # Split horizontally
                    h1 = interior_height // 3
                    h2 = 2 * interior_height // 3
                    regions = [
                        (0, 0, h1, interior_width),
                        (h1, 0, h2, interior_width),
                        (h2, 0, interior_height, interior_width)
                    ]
                else:
                    # Split vertically
                    w1 = interior_width // 3
                    w2 = 2 * interior_width // 3
                    regions = [
                        (0, 0, interior_height, w1),
                        (0, w1, interior_height, w2),
                        (0, w2, interior_height, interior_width)
                    ]
            else:
                # Create a 2x2 grid with one quadrant empty
                mid_height = interior_height // 2
                mid_width = interior_width // 2
                
                # All potential regions (quadrants)
                potential_regions = [
                    (0, 0, mid_height, mid_width),            # Top-left
                    (0, mid_width, mid_height, interior_width),  # Top-right
                    (mid_height, 0, interior_height, mid_width),  # Bottom-left
                    (mid_height, mid_width, interior_height, interior_width)  # Bottom-right
                ]
                
                # Randomly select which quadrant to leave empty
                empty_idx = random.randrange(4)
                regions = [r for i, r in enumerate(potential_regions) if i != empty_idx]
        
        # Create objects in each region
        for start_r, start_c, end_r, end_c in regions:
            region_height = end_r - start_r
            region_width = end_c - start_c
            
            if region_height <= 0 or region_width <= 0:
                continue
                
            # Create an 8-way connected object for this region
            obj = create_object(
                height=region_height,
                width=region_width,
                color_palette=object_color,
                contiguity=Contiguity.EIGHT,
                background=0
            )
            
            # Ensure the object has at least one colored cell
            if np.all(obj == 0):
                r, c = random.randrange(region_height), random.randrange(region_width)
                obj[r, c] = object_color
            
            # Place the object in its region
            for r in range(region_height):
                for c in range(region_width):
                    if obj[r, c] != 0:
                        object_canvas[start_r + r, start_c + c] = obj[r, c]
        
        # Place the objects inside the rectangle
        for r in range(interior_height):
            for c in range(interior_width):
                if object_canvas[r, c] != 0:
                    grid[min_row + 1 + r, min_col + 1 + c] = object_canvas[r, c]
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Find the four corner cells (they should have the same color)
        all_objects = find_connected_objects(grid, diagonal_connectivity=False, monochromatic=True)
        
        # Group objects by size - corner markers should be single cells
        single_cells = all_objects.with_size(1, 1)
        
        # Find groups of 4 cells with the same color (our corner markers)
        colors = {}
        for obj in single_cells:
            color = list(obj.colors)[0]  # Get the single color of this cell
            colors.setdefault(color, []).append(obj)
        
        # Find the color that has exactly 4 objects
        corner_color = None
        corner_objects = []
        for color, objs in colors.items():
            if len(objs) == 4:
                corner_color = color
                corner_objects = objs
                break
        
        if not corner_objects:
            # Fallback if not found
            return np.zeros((2, 2), dtype=int)
        
        # Get corner coordinates
        corners = []
        for obj in corner_objects:
            r, c, _ = next(iter(obj.cells))  # Get the single cell's coordinates
            corners.append((r, c))
        
        # Find min/max row/col to define the rectangle
        min_row = min(r for r, _ in corners)
        max_row = max(r for r, _ in corners)
        min_col = min(c for _, c in corners)
        max_col = max(c for _, c in corners)
        
        # Extract the interior (excluding the corners)
        interior_height = max_row - min_row - 1
        interior_width = max_col - min_col - 1
        
        # Create output grid with interior contents
        output = np.zeros((interior_height, interior_width), dtype=int)
        
        # Copy interior contents
        for r in range(interior_height):
            for c in range(interior_width):
                output[r, c] = grid[min_row + 1 + r, min_col + 1 + c]
        
        return output

