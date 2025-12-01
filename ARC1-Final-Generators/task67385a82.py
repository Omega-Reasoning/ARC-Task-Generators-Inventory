from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, retry, Contiguity, random_cell_coloring
from transformation_library import find_connected_objects, GridObject, GridObjects

class Task67385a82Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of different sizes.",
            "They contain several {color('object_color')} objects, where each object is made of 4-way connected cells.",
            "In addition to the objects, there are also single {color('object_color')} cells scattered throughout the grid.",
            "Each {color('object_color')} object is completely separated from the others, while some single {color('object_color')} cells may be diagonally connected to a {color('object_color')} object.",
            "All other cells remain empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying all {color('object_color')} objects made of more than one 4-way connected  {color('object_color')} cells.",
            "Once identified, the color of each {color('object_color')} object is changed to {color('object_color2')}.",
            "All single {color('object_color')} cells remain unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'object_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
        }
        
        # Ensure object_color2 is different from object_color
        available_colors = [c for c in range(1, 10) if c != taskvars['object_color']]
        taskvars['object_color2'] = random.choice(available_colors)
        
        # Generate 4 train examples and 1 test example
        train_examples = []
        for _ in range(4):
            grid_height = random.randint(6, 15)
            grid_width = random.randint(6, 15)
            gridvars = {
                'height': grid_height,
                'width': grid_width,
                'num_objects': random.randint(2, 4),
                'num_single_cells': random.randint(2, 6)
            }
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        grid_height = random.randint(8, 15)
        grid_width = random.randint(8, 15)
        gridvars = {
            'height': grid_height,
            'width': grid_width,
            'num_objects': random.randint(3, 5),
            'num_single_cells': random.randint(3, 7)
        }
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': input_grid, 'output': output_grid}]
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        height = gridvars['height']
        width = gridvars['width']
        num_objects = gridvars['num_objects']
        num_single_cells = gridvars['num_single_cells']
        object_color = taskvars['object_color']
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Place connected objects
        placed_objects = 0
        max_attempts = 100
        attempts = 0
        
        while placed_objects < num_objects and attempts < max_attempts:
            attempts += 1
            
            # Create a random object
            obj_height = random.randint(2, min(4, height // 2))
            obj_width = random.randint(2, min(4, width // 2))
            
            obj = retry(
                lambda: create_object(
                    obj_height, 
                    obj_width, 
                    object_color, 
                    contiguity=Contiguity.FOUR,
                    background=0
                ),
                lambda x: np.sum(x > 0) >= 2  # Ensure at least 2 cells
            )
            
            # Find potential placement positions
            r_pos = random.randint(0, height - obj_height)
            c_pos = random.randint(0, width - obj_width)
            
            # Check if position is valid (no overlap with existing objects)
            valid_position = True
            for r in range(r_pos, r_pos + obj_height):
                for c in range(c_pos, c_pos + obj_width):
                    if obj[r - r_pos, c - c_pos] > 0:
                        # Check 4-connected neighborhood to ensure no 4-connectivity with other objects
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < height and 0 <= nc < width:
                                if grid[nr, nc] == object_color:
                                    valid_position = False
                                    break
                        
                        # Also check 8-connected neighborhood to ensure no connectivity with other objects
                        if valid_position:  # Only check if still valid
                            for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < height and 0 <= nc < width:
                                    # Check if the diagonal cell is part of an object (not a single cell)
                                    if grid[nr, nc] == object_color:
                                        # Look at its neighbors to determine if it's an object or single cell
                                        is_part_of_object = False
                                        for dr2, dc2 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                            nr2, nc2 = nr + dr2, nc + dc2
                                            if 0 <= nr2 < height and 0 <= nc2 < width:
                                                if grid[nr2, nc2] == object_color and not (nr2 == r and nc2 == c):
                                                    is_part_of_object = True
                                                    break
                                        
                                        # If it's part of an object, this position is invalid
                                        if is_part_of_object:
                                            valid_position = False
                                            break
                                    
                        if not valid_position:
                            break
                if not valid_position:
                    break
            
            if valid_position:
                # Place the object
                for r in range(r_pos, r_pos + obj_height):
                    for c in range(c_pos, c_pos + obj_width):
                        if obj[r - r_pos, c - c_pos] > 0:
                            grid[r, c] = obj[r - r_pos, c - c_pos]
                
                placed_objects += 1
                attempts = 0  # Reset attempts counter after successful placement
        
        # Place single cells
        placed_singles = 0
        attempts = 0
        
        while placed_singles < num_single_cells and attempts < max_attempts:
            attempts += 1
            
            r = random.randint(0, height - 1)
            c = random.randint(0, width - 1)
            
            # Check if position is valid (not already occupied and not 4-connected to an object)
            if grid[r, c] == 0:
                valid_position = True
                
                # Check 4-connected neighborhood - no connection to any object_color cell
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if grid[nr, nc] == object_color:
                            valid_position = False
                            break
                
                # Check 8-connected neighborhood - no connection to another single cell
                if valid_position:
                    for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if grid[nr, nc] == object_color:
                                # Check if it's a single cell by examining its 4-neighbors
                                is_single_cell = True
                                for dr2, dc2 in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                    nr2, nc2 = nr + dr2, nc + dc2
                                    if 0 <= nr2 < height and 0 <= nc2 < width:
                                        if grid[nr2, nc2] == object_color:
                                            is_single_cell = False
                                            break
                                
                                # If it's a single cell, this position is invalid
                                if is_single_cell:
                                    valid_position = False
                                    break
                
                if valid_position:
                    grid[r, c] = object_color
                    placed_singles += 1
                    attempts = 0  # Reset attempts counter after successful placement
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        object_color = taskvars['object_color']
        object_color2 = taskvars['object_color2']
        
        # Create a copy of the input grid
        output_grid = np.copy(grid)
        
        # Find all connected objects of the specified color
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Filter objects by color and size (more than 1 cell)
        multi_cell_objects = objects.filter(lambda obj: obj.has_color(object_color) and obj.size > 1)
        
        # Change the color of multi-cell objects
        for obj in multi_cell_objects:
            for r, c, _ in obj.cells:
                output_grid[r, c] = object_color2
        
        return output_grid

