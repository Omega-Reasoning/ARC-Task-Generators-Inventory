from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry, random_cell_coloring, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task14b8e18cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}×{vars['grid_size']}.",
            "Each grid has a completely filled background of {color('background')} color, along with several same-colored objects. These objects can be filled or unfilled squares and rectangles, and sometimes include patterns like [[{color('background')}, c], [c, {color('background')}]], where c is the object color.",
            "Each grid contains at least one square-shaped object.",
            "All objects must be completely separated from each other.",
            "In some cases, a smaller object may appear inside an unfilled rectangle or square, but it must not touch the surrounding shape."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying all filled and unfilled squares.",
            "Once identified, 8 {color('background')} cells around each of these squares are recolored to {color('fill')}.",
            "The recolored {color('fill')} cells are the exterior cells surrounding the square.",
            "Specifically, consider the 4 corner cells of each square, and recolor the adjacent 4-way connected exterior cells—those directly above, below, to the left, or to the right of the square—but only if they lie outside the square."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'grid_size': random.randint(10, 30),  # Increased minimum to accommodate 2-layer buffer
            'fill': random.randint(1, 9),
            'background': random.randint(1, 9)
        }
        
        # Ensure different colors
        while taskvars['fill'] == taskvars['background']:
            taskvars['fill'] = random.randint(1, 9)
        
        # Create 3 train examples and 1 test example
        train_data = []
        test_data = []
        
        # Ensure at least one unfilled square appears in training data
        has_unfilled_square = False
        
        # Choose which training grid gets the special diagonal object
        special_grid_index = random.randint(0, 2)
        
        for i in range(3):
            # Force at least one unfilled square in the training set
            force_unfilled = not has_unfilled_square and i == 2
            
            # Add diagonal object to exactly one training grid
            add_diagonal_object = (i == special_grid_index)
            
            input_grid = self.create_input(taskvars, {
                'force_unfilled': force_unfilled,
                'add_diagonal_object': add_diagonal_object
            })
            output_grid = self.transform_input(input_grid, taskvars)
            
            # Check if this grid has unfilled squares
            if self._has_unfilled_square(input_grid, taskvars):
                has_unfilled_square = True
                
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_data.append({'input': test_input, 'output': test_output})
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def _has_unfilled_square(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> bool:
        """Check if grid contains any unfilled squares."""
        objects = find_connected_objects(grid, background=taskvars['background'])
        for obj in objects:
            if self._is_square(obj) and self._is_unfilled_square(obj, grid, taskvars):
                return True
        return False
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        background = taskvars['background']
        
        # Create grid filled with background color
        grid = np.full((grid_size, grid_size), background, dtype=int)
        
        # Choose object color (different from background and fill)
        object_colors = [c for c in range(1, 10) if c != background and c != taskvars['fill']]
        object_color = random.choice(object_colors)
        
        # Create several objects including at least one square
        num_objects = random.randint(2, 4)  # Reduced to accommodate 2-layer buffer
        objects_created = 0
        squares_created = 0
        
        force_unfilled = gridvars.get('force_unfilled', False)
        
        # Keep track of placed objects to ensure 2-layer separation
        placed_objects = []
        
        max_attempts = 100
        attempts = 0
        
        while (objects_created < num_objects or squares_created == 0) and attempts < max_attempts:
            attempts += 1
            
            # Randomly choose object type and size
            is_square = random.random() < 0.5 or squares_created == 0
            
            if is_square:
                width = height = random.randint(2, min(4, grid_size // 4))  # Reduced max size
            else:
                width = random.randint(2, min(4, grid_size // 4))
                height = random.randint(2, min(4, grid_size // 4))
                # Ensure it's not accidentally a square
                while width == height:
                    height = random.randint(2, min(4, grid_size // 4))
            
            # Choose position with 2-layer buffer from edges
            buffer = 2
            max_row = grid_size - height - buffer
            max_col = grid_size - width - buffer
            
            if max_row < buffer or max_col < buffer:
                continue
                
            row = random.randint(buffer, max_row)
            col = random.randint(buffer, max_col)
            
            # Check if position maintains 2-layer separation from all existing objects
            if self._has_sufficient_separation(row, col, height, width, placed_objects, buffer):
                
                # Decide if filled or unfilled
                if is_square and force_unfilled and squares_created == 0:
                    filled = False
                else:
                    filled = random.random() < 0.6
                
                self._place_object(grid, row, col, height, width, object_color, filled, background)
                
                # Record placed object
                placed_objects.append({
                    'row': row,
                    'col': col,
                    'height': height,
                    'width': width
                })
                
                objects_created += 1
                
                if is_square:
                    squares_created += 1
        
        # Add diagonal object if requested
        if gridvars.get('add_diagonal_object', False):
            self._add_diagonal_object(grid, object_color, background, placed_objects)
        
        return grid
    
    def _has_sufficient_separation(self, row: int, col: int, height: int, width: int, 
                                  placed_objects: List[Dict], buffer: int) -> bool:
        """Check if a new object has sufficient separation from all existing objects."""
        # Create bounding box for new object with buffer
        new_min_row = row - buffer
        new_max_row = row + height + buffer - 1
        new_min_col = col - buffer
        new_max_col = col + width + buffer - 1
        
        # Check against all existing objects
        for obj in placed_objects:
            # Create bounding box for existing object with buffer
            existing_min_row = obj['row'] - buffer
            existing_max_row = obj['row'] + obj['height'] + buffer - 1
            existing_min_col = obj['col'] - buffer
            existing_max_col = obj['col'] + obj['width'] + buffer - 1
            
            # Check if bounding boxes overlap
            if not (new_max_row < existing_min_row or new_min_row > existing_max_row or
                   new_max_col < existing_min_col or new_min_col > existing_max_col):
                return False
        
        return True
    
    def _add_diagonal_object(self, grid: np.ndarray, object_color: int, background: int, 
                           placed_objects: List[Dict]):
        """Add a 2-cell diagonal object to the grid with proper separation."""
        grid_height, grid_width = grid.shape
        
        # Define the 2 possible diagonal patterns
        diagonal_patterns = [
            [(0, 0), (1, 1)],  # top-left to bottom-right
            [(0, 1), (1, 0)]   # top-right to bottom-left
        ]
        
        # Choose a random diagonal pattern
        pattern = random.choice(diagonal_patterns)
        
        # Try to place the diagonal object
        max_attempts = 100
        buffer = 2
        
        for _ in range(max_attempts):
            # Choose a random position for the 2x2 bounding box with buffer from edges
            row = random.randint(buffer, grid_height - 2 - buffer)
            col = random.randint(buffer, grid_width - 2 - buffer)
            
            # Check if both cells in the pattern are background color
            cell1_row, cell1_col = row + pattern[0][0], col + pattern[0][1]
            cell2_row, cell2_col = row + pattern[1][0], col + pattern[1][1]
            
            if (grid[cell1_row, cell1_col] == background and 
                grid[cell2_row, cell2_col] == background):
                
                # Check if there's enough separation from other objects
                if self._is_diagonal_placement_valid(row, col, placed_objects, buffer):
                    # Place the diagonal object
                    grid[cell1_row, cell1_col] = object_color
                    grid[cell2_row, cell2_col] = object_color
                    break
    
    def _is_diagonal_placement_valid(self, row: int, col: int, 
                                   placed_objects: List[Dict], buffer: int) -> bool:
        """Check if diagonal object placement has sufficient separation from other objects."""
        # The diagonal object occupies a 2x2 bounding box
        diagonal_height = 2
        diagonal_width = 2
        
        return self._has_sufficient_separation(row, col, diagonal_height, diagonal_width, 
                                             placed_objects, buffer)
    
    def _place_object(self, grid: np.ndarray, row: int, col: int, 
                     height: int, width: int, color: int, filled: bool, background: int):
        """Place a filled or unfilled rectangle/square on the grid."""
        if filled:
            # Fill entire rectangle
            grid[row:row+height, col:col+width] = color
        else:
            # Create unfilled rectangle (border only)
            grid[row:row+height, col:col+width] = background  # Clear area first
            grid[row, col:col+width] = color  # Top edge
            grid[row+height-1, col:col+width] = color  # Bottom edge
            grid[row:row+height, col] = color  # Left edge
            grid[row:row+height, col+width-1] = color  # Right edge
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        background = taskvars['background']
        fill = taskvars['fill']
        
        # Create copy of input grid
        output = grid.copy()
        
        # Find all connected objects
        objects = find_connected_objects(grid, background=background)
        
        # Process each object to check if it's a square
        for obj in objects:
            if self._is_square(obj):
                self._outline_square(output, obj, fill, background)
        
        return output
    
    def _is_square(self, obj: GridObject) -> bool:
        """Check if an object forms a square shape."""
        return obj.width == obj.height and obj.width >= 2
    
    def _is_unfilled_square(self, obj: GridObject, grid: np.ndarray, taskvars: Dict[str, Any]) -> bool:
        """Check if a square object is unfilled."""
        bbox = obj.bounding_box
        height = bbox[0].stop - bbox[0].start
        width = bbox[1].stop - bbox[1].start
        
        if height != width or height < 2:
            return False
        
        # Check if interior is background color
        interior = grid[bbox[0].start+1:bbox[0].stop-1, bbox[1].start+1:bbox[1].stop-1]
        return np.all(interior == taskvars['background'])
    
    def _outline_square(self, grid: np.ndarray, square: GridObject, fill: int, background: int):
        """Add outline around a square by coloring 8 cells around corners."""
        bbox = square.bounding_box
        top_row = bbox[0].start
        bottom_row = bbox[0].stop - 1
        left_col = bbox[1].start
        right_col = bbox[1].stop - 1
        
        grid_height, grid_width = grid.shape
        
        # Define the 8 positions around the 4 corners
        outline_positions = [
            # Around top-left corner
            (top_row - 1, left_col),     # above
            (top_row, left_col - 1),     # left
            # Around top-right corner
            (top_row - 1, right_col),    # above
            (top_row, right_col + 1),    # right
            # Around bottom-left corner
            (bottom_row, left_col - 1),  # left
            (bottom_row + 1, left_col),  # below
            # Around bottom-right corner
            (bottom_row, right_col + 1), # right
            (bottom_row + 1, right_col)  # below
        ]
        
        # Color valid positions that are currently background
        for row, col in outline_positions:
            if (0 <= row < grid_height and 0 <= col < grid_width and 
                grid[row, col] == background):
                grid[row, col] = fill
