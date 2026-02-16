from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from Framework.input_library import create_object, retry, Contiguity, enforce_object_height, enforce_object_width
from Framework.transformation_library import find_connected_objects, GridObject, GridObjects

class Task48d8fb45Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain a random number of same-colored (1-9) objects, each consisting of 8-way connected cells, with the object color varying across examples.",
            "The objects are shaped and sized to fit within a 3x3 subgrid, with each object having a distinct shape.",
            "Exactly one {color('topcell')} cell is placed directly above one of the colored objects, ensuring vertical connectivity.",
            "Each colored object should be completely separated from the others."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size 3x3.",
            "They are constructed by identifying all colored objects in the input grid.",
            "Once identified, locate the 8-way connected object that is directly below the {color('topcell')} marker cell.",
            "Extract this object and place it within the 3x3 output grid excluding the {color('topcell')} marker cell."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
       
        taskvars = {
            'rows': random.randint(10, 30),
            'cols': random.randint(10, 30),
            'topcell': random.randint(1, 9)
        }
        # Compute maximum objects that can reasonably fit (each object fits in 3x3)
        max_objects_possible = (taskvars['rows'] // 3) * (taskvars['cols'] // 3)
        max_objects = min(6, max(2, max_objects_possible))
        
        # Determine number of train examples
        n_train = random.randint(3, 4)
        
        # Generate train and test data
        train_data = []
        used_object_colors = set()
        
        for _ in range(n_train):
            # Each example may have a different number of objects (2..max_objects)
            num_objects = random.randint(2, max_objects)
            
            # Choose a color for objects different from topcell and not used before
            object_color_options = [c for c in range(1, 10) 
                                   if c != taskvars['topcell'] and c not in used_object_colors]
            
            # If we're running out of colors, reset the used colors
            if not object_color_options:
                used_object_colors.clear()
                object_color_options = [c for c in range(1, 10) if c != taskvars['topcell']]
                
            object_color = random.choice(object_color_options)
            used_object_colors.add(object_color)
            
            gridvars = {
                'num_objects': num_objects,
                'object_color': object_color
            }
            
            # Create input grid with guaranteed marker placement
            input_grid = retry(
                lambda: self.create_input(taskvars, gridvars),
                lambda grid: self._verify_topcell_present(grid, taskvars['topcell']),
                max_attempts=50
            )
            
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Generate test data (always with just one example)
        num_objects = random.randint(2, max_objects)  # Test can have 2..max_objects objects
        
        # Choose a different color for test example
        object_color_options = [c for c in range(1, 10) 
                              if c != taskvars['topcell'] and c not in used_object_colors]
        if not object_color_options:
            object_color_options = [c for c in range(1, 10) if c != taskvars['topcell']]
        
        object_color = random.choice(object_color_options)
        
        gridvars = {
            'num_objects': num_objects,
            'object_color': object_color
        }
        
        # Create test input grid with guaranteed marker placement
        test_input = retry(
            lambda: self.create_input(taskvars, gridvars),
            lambda grid: self._verify_topcell_present(grid, taskvars['topcell']),
            max_attempts=50
        )
        
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_data, 'test': test_data}
    
    def _verify_topcell_present(self, grid: np.ndarray, topcell_color: int) -> bool:
        """Verify that the topcell marker is present in the grid."""
        return np.any(grid == topcell_color)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows, cols = taskvars['rows'], taskvars['cols']
        object_color = gridvars.get('object_color')
        topcell_color = taskvars['topcell']
        # Fallback to a size-dependent number of objects when not provided
        max_objects_possible = (rows // 3) * (cols // 3)
        default_max_objects = min(6, max(2, max_objects_possible))
        num_objects = gridvars.get('num_objects', random.randint(2, default_max_objects))
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Generate distinct small shapes that fit in 3x3 grid and have cells in each row and column
        shapes = self._generate_distinct_shapes(num_objects, object_color)
        
        # Place shapes on grid with sufficient spacing
        placed_objects = self._place_shapes_with_spacing(grid, shapes)
        
        # If no objects could be placed, return empty grid (will be retried)
        if not placed_objects:
            return grid
            
        # Choose one object to place the marker above
        target_object = random.choice(placed_objects)
        
        # Find top-most cell of the target object
        top_cells = [(r, c) for r, c, _ in target_object.cells 
                    if all((r-1, c, _) not in target_object.cells for _ in range(10))]
        
        # If no top cells found, return grid (will be retried)
        if not top_cells:
            return grid
            
        top_r, top_c = min(top_cells, key=lambda x: x[0])
        
        # Place marker above the object, ensuring it's within grid bounds
        marker_r = top_r - 1
        if marker_r >= 0:  # Ensure we're within grid bounds
            grid[marker_r, top_c] = topcell_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Create a 3x3 output grid
        output_grid = np.zeros((3, 3), dtype=int)
        
        # Find the marker cell
        marker_color = taskvars['topcell']
        marker_positions = np.where(grid == marker_color)
        
        # If no marker found, return empty output
        if len(marker_positions[0]) == 0:
            return output_grid
        
        marker_r, marker_c = marker_positions[0][0], marker_positions[1][0]
        
        # Find all colored objects
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        # Find the object directly below the marker
        target_object = None
        
        for obj in objects:
            # Skip the marker itself
            if marker_color in obj.colors:
                continue
                
            # Check if any cell of this object is directly below the marker
            below_cells = [(r, c) for r, c, _ in obj.cells if c == marker_c and r > marker_r]
            
            if below_cells:
                # Found the target object
                target_object = obj
                break
        
        # If target object found, extract it to output grid
        if target_object:
            obj_array = target_object.to_array()
            
            # Center the object in the 3x3 output grid
            h, w = obj_array.shape
            r_start = (3 - h) // 2
            c_start = (3 - w) // 2
            
            # Paste the object into the output grid
            for r in range(h):
                for c in range(w):
                    if obj_array[r, c] != 0:
                        output_r, output_c = r_start + r, c_start + c
                        if 0 <= output_r < 3 and 0 <= output_c < 3:
                            output_grid[output_r, output_c] = obj_array[r, c]
        
        return output_grid
    
    def _generate_distinct_shapes(self, num_shapes: int, color: int) -> List[np.ndarray]:
        """Generate distinct shapes that fit within a 3x3 grid with cells in each row and column."""
        shapes = []
        existing_shapes = set()
        
        def shape_to_tuple(arr):
            """Convert a shape array to a hashable tuple representation."""
            return tuple(map(tuple, arr))
        
        while len(shapes) < num_shapes:
            # Create a random shape within 3x3 grid that has cells in each row and column
            def create_full_shape():
                return enforce_object_height(
                    lambda: enforce_object_width(
                        lambda: create_object(
                            height=3, 
                            width=3, 
                            color_palette=color,
                            contiguity=Contiguity.EIGHT,
                            background=0
                        )
                    )
                )
            
            shape = retry(
                create_full_shape,
                lambda x: (
                    np.any(x != 0) and                       # At least one colored cell
                    self._is_fully_connected(x, color) and   # 8-way connected
                    np.all(np.any(x != 0, axis=1)) and       # Each row has at least one colored cell  
                    np.all(np.any(x != 0, axis=0))           # Each column has at least one colored cell
                )
            )
            
            # Convert to tuple for comparison
            shape_tuple = shape_to_tuple(shape)
            
            # Check if this shape is distinct from existing ones
            if shape_tuple not in existing_shapes:
                existing_shapes.add(shape_tuple)
                shapes.append(shape)
        
        return shapes
    
    def _is_fully_connected(self, grid: np.ndarray, color: int) -> bool:
        """Check if all cells of the given color form a single 8-connected component."""
        # Find all objects with 8-way connectivity
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0)
        
        # Filter objects with the specified color
        color_objects = [obj for obj in objects if color in obj.colors]
        
        # There should be exactly one object with this color
        return len(color_objects) == 1
    
    def _place_shapes_with_spacing(
        self, grid: np.ndarray, shapes: List[np.ndarray]) -> List[GridObject]:
        """Place shapes on the grid with at least 2 cells spacing between them."""
        placed_objects = []
        
        for shape in shapes:
            shape_height, shape_width = shape.shape
            
            # Define valid placement predicate
            def is_valid_placement(position):
                r, c = position
                
                # Check if shape fits within grid
                if r + shape_height > grid.shape[0] or c + shape_width > grid.shape[1]:
                    return False
                
                # Check spacing from other objects
                for existing_obj in placed_objects:
                    for er, ec, _ in existing_obj.cells:
                        for sr in range(shape_height):
                            for sc in range(shape_width):
                                if shape[sr, sc] != 0:
                                    # Calculate distance between cells
                                    dr = abs(r + sr - er)
                                    dc = abs(c + sc - ec)
                                    
                                    # Require at least 2 cells spacing
                                    if dr < 2 and dc < 2:
                                        return False
                
                return True
            
            # Try to find a valid position
            max_attempts = 100
            found_placement = False
            
            for _ in range(max_attempts):
                r = random.randint(0, grid.shape[0] - shape_height)
                c = random.randint(0, grid.shape[1] - shape_width)
                
                if is_valid_placement((r, c)):
                    # Place the shape
                    for sr in range(shape_height):
                        for sc in range(shape_width):
                            if shape[sr, sc] != 0:
                                grid[r + sr, c + sc] = shape[sr, sc]
                    
                    # Create and save the object
                    obj_cells = set()
                    for sr in range(shape_height):
                        for sc in range(shape_width):
                            if shape[sr, sc] != 0:
                                obj_cells.add((r + sr, c + sc, shape[sr, sc]))
                    
                    placed_objects.append(GridObject(obj_cells))
                    found_placement = True
                    break
            
            # If we couldn't place this shape, continue with the next one
            if not found_placement:
                continue
        
        return placed_objects
