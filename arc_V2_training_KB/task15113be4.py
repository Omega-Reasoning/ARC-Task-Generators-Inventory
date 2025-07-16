from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry, Contiguity
import numpy as np
import random

class Task15113be4Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {4*vars['grid_size'] + 3} × {4*vars['grid_size'] + 3}.",
            "Each input grid is divided into 3×3 subgrids using {color('grid_color')} vertical and horizontal lines placed at every 4th row and column (i.e., rows and columns 4, 8, and so on).",
            "Inside each of these 3×3 subgrids, an 8-way connected {color('object_color')} object is placed.",
            "In exactly one of the corners (top-left, top-right, bottom-left, or bottom-right), instead of a 3×3 subgrid, there is a larger 6×6 subgrid completely enclosed by a {color('grid_color')} frame.",
            "Inside this 6×6 subgrid lies a larger object formed by first creating a 3×3 object and then doubling the size of each of its cells, resulting in a scaled-up version that fits the 6×6 space. This enlarged object uses a new color different from {color('object_color')}.",
            "The position of this 6x6 subgrid should vary across examples,between top left right bottom left right corners."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the 6×6 subgrid that contains the larger (scaled) object.",
            "This larger object is used as a reference shape.",
            "Next, all {color('object_color')} objects in the 3×3 subgrids are examined.",
            "For each such object, if its shape matches the structure of the larger object when scaled down (i.e., it has all the corresponding cells needed), then all its original {color('object_color')} cells are recolored to the new color used in the 6×6 object.",
            "If the object in a 3×3 subgrid does not contain all the required cells to form the shape, it is left unchanged.",
            "The final output grid retains the same {color('grid_color')} structure, dividing the grid into multiple 3×3 subgrids and one 6×6 subgrid, with some objects in the 3×3 subgrids partially or fully recolored to replicate the reference shape from the 6×6 area."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        taskvars = {
            'grid_size': random.choice([3, 4, 5, 6]),
            'object_color': random.randint(1, 9),
            'grid_color': random.randint(1, 9)
        }
        
        # Ensure different colors
        while taskvars['grid_color'] == taskvars['object_color']:
            taskvars['grid_color'] = random.randint(1, 9)
        
        # Create reference color different from both
        ref_color = random.randint(1, 9)
        while ref_color in [taskvars['object_color'], taskvars['grid_color']]:
            ref_color = random.randint(1, 9)
        taskvars['ref_color'] = ref_color
        
        # Generate examples with different corner positions
        train_examples = []
        corner_positions = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
        
        for i in range(3):
            corner = corner_positions[i]
            gridvars = {'corner_position': corner}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Test example with remaining corner
        test_corner = corner_positions[3]
        test_gridvars = {'corner_position': test_corner}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        object_color = taskvars['object_color']
        grid_color = taskvars['grid_color']
        ref_color = taskvars['ref_color']
        corner_position = gridvars['corner_position']
        
        # Create grid
        full_size = 4 * grid_size + 3
        grid = np.zeros((full_size, full_size), dtype=int)
        
        # Step 1: Draw ALL grid lines first (complete 3x3 division)
        for i in range(1, grid_size + 1):
            line_pos = 4 * i - 1
            if line_pos < full_size:
                grid[line_pos, :] = grid_color  # horizontal lines
                grid[:, line_pos] = grid_color  # vertical lines
        
        # Step 2: Determine 8x8 frame position and clear internal lines
        if corner_position == 'top-left':
            frame_start_row, frame_start_col = 0, 0
            # Clear internal lines in 8x8 area
            grid[3, 0:8] = 0  # Remove horizontal line at row 3
            grid[7, 0:8] = 0  # Remove horizontal line at row 7
            grid[0:8, 3] = 0  # Remove vertical line at col 3
            grid[0:8, 7] = 0  # Remove vertical line at col 7
            
        elif corner_position == 'top-right':
            frame_start_row, frame_start_col = 0, full_size - 8
            # Clear internal lines in 8x8 area
            grid[3, frame_start_col:frame_start_col+8] = 0  # Remove horizontal line at row 3
            grid[7, frame_start_col:frame_start_col+8] = 0  # Remove horizontal line at row 7
            grid[0:8, frame_start_col + 3] = 0  # Remove vertical line
            grid[0:8, frame_start_col + 7] = 0  # Remove vertical line
            
        elif corner_position == 'bottom-left':
            frame_start_row, frame_start_col = full_size - 8, 0
            # Clear internal lines in 8x8 area
            grid[frame_start_row + 3, 0:8] = 0  # Remove horizontal line
            grid[frame_start_row + 7, 0:8] = 0  # Remove horizontal line
            grid[frame_start_row:frame_start_row+8, 3] = 0  # Remove vertical line at col 3
            grid[frame_start_row:frame_start_row+8, 7] = 0  # Remove vertical line at col 7
            
        else:  # bottom-right
            frame_start_row, frame_start_col = full_size - 8, full_size - 8
            # Clear internal lines in 8x8 area
            grid[frame_start_row + 3, frame_start_col:frame_start_col+8] = 0  # Remove horizontal line
            grid[frame_start_row + 7, frame_start_col:frame_start_col+8] = 0  # Remove horizontal line
            grid[frame_start_row:frame_start_row+8, frame_start_col + 3] = 0  # Remove vertical line
            grid[frame_start_row:frame_start_row+8, frame_start_col + 7] = 0  # Remove vertical line
        
        # Step 3: Create complete 8x8 frame
        # Fill entire 8x8 area with grid_color first
        grid[frame_start_row:frame_start_row+8, frame_start_col:frame_start_col+8] = grid_color
        
        # Clear the inner 6x6 area (leaving 1-cell border)
        grid[frame_start_row+1:frame_start_row+7, frame_start_col+1:frame_start_col+7] = 0
        
        # Step 4: Create reference 3x3 shape
        reference_shape = retry(
            lambda: create_object(3, 3, object_color, Contiguity.EIGHT),
            lambda x: np.sum(x != 0) >= 2 and np.sum(x != 0) <= 6
        )
        
        # Store reference shape for transformation
        self.reference_shape = reference_shape
        
        # Step 5: Create 6x6 scaled version
        scaled_shape = np.zeros((6, 6), dtype=int)
        for r in range(3):
            for c in range(3):
                if reference_shape[r, c] != 0:
                    scaled_shape[2*r:2*r+2, 2*c:2*c+2] = ref_color
        
        # Step 6: Place 6x6 object inside the frame
        object_start_row = frame_start_row + 1
        object_start_col = frame_start_col + 1
        
        for r in range(6):
            for c in range(6):
                if scaled_shape[r, c] != 0:
                    grid[object_start_row + r, object_start_col + c] = scaled_shape[r, c]
        
        # Step 7: Find ALL 3x3 subgrids that don't overlap with 8x8 frame
        objects_to_create = []
        for r in range(grid_size + 1):
            for c in range(grid_size + 1):
                subgrid_start_row = r * 4
                subgrid_start_col = c * 4
                
                # Skip if subgrid would go beyond grid boundaries
                if subgrid_start_row + 3 > full_size or subgrid_start_col + 3 > full_size:
                    continue
                
                # Check if this 3x3 subgrid overlaps with the 8x8 frame area
                overlaps = False
                if (subgrid_start_row < frame_start_row + 8 and 
                    subgrid_start_row + 3 > frame_start_row and
                    subgrid_start_col < frame_start_col + 8 and 
                    subgrid_start_col + 3 > frame_start_col):
                    overlaps = True
                
                if not overlaps:
                    objects_to_create.append((r, c))
        
        # Determine number of matching objects (2-4, but not more than available objects)
        max_possible_matching = min(4, len(objects_to_create))
        min_matching = min(2, len(objects_to_create))
        
        if max_possible_matching >= min_matching:
            num_matching = random.randint(min_matching, max_possible_matching)
        else:
            num_matching = len(objects_to_create)  # Use all available if less than minimum
        
        matching_positions = random.sample(objects_to_create, num_matching) if objects_to_create else []
        
        # Store for debugging
        self.num_matching_objects = num_matching
        
        # Place objects in ALL valid 3x3 subgrids
        for r, c in objects_to_create:
            start_row = r * 4
            start_col = c * 4
            
            if (r, c) in matching_positions:
                # Create object that contains the reference shape
                obj = self._create_matching_object(reference_shape, object_color)
            else:
                # Create different shaped object
                obj = self._create_different_object(reference_shape, object_color)
            
            # Place object in 3x3 subgrid
            for obj_r in range(3):
                for obj_c in range(3):
                    if obj[obj_r, obj_c] != 0:
                        grid[start_row + obj_r, start_col + obj_c] = obj[obj_r, obj_c]
        
        return grid
    
    def _create_matching_object(self, reference_shape, object_color):
        """Create an object that contains the reference shape"""
        def generator():
            obj = create_object(3, 3, object_color, Contiguity.EIGHT)
            # Ensure it contains all reference cells
            for r in range(3):
                for c in range(3):
                    if reference_shape[r, c] != 0:
                        obj[r, c] = object_color
            return obj
        
        return retry(generator, lambda x: np.sum(x != 0) >= np.sum(reference_shape != 0))
    
    def _create_different_object(self, reference_shape, object_color):
        """Create an object with different shape from reference"""
        def generator():
            return create_object(3, 3, object_color, Contiguity.EIGHT)
        
        def is_different(obj):
            # Check if object has different shape (missing some reference cells)
            ref_cells = set(zip(*np.where(reference_shape != 0)))
            obj_cells = set(zip(*np.where(obj != 0)))
            return not ref_cells.issubset(obj_cells)
        
        return retry(generator, is_different)
    
    def transform_input(self, grid, taskvars):
        """Main transformation function that orchestrates the shape matching process"""
        output_grid = grid.copy()
        
        # Extract reference shape pattern from the 6x6 subgrid in the current grid
        reference_pattern = self._extract_reference_pattern_from_grid(grid, taskvars['ref_color'], taskvars['grid_size'])
        
        # Find all valid 3x3 subgrids and process them efficiently
        valid_subgrids = self._find_processable_3x3_subgrids(grid, taskvars['grid_size'], taskvars['ref_color'])
        
        # Process each valid subgrid for shape matching and recoloring
        for subgrid_info in valid_subgrids:
            self._process_subgrid_for_shape_matching(output_grid, grid, subgrid_info, reference_pattern, taskvars['object_color'], taskvars['ref_color'])
        
        return output_grid
    
    def _extract_reference_pattern_from_grid(self, grid, ref_color, grid_size):
        """Extract the reference shape pattern from the 6x6 subgrid in the current grid"""
        full_size = 4 * grid_size + 3
        
        # Find the 6x6 subgrid by looking for the ref_color
        corner_positions = [
            (0, 0),  # top-left
            (0, full_size - 8),  # top-right
            (full_size - 8, 0),  # bottom-left
            (full_size - 8, full_size - 8)  # bottom-right
        ]
        
        for frame_start_row, frame_start_col in corner_positions:
            # Check if this position contains the 6x6 subgrid with ref_color
            if (frame_start_row + 8 <= full_size and frame_start_col + 8 <= full_size):
                subgrid_6x6 = grid[frame_start_row+1:frame_start_row+7, frame_start_col+1:frame_start_col+7]
                if ref_color in subgrid_6x6:
                    # Found the 6x6 subgrid, now extract the 3x3 pattern
                    pattern_3x3 = self._scale_down_6x6_to_3x3(subgrid_6x6, ref_color)
                    return set(zip(*np.where(pattern_3x3 != 0)))
        
        # If no 6x6 subgrid found, return empty set
        return set()
    
    def _scale_down_6x6_to_3x3(self, subgrid_6x6, ref_color):
        """Scale down a 6x6 subgrid to extract the original 3x3 pattern"""
        pattern_3x3 = np.zeros((3, 3), dtype=int)
        
        for r in range(3):
            for c in range(3):
                # Check if the 2x2 block in the 6x6 subgrid contains the ref_color
                block_2x2 = subgrid_6x6[2*r:2*r+2, 2*c:2*c+2]
                if ref_color in block_2x2:
                    pattern_3x3[r, c] = 1  # Use 1 to indicate presence
        
        return pattern_3x3
    
    def _find_processable_3x3_subgrids(self, grid, grid_size, ref_color):
        """Find all valid 3x3 subgrids that should be processed for transformation"""
        processable_subgrids = []
        
        for r in range(grid_size + 1):
            for c in range(grid_size + 1):
                start_row = r * 4
                start_col = c * 4
                
                # Skip if subgrid would go beyond grid boundaries
                if start_row + 3 > grid.shape[0] or start_col + 3 > grid.shape[1]:
                    continue
                
                # Extract and validate subgrid in one step
                subgrid = grid[start_row:start_row+3, start_col:start_col+3]
                
                # Check if subgrid is valid for processing
                if subgrid.shape == (3, 3) and ref_color not in subgrid:
                    subgrid_info = {
                        'grid_row': r,
                        'grid_col': c,
                        'start_row': start_row,
                        'start_col': start_col,
                        'subgrid': subgrid  # Store the subgrid to avoid re-extraction
                    }
                    processable_subgrids.append(subgrid_info)
        
        return processable_subgrids
    
    def _process_subgrid_for_shape_matching(self, output_grid, input_grid, subgrid_info, reference_pattern, object_color, ref_color):
        """Process a single subgrid: check for shape matching and recolor if needed"""
        # Extract object cells from the stored subgrid
        subgrid = subgrid_info['subgrid']
        object_cells_in_subgrid = set(zip(*np.where(subgrid == object_color)))
        
        # Check if object contains all reference pattern cells (shape matching)
        if reference_pattern.issubset(object_cells_in_subgrid):
            # Recolor the matching reference shape cells
            start_row = subgrid_info['start_row']
            start_col = subgrid_info['start_col']
            
            for pattern_r, pattern_c in reference_pattern:
                output_grid[start_row + pattern_r, start_col + pattern_c] = ref_color

