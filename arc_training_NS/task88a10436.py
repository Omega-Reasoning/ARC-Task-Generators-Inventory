from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject, GridObjects, find_connected_objects
from input_library import create_object, retry, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class task88a10436(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains one object that fits exactly within a {vars['n']} × {vars['n']} bounding box, where {vars['n']} is always an odd number.",
            "The cells in the object have 4-connectivity.",
            "The object contains {vars['color_num']} random colors.",
            "Each input grid contains a single-colored cell of the color {color('cell_color')}."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The object and the single-colored cell are identified.",
            "A copy of the object is placed so that the single colored cell is at the center of the surrounding square of the object and is covered by the object."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self,
                     taskvars: Dict[str, Any],
                     gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = gridvars['grid_size']
        n = taskvars['n']  # The n × n bounding box (always odd)
        cell_color = taskvars['cell_color']
        color_num = taskvars['color_num']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Generate available colors (excluding 0 and cell_color)
        available_colors = [c for c in range(1, 10) if c != cell_color]
        object_colors = random.sample(available_colors, min(color_num, len(available_colors)))
        
        # Create the object that will have a bounding box of n × n with 4-connectivity
        def create_valid_object():
            obj_array = create_object(
                height=n,
                width=n,
                color_palette=object_colors,
                contiguity=Contiguity.FOUR,  # Changed to FOUR for 4-connectivity
                background=0
            )
            
            # Check if the bounding box is actually n × n
            rows_with_color = np.any(obj_array != 0, axis=1)
            cols_with_color = np.any(obj_array != 0, axis=0)
            
            if not np.any(rows_with_color) or not np.any(cols_with_color):
                return None
                
            first_row = np.argmax(rows_with_color)
            last_row = len(rows_with_color) - 1 - np.argmax(rows_with_color[::-1])
            first_col = np.argmax(cols_with_color)
            last_col = len(cols_with_color) - 1 - np.argmax(cols_with_color[::-1])
            
            actual_height = last_row - first_row + 1
            actual_width = last_col - first_col + 1
            
            # We want the bounding box to be exactly n × n
            # This means both height and width should be n
            if actual_height == n and actual_width == n:
                # Check if object actually uses the required number of colors
                unique_colors = len(np.unique(obj_array[obj_array != 0]))
                if unique_colors == color_num:
                    return obj_array
            
            return None
        
        # Ensure object has the right bounding box, right number of colors, and at least a few cells
        obj_array = retry(
            create_valid_object,
            lambda x: x is not None and np.sum(x != 0) >= color_num + 1,
            max_attempts=500
        )
        
        # Place the object in grid with some margin
        max_offset = grid_size - n
        if max_offset < 1:
            obj_r, obj_c = 0, 0
        else:
            # Leave some space for the marker
            margin = min(n, max_offset // 3) if max_offset >= 3 else 0
            obj_r = random.randint(margin, max(margin, max_offset - margin))
            obj_c = random.randint(margin, max(margin, max_offset - margin))
        
        # Paste object into grid
        grid[obj_r:obj_r+n, obj_c:obj_c+n] = obj_array
        
        # Calculate the center of the object's bounding box
        center_r = obj_r + n // 2
        center_c = obj_c + n // 2
        
        # Place the marker cell at a valid distance from the object
        # The marker should be placed such that when a copy of the object is centered on it,
        # the copy doesn't overlap or touch the original object
        
        def place_marker():
            # Try to place marker at random location
            marker_r = random.randint(0, grid_size - 1)
            marker_c = random.randint(0, grid_size - 1)
            
            # Check if location is empty
            if grid[marker_r, marker_c] != 0:
                return None
            
            # Check if placing bbox centered on marker would stay in bounds
            marker_bbox_r = marker_r - n // 2
            marker_bbox_c = marker_c - n // 2
            if (marker_bbox_r < 0 or marker_bbox_r + n > grid_size or
                marker_bbox_c < 0 or marker_bbox_c + n > grid_size):
                return None
            
            # Check that the new bbox regions don't touch or overlap
            # We need at least 1 cell gap between the two bounding boxes
            new_bbox_r_start = marker_bbox_r
            new_bbox_r_end = marker_bbox_r + n - 1
            new_bbox_c_start = marker_bbox_c
            new_bbox_c_end = marker_bbox_c + n - 1
            
            orig_bbox_r_start = obj_r
            orig_bbox_r_end = obj_r + n - 1
            orig_bbox_c_start = obj_c
            orig_bbox_c_end = obj_c + n - 1
            
            # Check if bounding boxes are separated by at least 1 cell
            # They don't touch if:
            # - One is completely to the left of the other (with gap), OR
            # - One is completely above the other (with gap)
            horizontal_gap = (new_bbox_c_end < orig_bbox_c_start - 1) or (new_bbox_c_start > orig_bbox_c_end + 1)
            vertical_gap = (new_bbox_r_end < orig_bbox_r_start - 1) or (new_bbox_r_start > orig_bbox_r_end + 1)
            
            # Bounding boxes must not touch or overlap
            if not (horizontal_gap or vertical_gap):
                return None
            
            return (marker_r, marker_c)
        
        marker_pos = retry(
            place_marker,
            lambda x: x is not None,
            max_attempts=300
        )
        
        marker_r, marker_c = marker_pos
        grid[marker_r, marker_c] = cell_color
        
        return grid
    
    def transform_input(self,
                        grid: np.ndarray,
                        taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        cell_color = taskvars['cell_color']
        n = taskvars['n']
        
        # Find the marker cell
        marker_positions = np.where(grid == cell_color)
        marker_r, marker_c = marker_positions[0][0], marker_positions[1][0]
        
        # Find the main object (largest connected component excluding marker)
        # Using 4-connectivity (diagonal_connectivity=False)
        temp_grid = grid.copy()
        temp_grid[marker_r, marker_c] = 0  # Temporarily remove marker
        
        objects = find_connected_objects(
            temp_grid,
            diagonal_connectivity=False,  # Changed to False for 4-connectivity
            background=0,
            monochromatic=False
        )
        
        # Get the largest object (the main object)
        if len(objects) > 0:
            main_object = max(objects, key=lambda obj: len(obj))
            
            # Get the object's bounding box
            bbox = main_object.bounding_box
            bbox_r_start = bbox[0].start
            bbox_c_start = bbox[1].start
            bbox_height = bbox[0].stop - bbox[0].start
            bbox_width = bbox[1].stop - bbox[1].start
            
            # The bounding box should be n × n where n is odd
            bbox_size = max(bbox_height, bbox_width)
            
            # Extract the bounding box region containing the object
            bbox_array = temp_grid[bbox_r_start:bbox_r_start+bbox_size, 
                                   bbox_c_start:bbox_c_start+bbox_size].copy()
            
            # Place copy of the object (with its bounding box) centered on marker
            center_offset = bbox_size // 2
            start_r = marker_r - center_offset
            start_c = marker_c - center_offset
            
            # Paste the object copy
            output[start_r:start_r+bbox_size, start_c:start_c+bbox_size] = bbox_array
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        cell_color = random.randint(1, 9)
        color_num = random.randint(2, 4)  # Greater than 1, typically 2-4 colors
        n = random.choice([3, 5, 7])  # Odd bounding box size (limited to ensure grid fits)
        
        taskvars = {
            'cell_color': cell_color,
            'color_num': color_num,
            'n': n
        }
        
        # Create train and test examples
        num_train = random.randint(3, 6)
        
        train_pairs = []
        for _ in range(num_train):
            # Vary grid size based on n
            # We need enough space for two n×n bboxes plus gaps
            min_grid_size = n * 3 + 2
            max_grid_size = 30  # ARC-AGI max
            grid_size = random.randint(min_grid_size, max_grid_size)
            
            gridvars = {
                'grid_size': grid_size
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_pairs.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        min_grid_size = n * 3 + 2
        max_grid_size = 30
        grid_size = random.randint(min_grid_size, max_grid_size)
        
        gridvars = {
            'grid_size': grid_size
        }
        
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_pairs = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_pairs,
            'test': test_pairs
        }


