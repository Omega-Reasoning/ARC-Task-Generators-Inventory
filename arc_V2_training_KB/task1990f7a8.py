from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject
from input_library import create_object, enforce_object_width, enforce_object_height, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Task1990f7a8Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains exactly 4 {color('color1')} objects, made of 8-way connected cells, with the remaining cells being empty.",
            "Each object is shaped and sized so that it can fit into a 3×3 subgrid. Each row and column of the subgrid must contain at least one cell.",
            "All objects must be completely separated from each other.",
            "The 4 objects are placed around the 4 corners of the grid (top-left, top-right, bottom-left, bottom-right)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size 7×7.",
            "They are constructed by identifying the 4 {color('color1')} objects in the input grid and their respective positions.",
            "Each object is placed in the output grid according to its position in the input grid.",
            "The object located in the top-left corner of the input grid is placed in the top-left 3×3 subgrid of the output grid, and so on for the other three corners.",
            "This placement leaves the middle row and middle column of the output grid empty, with the 4 objects located in the corners."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        color1 = taskvars['color1']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Divide the grid into 4 quadrants, each object goes in one quadrant
        # But with random positioning within each quadrant (not exactly at corners)
        mid_r = grid_size // 2
        mid_c = grid_size // 2
        
        # Define the quadrant boundaries with margins from edges and center
        margin = 3  # minimum distance from edges and center
        
        # Define quadrant ranges where objects can be placed
        quadrants = [
            # top-left quadrant: rows [margin, mid_r-margin], cols [margin, mid_c-margin]
            (margin, mid_r - margin - 3, margin, mid_c - margin - 3),
            # top-right quadrant: rows [margin, mid_r-margin], cols [mid_c+margin, grid_size-margin]  
            (margin, mid_r - margin - 3, mid_c + margin, grid_size - margin - 3),
            # bottom-left quadrant: rows [mid_r+margin, grid_size-margin], cols [margin, mid_c-margin]
            (mid_r + margin, grid_size - margin - 3, margin, mid_c - margin - 3),
            # bottom-right quadrant: rows [mid_r+margin, grid_size-margin], cols [mid_c+margin, grid_size-margin]
            (mid_r + margin, grid_size - margin - 3, mid_c + margin, grid_size - margin - 3)
        ]
        
        # Create objects in each quadrant with random positioning
        for r_min, r_max, c_min, c_max in quadrants:
            # Randomly place the 3x3 object within the quadrant bounds
            if r_max > r_min and c_max > c_min:
                r_start = random.randint(r_min, r_max)
                c_start = random.randint(c_min, c_max)
            else:
                # Fallback if quadrant is too small
                r_start = max(margin, min(r_min, grid_size - margin - 3))
                c_start = max(margin, min(c_min, grid_size - margin - 3))
            
            # Generate an object that fits in 3x3 and has full width/height coverage
            obj = enforce_object_height(
                lambda: enforce_object_width(
                    lambda: create_object(3, 3, color1, Contiguity.EIGHT, background=0)
                )
            )
            
            # Place the object in the grid
            grid[r_start:r_start+3, c_start:c_start+3] = obj
            
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        color1 = taskvars['color1']
        
        # Create 7x7 output grid
        output = np.zeros((7, 7), dtype=int)
        
        # Find all colored objects
        objects = find_connected_objects(grid, diagonal_connectivity=True, background=0, monochromatic=True)
        colored_objects = objects.with_color(color1)
        
        # Sort objects by position to identify corners
        grid_height, grid_width = grid.shape
        
        corner_objects = []
        for obj in colored_objects:
            bbox = obj.bounding_box
            center_r = (bbox[0].start + bbox[0].stop - 1) / 2
            center_c = (bbox[1].start + bbox[1].stop - 1) / 2
            
            # Determine which corner this object belongs to
            if center_r < grid_height / 2 and center_c < grid_width / 2:
                corner = "top-left"
                target_slice = (slice(0, 3), slice(0, 3))
            elif center_r < grid_height / 2 and center_c >= grid_width / 2:
                corner = "top-right"
                target_slice = (slice(0, 3), slice(4, 7))
            elif center_r >= grid_height / 2 and center_c < grid_width / 2:
                corner = "bottom-left"
                target_slice = (slice(4, 7), slice(0, 3))
            else:
                corner = "bottom-right"
                target_slice = (slice(4, 7), slice(4, 7))
            
            corner_objects.append((obj, target_slice))
        
        # Place each object in its corresponding 3x3 subgrid in output
        for obj, target_slice in corner_objects:
            # Extract the object as a 3x3 array
            obj_array = obj.to_array()
            
            # Ensure it fits in 3x3 (pad or crop if necessary)
            if obj_array.shape != (3, 3):
                temp = np.zeros((3, 3), dtype=int)
                min_r = min(3, obj_array.shape[0])
                min_c = min(3, obj_array.shape[1])
                temp[:min_r, :min_c] = obj_array[:min_r, :min_c]
                obj_array = temp
            
            # Place in output grid
            output[target_slice] = obj_array
            
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'grid_size': random.randint(15, 30),
            'color1': random.randint(1, 9)
        }
        
        # Generate train examples (3-5 examples)
        num_train = random.randint(3, 5)
        
        # Generate examples
        train_examples = []
        for _ in range(num_train):
            # Generate new grid size for variety
            train_taskvars = {
                'grid_size': random.randint(15, 30),
                'color1': taskvars['color1']  # Keep same color
            }
            
            input_grid = self.create_input(train_taskvars, {})
            output_grid = self.transform_input(input_grid, train_taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_taskvars = {
            'grid_size': random.randint(15, 30),
            'color1': taskvars['color1']
        }
        
        test_input = self.create_input(test_taskvars, {})
        test_output = self.transform_input(test_input, test_taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data


