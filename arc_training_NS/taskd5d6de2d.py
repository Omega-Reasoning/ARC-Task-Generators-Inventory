from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from scipy.ndimage import binary_fill_holes

class Taskd5d6de2d(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each grid contains 2 colored rectangle shapes, composed of colored cells of color {color('border_color')}.",
            "The border of each rectangular shape fully encloses zero or more internal cells, all of which are empty (0).",
            "All the remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All cells of color {color('border_color')} are identified.",
            "All empty (0) internal cells enclosed by the colored border are filled with color {color('internal_color')}.If no internal cells exist, no filling occurs for that rectangle.",
            "All cells of color {color('border_color')} are transformed to empty(0)."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'border_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'internal_color': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        # Ensure border and internal colors are different
        while taskvars['internal_color'] == taskvars['border_color']:
            taskvars['internal_color'] = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # Create train examples (3-6)
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        def generate_grid_with_rectangles():
            # Random grid size between 8 and 30 to ensure space for rectangles
            height = random.randint(8, 30)
            width = random.randint(8, 30)
            grid = np.zeros((height, width), dtype=int)
            
            border_color = taskvars['border_color']
            
            # Create 2 rectangle borders
            rectangles_created = 0
            attempts = 0
            max_attempts = 100
            
            while rectangles_created < 2 and attempts < max_attempts:
                attempts += 1
                
                # Random rectangle dimensions (minimum 3x3 to ensure internal space)
                # Make sure we have valid ranges
                max_rect_height = min(height//3, 8)  # Use //3 to leave space for 2 rectangles
                max_rect_width = min(width//3, 8)
                
                if max_rect_height < 3 or max_rect_width < 3:
                    # If grid is too small, skip this attempt
                    continue
                
                rect_height = random.randint(3, max_rect_height)
                rect_width = random.randint(3, max_rect_width)
                
                # Random position
                max_row = height - rect_height
                max_col = width - rect_width
                
                if max_row < 0 or max_col < 0:
                    continue
                    
                start_row = random.randint(0, max_row)
                start_col = random.randint(0, max_col)
                
                # Check if this rectangle would overlap with existing content
                rect_area = grid[start_row:start_row+rect_height, start_col:start_col+rect_width]
                if np.any(rect_area != 0):
                    continue
                
                # Create rectangle border
                # Top and bottom borders
                grid[start_row, start_col:start_col+rect_width] = border_color
                grid[start_row+rect_height-1, start_col:start_col+rect_width] = border_color
                
                # Left and right borders
                grid[start_row:start_row+rect_height, start_col] = border_color
                grid[start_row:start_row+rect_height, start_col+rect_width-1] = border_color
                
                rectangles_created += 1
            
            return grid
        
        # Use retry to ensure we get a valid grid with fillable internal cells
        def has_fillable_cells(grid):
            # Check if there are internal cells that can be filled
            border_color = taskvars['border_color']
            
            # Find rectangles and check for internal cells
            border_mask = (grid == border_color)
            
            # Use flood fill to identify enclosed areas
            filled_mask = binary_fill_holes(border_mask)
            
            # Internal cells are those that are filled but weren't originally border cells
            internal_mask = filled_mask & ~border_mask & (grid == 0)
            
            # Return True if there are any fillable internal cells
            return np.any(internal_mask)
        
        return retry(generate_grid_with_rectangles, has_fillable_cells, max_attempts=50)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        border_color = taskvars['border_color']
        internal_color = taskvars['internal_color']
        
        output_grid = grid.copy()
        
        # Create mask for border cells
        border_mask = (grid == border_color)
        
        # Fill internal cells using binary_fill_holes
        filled_mask = binary_fill_holes(border_mask)
        
        # Internal cells are those that are filled but weren't originally border cells
        internal_mask = filled_mask & ~border_mask & (grid == 0)
        
        # Fill internal cells with internal color
        output_grid[internal_mask] = internal_color
        
        # Remove border cells (set to 0)
        output_grid[border_mask] = 0
        
        return output_grid

