from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class taskh9woyuLPvSxM3pCCxQZKZCGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains a {color('frame_colour')} square frame with a size of at least 3×3 or larger.",
            "The exterior of the frame is completely empty (0).",
            "The interior is either completely filled with the {color('interior_colour')} color or contains a checkerboard pattern, where empty cells and {color('interior_colour')} cells alternate across rows and columns.",
            "The size of the frame varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the {color('frame_colour')} frame and its interior.",
            "If the interior is completely filled with {color('interior_colour')} cells, then change all interior {color('interior_colour')} cells to {color('new_interior_colour')}.",
            "Otherwise, leave the interior unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and generate train/test data."""
        
        # Generate random colors (all different)
        colors = random.sample(range(1, 10), 3)
        frame_colour, interior_colour, new_interior_colour = colors
        
        # Random grid size between 5 and 30
        grid_size = random.randint(5, 30)
        
        taskvars = {
            'grid_size': grid_size,
            'frame_colour': frame_colour,
            'interior_colour': interior_colour,
            'new_interior_colour': new_interior_colour
        }
        
        # Generate 4-6 training examples + 1 test (to ensure at least 2 of each type)
        num_train = random.randint(4, 5)
        
        train_examples = []
        test_examples = []
        
        # Track used frame sizes to ensure all are different
        used_frame_sizes = set()
        
        # Count how many of each type we have
        filled_count = 0
        checkerboard_count = 0
        
        # Generate training examples ensuring at least 2 of each type
        for i in range(num_train):
            # Determine fill type based on requirements
            if filled_count < 2:
                fill_type = 'filled'
            elif checkerboard_count < 2:
                fill_type = 'checkerboard'
            else:
                fill_type = random.choice(['filled', 'checkerboard'])
                
            if fill_type == 'filled':
                filled_count += 1
            else:
                checkerboard_count += 1
                
            gridvars = {'fill_type': fill_type, 'used_frame_sizes': used_frame_sizes}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example with unique frame size
        test_fill_type = random.choice(['filled', 'checkerboard'])
        test_gridvars = {'fill_type': test_fill_type, 'used_frame_sizes': used_frame_sizes}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples.append({
            'input': test_input,
            'output': test_output
        })
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid with a frame and interior pattern."""
        
        grid_size = taskvars['grid_size']
        frame_colour = taskvars['frame_colour']
        interior_colour = taskvars['interior_colour']
        fill_type = gridvars['fill_type']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Determine frame size (at least 3x3, but leave some margin)
        min_frame_size = 4
        max_frame_size = min(grid_size - 2, 15)  # Leave at least 1 cell margin
        frame_size = random.randint(min_frame_size, max_frame_size)
        
        # Position frame randomly but ensure it fits completely
        max_start_row = grid_size - frame_size
        max_start_col = grid_size - frame_size
        start_row = random.randint(0, max_start_row)
        start_col = random.randint(0, max_start_col)
        
        # Draw frame border
        end_row = start_row + frame_size
        end_col = start_col + frame_size
        
        # Top and bottom borders
        grid[start_row, start_col:end_col] = frame_colour
        grid[end_row-1, start_col:end_col] = frame_colour
        
        # Left and right borders  
        grid[start_row:end_row, start_col] = frame_colour
        grid[start_row:end_row, end_col-1] = frame_colour
        
        # Fill interior based on fill_type
        interior_start_row = start_row + 1
        interior_end_row = end_row - 1
        interior_start_col = start_col + 1
        interior_end_col = end_col - 1
        
        if fill_type == 'filled':
            # Completely fill interior with interior_colour
            grid[interior_start_row:interior_end_row, 
                 interior_start_col:interior_end_col] = interior_colour
        elif fill_type == 'checkerboard':
            # Create checkerboard pattern
            for r in range(interior_start_row, interior_end_row):
                for c in range(interior_start_col, interior_end_col):
                    # Checkerboard pattern: (r + c) % 2 determines if cell should be filled
                    if (r + c) % 2 == 0:
                        grid[r, c] = interior_colour
                    # else: leave as 0 (empty)
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input grid according to the rules."""
        
        frame_colour = taskvars['frame_colour']
        interior_colour = taskvars['interior_colour']
        new_interior_colour = taskvars['new_interior_colour']
        
        # Copy the input grid
        output = grid.copy()
        
        # Find the frame
        frame_cells = np.where(grid == frame_colour)
        if len(frame_cells[0]) == 0:
            return output  # No frame found
            
        # Find bounding box of frame
        min_row, max_row = frame_cells[0].min(), frame_cells[0].max()
        min_col, max_col = frame_cells[1].min(), frame_cells[1].max()
        
        # Interior is inside the frame
        interior_start_row = min_row + 1
        interior_end_row = max_row
        interior_start_col = min_col + 1
        interior_end_col = max_col
        
        # Extract interior region
        if (interior_start_row < interior_end_row and 
            interior_start_col < interior_end_col):
            
            interior_region = grid[interior_start_row:interior_end_row, 
                                 interior_start_col:interior_end_col]
            
            # Check if interior is completely filled with interior_colour
            interior_cells = interior_region.flatten()
            non_zero_cells = interior_cells[interior_cells != 0]
            
            # If all non-zero cells are interior_colour and there are no empty cells
            if (len(non_zero_cells) == len(interior_cells) and 
                np.all(non_zero_cells == interior_colour)):
                
                # Interior is completely filled - change color
                output[interior_start_row:interior_end_row, 
                       interior_start_col:interior_end_col] = new_interior_colour
                       
            # Otherwise (checkerboard pattern), leave unchanged
        
        return output


