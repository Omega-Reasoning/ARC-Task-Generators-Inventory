from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity, retry
from Framework.transformation_library import find_connected_objects

class Task41e4d17eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain one or two {vars['frame_size']}x{vars['frame_size']} {color('object_color')} frames, each being one-cell wide.",
            "The {color('object_color')} square frames have a completely filled interior, with both the interior and exterior completely filled with {color('background_color')} cells.",
            "If there are two square frames in a grid, they must not be connected and should be placed so that they are diagonal to eachother."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid, identifying the exact middle cell of each {color('object_color')} frame, and coloring the entire row and column containing it with {color('line_color')} color, except for the frame cells.",
            "This forms a {color('line_color')} cross shape centered on each frame.",
            "If there are two {color('object_color')} frames in a single grid, apply this transformation to both."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Task variables that will be used in the templates
        grid_size = random.choice([9, 15, 21, 27])
        frame_size = grid_size // 3
        
        # Choose three distinct colors for objects, lines, and background
        colors = random.sample(range(1, 10), 3)
        object_color, line_color, background_color = colors
        
        taskvars = {
            'grid_size': grid_size,
            'frame_size': frame_size,
            'object_color': object_color,
            'line_color': line_color,
            'background_color': background_color
        }
        
        # Generate the training and test examples with specific configurations
        train_examples = []
        
        # Training example 1: two frames - top-left and bottom-right
        gridvars = {'frame_positions': ['top_left', 'bottom_right']}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid.copy(), taskvars)
        train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Training example 2: one frame
        gridvars = {'frame_positions': ['center']}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid.copy(), taskvars)
        train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Training example 3: two frames - top-right and bottom-left
        gridvars = {'frame_positions': ['top_right', 'bottom_left']}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid.copy(), taskvars)
        train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Test example: randomly choose configuration
        test_config = random.choice([
            ['top_left', 'bottom_right'],
            ['top_right', 'bottom_left'],
            ['center']
        ])
        gridvars = {'frame_positions': test_config}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid.copy(), taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': input_grid, 'output': output_grid}]
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        frame_size = taskvars['frame_size']
        object_color = taskvars['object_color']
        background_color = taskvars['background_color']
        
        # Initialize grid with background color
        grid = np.full((grid_size, grid_size), background_color, dtype=int)
        
        # Add frames based on positions
        frame_positions = gridvars['frame_positions']
        
        # Define the quadrant size
        quadrant_size = grid_size // 2
        
        # Place frames based on specified positions
        for position in frame_positions:
            if position == 'top_left':
                # Calculate safe bounds for placement in top-left quadrant
                max_start = max(0, quadrant_size - frame_size)
                start_r = random.randint(0, max(1, max_start))
                start_c = random.randint(0, max(1, max_start))
                
            elif position == 'top_right':
                # Calculate safe bounds for placement in top-right quadrant
                max_start = max(0, quadrant_size - frame_size)
                start_r = random.randint(0, max(1, max_start))
                start_c = random.randint(quadrant_size, grid_size - frame_size)
                
            elif position == 'bottom_left':
                # Calculate safe bounds for placement in bottom-left quadrant
                max_start = max(0, quadrant_size - frame_size)
                start_r = random.randint(quadrant_size, grid_size - frame_size)
                start_c = random.randint(0, max(1, max_start))
                
            elif position == 'bottom_right':
                # Calculate safe bounds for placement in bottom-right quadrant
                start_r = random.randint(quadrant_size, grid_size - frame_size)
                start_c = random.randint(quadrant_size, grid_size - frame_size)
                
            else:  # center position
                # Center the frame with small random offset
                center = grid_size // 2
                offset = min(2, max(0, (grid_size - frame_size) // 4))
                start_r = random.randint(center - offset - frame_size // 2, 
                                         center + offset - frame_size // 2)
                start_c = random.randint(center - offset - frame_size // 2, 
                                         center + offset - frame_size // 2)
                # Make sure we're within grid bounds
                start_r = max(0, min(start_r, grid_size - frame_size))
                start_c = max(0, min(start_c, grid_size - frame_size))
            
            # Draw the frame (one-cell wide border)
            for r in range(start_r, start_r + frame_size):
                for c in range(start_c, start_c + frame_size):
                    # Only color the border cells, leave the interior as background_color
                    if (r == start_r or r == start_r + frame_size - 1 or 
                        c == start_c or c == start_c + frame_size - 1):
                        grid[r, c] = object_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        object_color = taskvars['object_color']
        line_color = taskvars['line_color']
        
        # First make a copy of the original grid to preserve the frame locations
        original_grid = grid.copy()
        
        # Find all connected objects with the frame color
        objects = find_connected_objects(grid, diagonal_connectivity=True, 
                                        background=taskvars['background_color'], 
                                        monochromatic=False)
        
        # Filter for objects with the frame color
        frame_objects = objects.with_color(object_color)
        
        # Get the frames (which should be the largest objects with the object_color)
        frames = frame_objects.sort_by_size(reverse=True)
        
        # Process each frame
        for frame in frames:
            # Get the bounding box of the frame
            r_slice, c_slice = frame.bounding_box
            
            # Calculate the center of the frame
            center_r = (r_slice.start + r_slice.stop - 1) // 2
            center_c = (c_slice.start + c_slice.stop - 1) // 2
            
            # Draw the cross
            # Draw the horizontal line, but don't overwrite frame cells
            for c in range(grid.shape[1]):
                if original_grid[center_r, c] != object_color:
                    grid[center_r, c] = line_color
                    
            # Draw the vertical line, but don't overwrite frame cells
            for r in range(grid.shape[0]):
                if original_grid[r, center_c] != object_color:
                    grid[r, center_c] = line_color
        
        return grid