from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, Contiguity
import numpy as np
import random
from typing import Dict, Any, Tuple, List, Optional

class Task15660dd6Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids vary in size, but the dimensions are determined by the structure and placement of the frames rather than chosen randomly.",
            "Each input grid contains n square {color('object_color')} frames, all of the same size (either 5×5 or 6×6) within a single grid. The frames are aligned vertically and horizontally, with exactly one row and one column of spacing between them.",
            "The first frame always starts at position (1, 1). No frame touches the grid border, and all background cells are colored {color('background_color')}.",
            "The first row, last row, and last column are completely filled with the {color('background_color')}. All background cells, including the spaces between frames, are colored {color('background_color')}.",
            "The n square {color('object_color')} frames are distributed across groups of rows and columns. Each group of columns that contains frames must include exactly one frame whose interior is completely filled with a single color, different from both the {color('background_color')} and the frame {color('object_color')}.",
            "The first column of the grid contains vertically stacked colored strips, located exactly to the left of each frame. Each strip is separated by exactly one {color('background_color')} cell. The colors of the strips are assigned in order: the first is {color('strip_color1')}, the second is {color('strip_color2')}, and the third is {color('strip_color3')}.",
            "The objects are made of 8-way connected cells colored with {color('object_color2')}. All objects placed within the same set of columns must have exactly the same shape, except for one, which is completely filled with a single color different from the others.",
            "All remaining cells within these frames, other than the {color('object_color2')} colored object cells, are filled with the {color('background_color')} color.",
            "Each input grid contains either 3 or 4 frames arranged in rows, and this number must be consistent within the same grid. The number of frames per column group is always exactly 3.",
            "The total grid size depends on the chosen frame size: If frames are 5×5, valid grid sizes include 19×19 (3×3 frames) and 19×25 (4×3 frames). If frames are 6×6, valid grid sizes include 22×22 (3×3 frames) and 22×29 (4×3 frames)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by first identifying all square {color('object_color')} frames in the input grid.",
            "These frames are grouped by their column alignment (i.e., vertical strips of frames with the same x-position).",
            "Count the number of groups (i.e., sets of vertically aligned frames), and place exactly one frame for each group in the output grid. Each output frame should be separated by exactly one column filled with {color('background_color')} color.",
            "For each group: the shape within each frame is the same, with exactly one frame having its entire interior filled. Identify the unique {color('object_color2')} object shape for the group and place it in the corresponding position in the output grid.",
            "Within each group, find the frame whose entire interior (excluding the border) is filled with a single color that is different from both {color('background_color')}, {color('object_color')} and {color('object_color2')}. Use this color to fill the copied object of the same shape in the output.",
            "Then, identify the colored vertical strips in the first column of the input grid. Match each strip to the row group that contains the fully filled frame with the corresponding interior color.",
            "The output object—colored based on its interior—is framed using the strip color associated with its row group."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate shared task variables (colors only)
        taskvars = {}
        
        # Choose base colors that are consistent across all grids
        all_colors = list(range(1, 10))
        random.shuffle(all_colors)
        
        taskvars['background_color'] = all_colors[0]
        taskvars['object_color'] = all_colors[1]      # Frame color
        taskvars['object_color2'] = all_colors[2]     # Object shape color
        taskvars['strip_color1'] = all_colors[3]
        taskvars['strip_color2'] = all_colors[4]
        taskvars['strip_color3'] = 0  # Use 0 for the third strip color
        
        # Generate train and test examples with varying frame sizes and configurations
        train_data = []
        for _ in range(3):
            # Each grid gets its own frame size and configuration
            gridvars = self._generate_grid_config(taskvars)
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Test grid also gets its own configuration
        test_gridvars = self._generate_grid_config(taskvars)
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_data = [{'input': test_input, 'output': test_output}]
        
        # Return only the shared task variables
        return taskvars, {'train': train_data, 'test': test_data}
    
    def _generate_grid_config(self, taskvars: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration for a single grid (frame size, layout, and fill colors)."""
        gridvars = {}
        
        # Choose frame size for this specific grid
        frame_size = random.choice([5, 6])
        gridvars['frame_size'] = frame_size
        
        # Choose grid configuration based on frame size
        if frame_size == 5:
            grid_configs = [(19, 19), (19, 25)]  # 3x3 or 3x4 frames
        else:
            grid_configs = [(22, 22), (22, 29)]  # 3x3 or 3x4 frames
        
        grid_height, grid_width = random.choice(grid_configs)
        gridvars['grid_height'] = grid_height
        gridvars['grid_width'] = grid_width
        
        # Calculate number of column groups
        if grid_width == grid_height:  # Square grid = 3 columns
            gridvars['num_col_groups'] = 3
        else:  # Rectangular = 4 columns
            gridvars['num_col_groups'] = 4
        
        # Generate unique fill colors for this grid
        used_colors = {taskvars['background_color'], taskvars['object_color'], 
                      taskvars['object_color2'], taskvars['strip_color1'], 
                      taskvars['strip_color2']}  # Note: strip_color3 is 0, so not in 1-9 range
        
        available_colors = [c for c in range(1, 10) if c not in used_colors]
        
        # Select random fill colors for this grid
        gridvars['fill_colors'] = random.sample(available_colors, gridvars['num_col_groups'])
            
        return gridvars
    
    def _infer_frame_size(self, grid: np.ndarray, object_color: int) -> int:
        """Infer frame size by finding the first frame."""
        for size in [5, 6]:
            try:
                frame_region = grid[1:1+size, 1:1+size]
                # Check if it's a proper frame (border is object_color)
                if (np.all(frame_region[0, :] == object_color) and  # top
                    np.all(frame_region[-1, :] == object_color) and  # bottom
                    np.all(frame_region[:, 0] == object_color) and  # left
                    np.all(frame_region[:, -1] == object_color)):  # right
                    return size
            except IndexError:
                continue
        raise ValueError("Could not infer frame size from grid")
    
    def _infer_num_col_groups(self, grid: np.ndarray, frame_size: int) -> int:
        """Calculate number of column groups based on grid width."""
        grid_height, grid_width = grid.shape
        return (grid_width - 1) // (frame_size + 1)
    
    def _find_fill_colors(self, grid: np.ndarray, frame_size: int, num_col_groups: int, 
                         background_color: int, object_color: int, object_color2: int) -> List[int]:
        """Find all unique fill colors by examining frame interiors."""
        fill_colors = []
        used_base_colors = {background_color, object_color, object_color2}
        
        for col in range(num_col_groups):
            for row in range(3):
                frame_row = 1 + row * (frame_size + 1)
                frame_col = 1 + col * (frame_size + 1)
                
                # Extract frame interior
                interior = grid[frame_row+1:frame_row+frame_size-1, frame_col+1:frame_col+frame_size-1]
                unique_colors = np.unique(interior)
                
                # If it's a single color that's not a base color, it's a fill color
                if (len(unique_colors) == 1 and 
                    unique_colors[0] not in used_base_colors and
                    unique_colors[0] not in fill_colors):
                    fill_colors.append(unique_colors[0])
        
        return fill_colors
    
    def _infer_grid_config(self, grid: np.ndarray, background_color: int, object_color: int, object_color2: int) -> Dict[str, Any]:
        """Infer grid configuration from the input grid."""
        frame_size = self._infer_frame_size(grid, object_color)
        num_col_groups = self._infer_num_col_groups(grid, frame_size)
        fill_colors = self._find_fill_colors(grid, frame_size, num_col_groups, background_color, object_color, object_color2)
        
        return {
            'frame_size': frame_size,
            'num_col_groups': num_col_groups,
            'fill_colors': fill_colors
        }
    
    def _extract_frame_info(self, grid: np.ndarray, col: int, frame_size: int, 
                           fill_colors: List[int], object_color2: int) -> Tuple[Optional[int], Optional[np.ndarray], Optional[int]]:
        """Extract the filled frame color, object shape, and filled frame row for a column group."""
        filled_frame_color = None
        object_shape = None
        filled_frame_row = None
        
        # Check each frame in this column group
        for row in range(3):
            frame_row = 1 + row * (frame_size + 1)
            frame_col = 1 + col * (frame_size + 1)
            
            # Extract frame interior
            interior = grid[frame_row+1:frame_row+frame_size-1, frame_col+1:frame_col+frame_size-1]
            
            # Check if this is a filled frame (single color, not background or object_color2)
            unique_colors = np.unique(interior)
            if len(unique_colors) == 1 and unique_colors[0] in fill_colors:
                filled_frame_color = unique_colors[0]
                filled_frame_row = row
            elif np.any(interior == object_color2):
                # This frame contains the object shape
                object_shape = interior.copy()
        
        return filled_frame_color, object_shape, filled_frame_row
    
    def _draw_output_frame_border(self, output_grid: np.ndarray, output_col: int, frame_size: int, frame_strip_color: int):
        """Draw the border of an output frame."""
        for i in range(frame_size):
            for j in range(frame_size):
                if i == 0 or i == frame_size-1 or j == 0 or j == frame_size-1:
                    output_grid[i, output_col + j] = frame_strip_color
    
    def _fill_output_frame_interior(self, output_grid: np.ndarray, output_col: int, frame_size: int,
                                   object_shape: np.ndarray, object_color2: int, 
                                   filled_frame_color: int, background_color: int):
        """Fill the interior of an output frame with the object shape."""
        for i in range(1, frame_size-1):
            for j in range(1, frame_size-1):
                if object_shape[i-1, j-1] == object_color2:
                    output_grid[i, output_col + j] = filled_frame_color
                else:
                    output_grid[i, output_col + j] = background_color
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Get grid-specific configuration
        grid_height = gridvars['grid_height']
        grid_width = gridvars['grid_width']
        frame_size = gridvars['frame_size']
        num_col_groups = gridvars['num_col_groups']
        fill_colors = gridvars['fill_colors']
        
        # Get shared colors
        background_color = taskvars['background_color']
        object_color = taskvars['object_color']
        object_color2 = taskvars['object_color2']
        strip_color1 = taskvars['strip_color1']
        strip_color2 = taskvars['strip_color2']
        strip_color3 = taskvars['strip_color3']
        
        # Initialize grid with background
        grid = np.full((grid_height, grid_width), background_color, dtype=int)
        
        # Create object shape for each column group
        object_shapes = {}
        for col in range(num_col_groups):
            # Generate a random object shape using a temporary background
            temp_bg = -1  # Use -1 as temp background to avoid conflicts
            shape = retry(
                lambda: create_object(frame_size-2, frame_size-2, object_color2, 
                                    Contiguity.EIGHT, temp_bg),
                lambda x: np.sum(x == object_color2) >= 2 and np.sum(x == object_color2) <= (frame_size-2)**2//2
            )
            # Convert temp background to actual background color
            shape[shape == temp_bg] = background_color
            object_shapes[col] = shape
        
        # Place frames and objects
        filled_frame_per_col = {}  # Track which row has the filled frame for each column
        for col in range(num_col_groups):
            # Choose one frame per column group to be completely filled
            filled_row = random.choice([0, 1, 2])
            filled_frame_per_col[col] = filled_row
            
            for row in range(3):
                frame_row = 1 + row * (frame_size + 1)
                frame_col = 1 + col * (frame_size + 1)
                
                # Draw frame border
                for i in range(frame_size):
                    for j in range(frame_size):
                        if i == 0 or i == frame_size-1 or j == 0 or j == frame_size-1:
                            grid[frame_row + i, frame_col + j] = object_color
                
                # Fill frame interior
                if row == filled_row:
                    # This frame gets completely filled with fill color
                    for i in range(1, frame_size-1):
                        for j in range(1, frame_size-1):
                            grid[frame_row + i, frame_col + j] = fill_colors[col]
                else:
                    # This frame gets the object shape
                    shape = object_shapes[col]
                    for i in range(1, frame_size-1):
                        for j in range(1, frame_size-1):
                            grid[frame_row + i, frame_col + j] = shape[i-1, j-1]
        
        # Place colored strips in first column with 3 distinct colors
        strip_colors = [strip_color1, strip_color2, strip_color3]
        for row in range(3):
            strip_row = 1 + row * (frame_size + 1)
            strip_color = strip_colors[row]  # Use exact order: row 0->color1, row 1->color2, row 2->color3
            # Place strip with height equal to frame size
            for i in range(frame_size):
                grid[strip_row + i, 0] = strip_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Get shared colors
        background_color = taskvars['background_color']
        object_color = taskvars['object_color']
        object_color2 = taskvars['object_color2']
        strip_color1 = taskvars['strip_color1']
        strip_color2 = taskvars['strip_color2']
        strip_color3 = taskvars['strip_color3']
        
        # Infer grid configuration from the input grid
        config = self._infer_grid_config(grid, background_color, object_color, object_color2)
        frame_size = config['frame_size']
        num_col_groups = config['num_col_groups']
        fill_colors = config['fill_colors']
        
        strip_colors = [strip_color1, strip_color2, strip_color3]
        
        # Create output grid
        output_width = num_col_groups * (frame_size + 1) - 1
        output_height = frame_size
        output_grid = np.full((output_height, output_width), background_color, dtype=int)
        
        # Process each column group
        for col in range(num_col_groups):
            # Extract information from this column group
            filled_frame_color, object_shape, filled_frame_row = self._extract_frame_info(
                grid, col, frame_size, fill_colors, object_color2)
            
            # Calculate output column position
            output_col = col * (frame_size + 1)
            
            # Draw frame border using the strip color corresponding to the filled frame's row
            frame_strip_color = strip_colors[filled_frame_row]
            self._draw_output_frame_border(output_grid, output_col, frame_size, frame_strip_color)
            
            # Fill interior with object shape using filled frame color
            if object_shape is not None and filled_frame_color is not None:
                self._fill_output_frame_interior(output_grid, output_col, frame_size, 
                                               object_shape, object_color2, filled_frame_color, background_color)
        
        return output_grid
