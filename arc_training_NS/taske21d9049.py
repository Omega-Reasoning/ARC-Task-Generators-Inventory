from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import create_object, retry, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taske21d9049(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "A single cell, located at least two cells away from all grid boundaries, is designated as the center cell. This cell is colored with {color('center_color')}, and its coordinates are denoted as (i, j).",
            "One of the following four directional sets is randomly selected: {{(i+1,j), (i,j+1), (i+2,j), (i,j+2)}} , {{(i+1,j) , (i,j-1) , (i+2,j),  (i, j-2)}} , {{(i-1,j), (i,j+1) , (i-2, j) (i,j+2)}}, {{(i-1,j), (i, j-1) , (i-2,j), (i, j-2)}}.",
            "Among the selected four cells: The two adjacent cells to the center (those at distance 1) are colored with {color('color_1')}. The two further cells (those at distance 2) are colored with {color('color_2')}.",
            "In some grids, the reflected counterparts of the two adjacent cells (relative to the center) are also colored with {color('color_3')}. In other grids, these reflection cells are left uncolored."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All colored cells are identified to determine the presence of a cross-like pattern centered at (i, j): The vertical bar consists of all colored cells at positions (i + k, j) for any integer k. The horizontal bar consists of all colored cells at positions (i, j + k) for any integer k.",
            "Based on the color pattern observed on the vertical bar (column j), the same pattern is extended upward and downward along the entire column j in the output grid.",
            "Similarly, based on the color pattern observed on the horizontal bar (row i), the same pattern is extended left and right along the entire row i in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        # Create grid with random size
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        grid = np.zeros((height, width), dtype=int)
        
        # Find valid center position (at least 2 cells from boundary)
        center_row = random.randint(2, height - 3)
        center_col = random.randint(2, width - 3)
        
        # Color the center cell
        grid[center_row, center_col] = taskvars['center_color']
        
        # Define the four directional patterns
        patterns = [
            # Down-right: (i+1,j), (i,j+1), (i+2,j), (i,j+2)
            [(1, 0), (0, 1), (2, 0), (0, 2)],
            # Down-left: (i+1,j), (i,j-1), (i+2,j), (i,j-2)
            [(1, 0), (0, -1), (2, 0), (0, -2)],
            # Up-right: (i-1,j), (i,j+1), (i-2,j), (i,j+2)
            [(-1, 0), (0, 1), (-2, 0), (0, 2)],
            # Up-left: (i-1,j), (i,j-1), (i-2,j), (i,j-2)
            [(-1, 0), (0, -1), (-2, 0), (0, -2)]
        ]
        
        # Select one pattern randomly
        selected_pattern = random.choice(patterns)
        
        # Color the four cells in the selected pattern
        # First two are adjacent (distance 1), colored with color_1
        # Last two are further (distance 2), colored with color_2
        for i, (dr, dc) in enumerate(selected_pattern):
            r, c = center_row + dr, center_col + dc
            if 0 <= r < height and 0 <= c < width:
                if i < 2:  # Adjacent cells
                    grid[r, c] = taskvars['color_1']
                else:  # Further cells
                    grid[r, c] = taskvars['color_2']
        
        # Optionally add reflected cells
        if gridvars.get('add_reflected', False):
            # Add reflected counterparts of adjacent cells
            for i in range(2):  # Only the first two (adjacent) cells
                dr, dc = selected_pattern[i]
                # Reflect relative to center
                refl_r, refl_c = center_row - dr, center_col - dc
                if 0 <= refl_r < height and 0 <= refl_c < width:
                    grid[refl_r, refl_c] = taskvars['color_3']
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        height, width = grid.shape
        
        # Find the center cell (colored with center_color)
        center_positions = np.where(grid == taskvars['center_color'])
        if len(center_positions[0]) == 0:
            return output_grid
        
        center_row, center_col = center_positions[0][0], center_positions[1][0]
        
        # Extract vertical bar pattern (all colored cells in column center_col)
        vertical_pattern = []  # List of (offset, color) pairs
        for r in range(height):
            if grid[r, center_col] != 0:  # If cell is colored
                offset = r - center_row
                vertical_pattern.append((offset, grid[r, center_col]))
        
        # Extract horizontal bar pattern (all colored cells in row center_row)
        horizontal_pattern = []  # List of (offset, color) pairs
        for c in range(width):
            if grid[center_row, c] != 0:  # If cell is colored
                offset = c - center_col
                horizontal_pattern.append((offset, grid[center_row, c]))
        
        # Extend vertical pattern to fill entire column center_col
        if vertical_pattern:
            # Sort pattern by offset for easier processing
            vertical_pattern.sort()
            
            # Fill the entire column by repeating/extending the pattern
            for r in range(height):
                offset = r - center_row
                
                # Find the appropriate color for this offset
                color_to_use = None
                
                # First check if we have an exact match
                for pattern_offset, pattern_color in vertical_pattern:
                    if offset == pattern_offset:
                        color_to_use = pattern_color
                        break
                
                # If no exact match, extend the pattern
                if color_to_use is None and vertical_pattern:
                    # Create a repeating pattern based on the observed pattern
                    if len(vertical_pattern) == 1:
                        # Single colored cell - extend that color to entire column
                        color_to_use = vertical_pattern[0][1]
                    else:
                        # Multiple colored cells - create repeating pattern
                        offsets = [p[0] for p in vertical_pattern]
                        min_offset = min(offsets)
                        max_offset = max(offsets)
                        pattern_length = max_offset - min_offset + 1
                        
                        # Map current offset to pattern space
                        if offset < min_offset:
                            # Extend upward
                            pattern_index = (offset - min_offset) % pattern_length
                        elif offset > max_offset:
                            # Extend downward
                            pattern_index = (offset - min_offset) % pattern_length
                        else:
                            pattern_index = offset - min_offset
                        
                        # Find color at this pattern position
                        target_offset = min_offset + pattern_index
                        for pattern_offset, pattern_color in vertical_pattern:
                            if pattern_offset == target_offset:
                                color_to_use = pattern_color
                                break
                        
                        # If still no match, use the closest color
                        if color_to_use is None:
                            closest_pattern = min(vertical_pattern, key=lambda x: abs(x[0] - target_offset))
                            color_to_use = closest_pattern[1]
                
                if color_to_use is not None:
                    output_grid[r, center_col] = color_to_use
        
        # Extend horizontal pattern to fill entire row center_row
        if horizontal_pattern:
            # Sort pattern by offset for easier processing
            horizontal_pattern.sort()
            
            # Fill the entire row by repeating/extending the pattern
            for c in range(width):
                offset = c - center_col
                
                # Find the appropriate color for this offset
                color_to_use = None
                
                # First check if we have an exact match
                for pattern_offset, pattern_color in horizontal_pattern:
                    if offset == pattern_offset:
                        color_to_use = pattern_color
                        break
                
                # If no exact match, extend the pattern
                if color_to_use is None and horizontal_pattern:
                    # Create a repeating pattern based on the observed pattern
                    if len(horizontal_pattern) == 1:
                        # Single colored cell - extend that color to entire row
                        color_to_use = horizontal_pattern[0][1]
                    else:
                        # Multiple colored cells - create repeating pattern
                        offsets = [p[0] for p in horizontal_pattern]
                        min_offset = min(offsets)
                        max_offset = max(offsets)
                        pattern_length = max_offset - min_offset + 1
                        
                        # Map current offset to pattern space
                        if offset < min_offset:
                            # Extend leftward
                            pattern_index = (offset - min_offset) % pattern_length
                        elif offset > max_offset:
                            # Extend rightward
                            pattern_index = (offset - min_offset) % pattern_length
                        else:
                            pattern_index = offset - min_offset
                        
                        # Find color at this pattern position
                        target_offset = min_offset + pattern_index
                        for pattern_offset, pattern_color in horizontal_pattern:
                            if pattern_offset == target_offset:
                                color_to_use = pattern_color
                                break
                        
                        # If still no match, use the closest color
                        if color_to_use is None:
                            closest_pattern = min(horizontal_pattern, key=lambda x: abs(x[0] - target_offset))
                            color_to_use = closest_pattern[1]
                
                if color_to_use is not None:
                    output_grid[center_row, c] = color_to_use
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random colors ensuring they're all different
        all_colors = list(range(1, 10))  # Exclude 0 (background)
        selected_colors = random.sample(all_colors, 4)
        
        taskvars = {
            'center_color': selected_colors[0],
            'color_1': selected_colors[1],
            'color_2': selected_colors[2],
            'color_3': selected_colors[3]
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Randomly decide whether to add reflected cells
            gridvars = {'add_reflected': random.choice([True, False])}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_gridvars = {'add_reflected': random.choice([True, False])}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
