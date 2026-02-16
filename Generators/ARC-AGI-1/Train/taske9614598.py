from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taske9614598Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each input grid contains two single-colored cells of color {color('color_1')}.",
            "These two single-colored cells are located either in the same row or the same column, but never on the first or last row/column.",
            "There are at least five empty cells between the single-colored cells.",
            "The number of empty cells between the two single-colored cells is always an odd number."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "Two single-colored cells are identified.",
            "A cross shape with 5 cells is filled in the middle of these two single colored cells with color {color('cross_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        color_1 = taskvars['color_1']
        
        def generate_valid_grid():
            grid = np.zeros((n, n), dtype=int)
            
            # Randomly choose horizontal or vertical alignment
            is_horizontal = random.choice([True, False])
            
            if is_horizontal:
                # Same row, different columns
                # Choose row (not first or last)
                row = random.randint(1, n - 2)
                
                # Choose two columns with proper spacing
                # Need at least 5 empty cells between, so minimum distance is 6
                # And odd number of cells between, so distance must be even (6, 8, 10, ...)
                min_distance = 6  # 5 empty cells + 1
                max_distance = min(n - 2, 20)  # Reasonable upper bound
                
                # Ensure even distance for odd number of empty cells
                valid_distances = [d for d in range(min_distance, max_distance + 1, 2)]
                if not valid_distances:
                    return None
                
                distance = random.choice(valid_distances)
                
                # Choose starting column ensuring both cells fit and aren't on edges
                max_start_col = n - distance - 1
                if max_start_col < 1:  # Can't place without hitting edges
                    return None
                
                col1 = random.randint(1, max_start_col)
                col2 = col1 + distance
                
                # Ensure col2 is not on the last column
                if col2 >= n - 1:
                    return None
                
                grid[row, col1] = color_1
                grid[row, col2] = color_1
                
            else:
                # Same column, different rows
                # Choose column (not first or last)
                col = random.randint(1, n - 2)
                
                # Choose two rows with proper spacing
                min_distance = 6  # 5 empty cells + 1
                max_distance = min(n - 2, 20)  # Reasonable upper bound
                
                # Ensure even distance for odd number of empty cells
                valid_distances = [d for d in range(min_distance, max_distance + 1, 2)]
                if not valid_distances:
                    return None
                
                distance = random.choice(valid_distances)
                
                # Choose starting row ensuring both cells fit and aren't on edges
                max_start_row = n - distance - 1
                if max_start_row < 1:  # Can't place without hitting edges
                    return None
                
                row1 = random.randint(1, max_start_row)
                row2 = row1 + distance
                
                # Ensure row2 is not on the last row
                if row2 >= n - 1:
                    return None
                
                grid[row1, col] = color_1
                grid[row2, col] = color_1
            
            return grid
        
        return retry(generate_valid_grid, lambda x: x is not None, max_attempts=50)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        color_1 = taskvars['color_1']
        cross_color = taskvars['cross_color']
        
        # Find the two colored cells
        colored_positions = np.where(grid == color_1)
        rows, cols = colored_positions[0], colored_positions[1]
        
        if len(rows) != 2:
            return output_grid  # Should have exactly 2 cells
        
        pos1 = (rows[0], cols[0])
        pos2 = (rows[1], cols[1])
        
        # Determine if they're in same row or column
        if pos1[0] == pos2[0]:  # Same row
            row = pos1[0]
            col1, col2 = min(pos1[1], pos2[1]), max(pos1[1], pos2[1])
            middle_col = (col1 + col2) // 2
            middle_pos = (row, middle_col)
        else:  # Same column
            col = pos1[1]
            row1, row2 = min(pos1[0], pos2[0]), max(pos1[0], pos2[0])
            middle_row = (row1 + row2) // 2
            middle_pos = (middle_row, col)
        
        # Draw cross shape with 5 cells centered at middle_pos
        center_r, center_c = middle_pos
        cross_positions = [
            (center_r, center_c),      # center
            (center_r - 1, center_c),  # up
            (center_r + 1, center_c),  # down
            (center_r, center_c - 1),  # left
            (center_r, center_c + 1)   # right
        ]
        
        # Fill cross positions that are within grid bounds
        for r, c in cross_positions:
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                output_grid[r, c] = cross_color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'n': random.randint(9, 30),  # Grid size that allows proper spacing
            'color_1': random.randint(1, 9),
            'cross_color': random.randint(1, 9)
        }
        
        # Ensure colors are different
        while taskvars['cross_color'] == taskvars['color_1']:
            taskvars['cross_color'] = random.randint(1, 9)
        
        # Generate training and test examples
        num_train = random.randint(3, 6)
        num_test = 1
        
        return taskvars, self.create_grids_default(num_train, num_test, taskvars)
