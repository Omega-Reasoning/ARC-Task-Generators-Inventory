from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskd364b489(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "The input grid contains a number of single-colored cells of color {color('object_color')}.",
            "Each single-colored cell is fully enclosed by a border of empty cells wherever the border lies within the grid, meaning it may be placed at an edge or corner as long as all neighboring cells inside the grid are empty.",
            "The border around each single-colored cell is unique and non-overlapping, no two single-colored cells share any border cell.",
            "At least one single-colored cell is guaranteed to have all four of its surrounding sides within the grid boundaries.",
            "The remaining cells are all empty.",
            "The number of colored cells may vary between input grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "All single-colored cells are identified in the grid.",
            "For each single-colored cell, let the (i, j) represent the coordinates of it.",
            "For each such cell at coordinates (i, j), color the cell above (i−1, j) with {color('color_1')}, the cell below (i+1, j) with {color('color_2')}, the cell to the left (i, j−1) with {color('color_3')}, and the cell to the right (i, j+1) with {color('color_4')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        object_color = taskvars['object_color']
        
        def generate_valid_grid():
            grid = np.zeros((n, n), dtype=int)
            
            # Number of colored cells to place (between 2 and n//4, with a reasonable maximum)
            max_cells = min(n//2, 15)  # Scale with grid size but cap at 10
            num_cells = random.randint(2, max(2, max_cells))
            
            placed_cells = []
            max_attempts = 500  # Increased for larger grids
            
            for _ in range(num_cells):
                attempts = 0
                placed = False
                
                while attempts < max_attempts and not placed:
                    r = random.randint(0, n-1)
                    c = random.randint(0, n-1)
                    
                    # Check if this position is valid
                    if self._is_valid_position(grid, r, c, object_color):
                        grid[r, c] = object_color
                        placed_cells.append((r, c))
                        placed = True
                    
                    attempts += 1
            
            return grid, placed_cells
        
        def is_valid_grid(result):
            grid, placed_cells = result
            # Must have at least 2 colored cells
            if len(placed_cells) < 2:
                return False
            
            # Check if at least one cell has all four neighbors within bounds
            has_internal_cell = False
            for r, c in placed_cells:
                if 1 <= r < n-1 and 1 <= c < n-1:
                    has_internal_cell = True
                    break
            
            return has_internal_cell
        
        grid, _ = retry(generate_valid_grid, is_valid_grid)
        return grid
    
    def _is_valid_position(self, grid: np.ndarray, r: int, c: int, object_color: int) -> bool:
        """Check if we can place a colored cell at position (r, c)"""
        n = grid.shape[0]
        
        # Check if position is already occupied
        if grid[r, c] != 0:
            return False
        
        # Find all existing colored cells
        existing_cells = []
        for i in range(n):
            for j in range(n):
                if grid[i, j] == object_color:
                    existing_cells.append((i, j))
        
        # Check minimum Chebyshev distance of 3 from all existing colored cells
        # This ensures that neighbor zones don't overlap or touch
        for er, ec in existing_cells:
            chebyshev_dist = max(abs(r - er), abs(c - ec))
            if chebyshev_dist < 3:
                return False
        
        return True
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        object_color = taskvars['object_color']
        color_1 = taskvars['color_1']  # above
        color_2 = taskvars['color_2']  # below
        color_3 = taskvars['color_3']  # left
        color_4 = taskvars['color_4']  # right
        
        n = grid.shape[0]
        
        # Find all cells with object_color
        object_positions = np.where(grid == object_color)
        
        for r, c in zip(object_positions[0], object_positions[1]):
            # Color the four neighbors if they exist
            neighbors = [
                (r-1, c, color_1),  # above
                (r+1, c, color_2),  # below
                (r, c-1, color_3),  # left
                (r, c+1, color_4)   # right
            ]
            
            for nr, nc, color in neighbors:
                if 0 <= nr < n and 0 <= nc < n:
                    output_grid[nr, nc] = color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables - using full ARC-AGI range
        n = random.randint(5, 30)  # Full range from 5 to 30
        
        # Select colors - ensure they're all different
        all_colors = list(range(1, 10))  # Colors 1-9 (excluding 0 which is background)
        selected_colors = random.sample(all_colors, 5)
        
        taskvars = {
            'n': n,
            'object_color': selected_colors[0],
            'color_1': selected_colors[1],  # above
            'color_2': selected_colors[2],  # below
            'color_3': selected_colors[3],  # left
            'color_4': selected_colors[4]   # right
        }
        
        # Generate training and test grids
        num_train = random.randint(3, 6)
        
        train_pairs = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_pairs.append({'input': input_grid, 'output': output_grid})
        
        # Generate test pair
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_pairs = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_pairs,
            'test': test_pairs
        }
        
        return taskvars, train_test_data
