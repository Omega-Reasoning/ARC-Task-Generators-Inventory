from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple

class Taskd10ecb37Generator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['columns']} × {vars['rows']}.",
            "The top-left 2×2 subgrid of the input serves as the initial main pattern. Each of the four cells in this 2×2 block is either filled with a distinct color or left empty (0). At most one of these four cells may be empty.",
            "The grid is divided into consecutive 2-row blocks (rows 0–1, 2–3, 4–5, etc.).",
            "The first two rows (rows 0–1) are filled with successive 90-degree clockwise rotations of the initial main pattern. Each consecutive 2×2 block, placed at 2-column intervals, is rotated 90 degrees clockwise relative to the preceding block.",
            "For each subsequent 2-row block starting from rows 2–3, the initial 2×2 pattern is copied from the last 2×2 block of the preceding 2-row block. This copied pattern serves as the base for filling the current 2-row section.",
            "If the index of the current 2-row block (starting from 1 for rows 0–1) is odd, the pattern rotations in that block proceed 90 degrees clockwise every two columns. If the index is even, the rotations are 90 degrees counterclockwise.",
            "This process repeats until the entire grid is filled."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the top-left 2×2 cells of the input grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'rows': random.choice([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]),  # Even numbers
            'columns': random.choice([6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])  # Even numbers
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        columns = taskvars['columns']
        
        # Initialize grid with zeros
        grid = np.zeros((rows, columns), dtype=int)
        
        # Create initial 2x2 pattern with at most one empty cell
        pattern = self._create_initial_pattern()
        
        # Fill the grid according to the rules
        self._fill_grid_with_pattern(grid, pattern, rows, columns)
        
        return grid
    
    def _create_initial_pattern(self) -> np.ndarray:
        """Create a 2x2 initial pattern with at most one empty cell"""
        pattern = np.zeros((2, 2), dtype=int)
        
        # Choose available colors (1-9, avoiding 0 which is background)
        available_colors = list(range(1, 10))
        
        # Decide how many cells to fill (3 or 4, since at most 1 can be empty)
        num_filled = random.choice([3, 4])
        
        # Randomly select positions to fill
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        filled_positions = random.sample(positions, num_filled)
        
        # Fill selected positions with distinct colors
        selected_colors = random.sample(available_colors, num_filled)
        for i, (r, c) in enumerate(filled_positions):
            pattern[r, c] = selected_colors[i]
        
        return pattern
    
    def _rotate_pattern_clockwise(self, pattern: np.ndarray) -> np.ndarray:
        """Rotate a 2x2 pattern 90 degrees clockwise"""
        return np.rot90(pattern, k=-1)  # k=-1 for clockwise
    
    def _rotate_pattern_counterclockwise(self, pattern: np.ndarray) -> np.ndarray:
        """Rotate a 2x2 pattern 90 degrees counterclockwise"""
        return np.rot90(pattern, k=1)  # k=1 for counterclockwise
    
    def _fill_grid_with_pattern(self, grid: np.ndarray, initial_pattern: np.ndarray, rows: int, columns: int):
        """Fill the grid according to the specified pattern rotation rules"""
        
        # Number of 2-row blocks
        num_row_blocks = rows // 2
        num_col_blocks = columns // 2
        
        for row_block_idx in range(num_row_blocks):
            start_row = row_block_idx * 2
            
            if row_block_idx == 0:
                # First row block: start with initial pattern, rotate clockwise
                current_pattern = initial_pattern.copy()
            else:
                # Get the last pattern from the previous row block
                prev_row_start = (row_block_idx - 1) * 2
                last_col_start = (num_col_blocks - 1) * 2
                current_pattern = grid[prev_row_start:prev_row_start+2, last_col_start:last_col_start+2].copy()
            
            # Fill the current row block
            for col_block_idx in range(num_col_blocks):
                start_col = col_block_idx * 2
                
                # Place current pattern
                grid[start_row:start_row+2, start_col:start_col+2] = current_pattern
                
                # Prepare next pattern (if not the last column block)
                if col_block_idx < num_col_blocks - 1:
                    # Block index starts from 1 (row_block_idx + 1)
                    block_index = row_block_idx + 1
                    
                    if block_index % 2 == 1:  # Odd block index
                        current_pattern = self._rotate_pattern_clockwise(current_pattern)
                    else:  # Even block index
                        current_pattern = self._rotate_pattern_counterclockwise(current_pattern)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Extract the top-left 2x2 subgrid as output"""
        return grid[:2, :2].copy()

