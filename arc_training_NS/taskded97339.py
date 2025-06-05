from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects
from input_library import retry, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskded97339(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['n']} x {vars['n']}.",
            "Each grid contains a variable number of single-colored cells, all of the same color: {color('color_1')}.",
            "Each row and each column, in the input grid, contains at most two such single-colored cells.",
            "Each single-colored cell is isolated — all of its neighboring cells in all 8 directions are empty (0).",
            "The positions and count of these single-colored cells vary, in each input grid.",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The single-colored cells are identified, let their color be {color('color_1')}.",
            "For any pair of single-colored cells in the same row or same column, all the cells between them in that row or column are filled with the same color: {color('color_1')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        n = taskvars['n']
        color_1 = taskvars['color_1']
        
        def generate_valid_grid():
            grid = np.zeros((n, n), dtype=int)
            
            # Generate random number of cells (ensure at least 2 for connections)
            num_cells = random.randint(4, min(12, n * 2))  # Reasonable range
            
            # Track cells per row and column
            row_counts = [0] * n
            col_counts = [0] * n
            
            placed_cells = []
            
            for _ in range(num_cells):
                # Find valid positions (at most 2 per row/col, isolated)
                valid_positions = []
                
                for r in range(n):
                    for c in range(n):
                        # Check row/column limits
                        if row_counts[r] >= 2 or col_counts[c] >= 2:
                            continue
                            
                        # Check if position is isolated (all 8 neighbors empty)
                        isolated = True
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < n and 0 <= nc < n and 
                                    grid[nr, nc] != 0):
                                    isolated = False
                                    break
                            if not isolated:
                                break
                        
                        if isolated and grid[r, c] == 0:
                            valid_positions.append((r, c))
                
                if not valid_positions:
                    break
                    
                # Place cell at random valid position
                r, c = random.choice(valid_positions)
                grid[r, c] = color_1
                row_counts[r] += 1
                col_counts[c] += 1
                placed_cells.append((r, c))
            
            return grid, placed_cells
        
        def is_valid_grid(grid_and_cells):
            grid, placed_cells = grid_and_cells
            if len(placed_cells) < 2:
                return False
                
            # Check if there's at least one pair that can be connected
            has_connection = False
            
            # Check rows
            for r in range(n):
                cells_in_row = [c for pr, c in placed_cells if pr == r]
                if len(cells_in_row) >= 2:
                    has_connection = True
                    break
            
            # Check columns  
            if not has_connection:
                for c in range(n):
                    cells_in_col = [r for r, pc in placed_cells if pc == c]
                    if len(cells_in_col) >= 2:
                        has_connection = True
                        break
            
            return has_connection
        
        grid, _ = retry(generate_valid_grid, is_valid_grid, max_attempts=200)
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        color_1 = taskvars['color_1']
        n = taskvars['n']
        
        # Find all cells with color_1
        colored_cells = []
        for r in range(n):
            for c in range(n):
                if grid[r, c] == color_1:
                    colored_cells.append((r, c))
        
        # Connect cells in same rows
        for r in range(n):
            cells_in_row = [c for pr, c in colored_cells if pr == r]
            if len(cells_in_row) >= 2:
                cells_in_row.sort()
                # Fill between consecutive pairs
                for i in range(len(cells_in_row) - 1):
                    start_col = cells_in_row[i]
                    end_col = cells_in_row[i + 1]
                    for c in range(start_col + 1, end_col):
                        output[r, c] = color_1
        
        # Connect cells in same columns
        for c in range(n):
            cells_in_col = [r for r, pc in colored_cells if pc == c]
            if len(cells_in_col) >= 2:
                cells_in_col.sort()
                # Fill between consecutive pairs
                for i in range(len(cells_in_col) - 1):
                    start_row = cells_in_col[i]
                    end_row = cells_in_col[i + 1]
                    for r in range(start_row + 1, end_row):
                        output[r, c] = color_1
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        taskvars = {
            'n': random.randint(5, 30),  # Grid size
            'color_1': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Color (not 0)
        }
        
        # Generate training and test examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data
