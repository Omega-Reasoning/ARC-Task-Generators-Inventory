from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import retry, create_object, random_cell_coloring, Contiguity
from transformation_library import find_connected_objects
import numpy as np
import random
from typing import Dict, Any, Tuple

class TaskHaYFUvUagZhLxKJpdjhDgtGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['rows']} × {vars['cols']}.",
            "Each input grid contains several {color('cell_color1')} cells, and in some cases, there may also be a single {color('cell_color2')} cell.",
            "All other cells are empty (0).",
            "The positions of the cells vary across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by adding {color('new_color')} cells to specific rows.",
            "These {color('new_color')} cells are added only to the rows that already contain {color('cell_color1')} cells.",
            "For each such row, {color('new_color')} cells are placed starting from the first empty cell immediately to the right of the first {color('cell_color1')} cell, and continuing all the way to the end of the row.",
            "If there are existing cells in this range, they remain unchanged, and {color('new_color')} cells are only added to the empty positions.",
            "Existing cells are never overwritten, and no overlaps occur."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create input grid with specified colors and constraints."""
        rows, cols = taskvars['rows'], taskvars['cols']
        grid = np.zeros((rows, cols), dtype=int)
        
        # Place several cell_color1 cells randomly
        num_color1_cells = random.randint(3, min(8, rows * cols // 4))
        positions = random.sample([(r, c) for r in range(rows) for c in range(cols)], 
                                num_color1_cells)
        
        for r, c in positions:
            grid[r, c] = taskvars['cell_color1']
        
        # Handle cell_color2 placement based on gridvars
        if gridvars.get('has_color2', False):
            # Find a row that doesn't have cell_color1 for isolated color2 placement
            rows_with_color1 = set(r for r, c in positions)
            empty_rows = [r for r in range(rows) if r not in rows_with_color1]
            
            if empty_rows and gridvars.get('isolated_color2', False):
                # Place color2 in a row without color1
                target_row = random.choice(empty_rows)
                available_cols = [c for c in range(cols) if grid[target_row, c] == 0]
                if available_cols:
                    col = random.choice(available_cols)
                    grid[target_row, col] = taskvars['cell_color2']
            else:
                # Place color2 anywhere that's empty
                empty_positions = [(r, c) for r in range(rows) for c in range(cols) 
                                 if grid[r, c] == 0]
                if empty_positions:
                    r, c = random.choice(empty_positions)
                    grid[r, c] = taskvars['cell_color2']
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input grid by adding new_color cells according to the rules."""
        output_grid = grid.copy()
        rows, cols = grid.shape
        
        # For each row, check if it contains cell_color1
        for r in range(rows):
            row = grid[r, :]
            color1_positions = np.where(row == taskvars['cell_color1'])[0]
            
            if len(color1_positions) > 0:
                # Find the first cell_color1 in this row
                first_color1_col = color1_positions[0]
                
                # Fill all empty cells from the first empty position to the end of the row
                for c in range(first_color1_col + 1, cols):
                    if output_grid[r, c] == 0:  # Only fill empty cells
                        output_grid[r, c] = taskvars['new_color']
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and generate train/test grids."""
        # Initialize task variables
        taskvars = {
            'rows': random.randint(5, 15),
            'cols': random.randint(5, 15),
            'cell_color1': random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'cell_color2': None,  # Will be set below
            'new_color': None     # Will be set below
        }
        
        # Ensure colors are different
        available_colors = [c for c in range(1, 10) if c != taskvars['cell_color1']]
        taskvars['cell_color2'] = random.choice(available_colors)
        available_colors.remove(taskvars['cell_color2'])
        taskvars['new_color'] = random.choice(available_colors)
        
        # Generate 3-5 training examples
        num_train = random.randint(3, 5)
        
        # Ensure exactly 2 grids have cell_color2, with at least one having isolated color2
        has_color2_indices = random.sample(range(num_train), 2)
        isolated_color2_index = random.choice(has_color2_indices)
        
        train_examples = []
        for i in range(num_train):
            gridvars = {
                'has_color2': i in has_color2_indices,
                'isolated_color2': i == isolated_color2_index
            }
            
            input_grid = retry(
                lambda: self.create_input(taskvars, gridvars),
                lambda g: self._is_valid_grid(g, taskvars, gridvars),
                max_attempts=50
            )
            
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        test_gridvars = {'has_color2': random.choice([True, False]), 'isolated_color2': False}
        test_input = retry(
            lambda: self.create_input(taskvars, test_gridvars),
            lambda g: self._is_valid_grid(g, taskvars, test_gridvars),
            max_attempts=50
        )
        test_output = self.transform_input(test_input, taskvars)
        
        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
        
        return taskvars, train_test_data
    
    def _is_valid_grid(self, grid: np.ndarray, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> bool:
        """Check if generated grid meets the requirements."""
        # Check that there are multiple color1 cells
        color1_count = np.sum(grid == taskvars['cell_color1'])
        if color1_count < 3:
            return False
        
        # Check color2 constraints
        color2_count = np.sum(grid == taskvars['cell_color2'])
        
        if gridvars.get('has_color2', False):
            if color2_count != 1:
                return False
            
            # If isolated_color2 is required, ensure color2 is in a row without color1
            if gridvars.get('isolated_color2', False):
                color2_pos = np.where(grid == taskvars['cell_color2'])
                if len(color2_pos[0]) > 0:
                    color2_row = color2_pos[0][0]
                    row_has_color1 = np.any(grid[color2_row, :] == taskvars['cell_color1'])
                    if row_has_color1:
                        return False
        else:
            if color2_count != 0:
                return False
        
        # Ensure transformation is possible (at least one row with color1 has space to the right)
        rows, cols = grid.shape
        can_transform = False
        for r in range(rows):
            row = grid[r, :]
            color1_positions = np.where(row == taskvars['cell_color1'])[0]
            if len(color1_positions) > 0:
                first_color1_col = color1_positions[0]
                if first_color1_col < cols - 1:  # Has space to the right
                    # Check if there's at least one empty cell to the right
                    for c in range(first_color1_col + 1, cols):
                        if grid[r, c] == 0:
                            can_transform = True
                            break
                if can_transform:
                    break
        
        return can_transform


