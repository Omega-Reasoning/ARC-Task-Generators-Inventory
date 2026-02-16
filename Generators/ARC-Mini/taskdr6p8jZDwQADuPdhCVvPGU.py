from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskdr6p8jZDwQADuPdhCVvPGUGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of size {vars['grid_size']} × {vars['grid_size']}.",
            "Each input grid contains colored cells, arranged so that each row has cells of the same color.",
            "The specific color of a row can vary across examples.",
            "Some rows may be completely empty.",
            "Colored rows can also contain empty cells; If there is only one empty cell, it may appear anywhere in the row. If there are multiple empty cells, they can only appear at the ends of the row."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by adding exactly one colored cell to rows that already contain at least one colored cell but also have some empty cells.",
            "For each such row, find the first empty cell (leftmost empty cell) and fill it with the same color as the other cells in that row.",
            "If a row is completely empty or already fully filled, it is left unchanged."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Available colors (excluding background 0)
        colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        for row_idx in range(grid_size):
            # Decide what type of row this will be
            row_type = random.choices(['empty', 'full', 'partial_single', 'partial_ends', 'first_empty'], 
                                    weights=[0.15, 0.2, 0.2, 0.2, 0.25])[0]
            
            if row_type == 'empty':
                # Row remains all zeros
                continue
            
            # Choose a color for this row
            row_color = random.choice(colors)
            
            if row_type == 'full':
                # Fill entire row with same color
                grid[row_idx, :] = row_color
            
            elif row_type == 'partial_single':
                # Fill row with color but leave exactly one empty cell
                grid[row_idx, :] = row_color
                empty_pos = random.randint(0, grid_size - 1)
                grid[row_idx, empty_pos] = 0
            
            elif row_type == 'partial_ends':
                # Fill from start, leave multiple empty cells as connected strip at right end only
                # The empty strip must touch the last column (right end)
                if grid_size >= 3:
                    # Decide how many cells to leave empty at the right end
                    right_empty = random.randint(2, min(4, grid_size - 1))
                    
                    # Fill from start up to the empty section
                    fill_until = grid_size - right_empty
                    if fill_until > 0:
                        grid[row_idx, :fill_until] = row_color
                    # Leave grid[row_idx, fill_until:] as empty (0), forming connected strip at end
                else:
                    # For small grids, leave last cell empty
                    grid[row_idx, :-1] = row_color
                    
            elif row_type == 'first_empty':
                # Leave first cell empty, fill the rest - this creates transformation opportunity
                if grid_size > 1:
                    grid[row_idx, 1:] = row_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        
        for row_idx in range(grid.shape[0]):
            row = grid[row_idx]
            
            # Check if row has any colored cells
            has_colored_cells = np.any(row != 0)
            
            # Check if row has any empty cells
            has_empty_cells = np.any(row == 0)
            
            # Only transform rows that have both colored and empty cells
            if has_colored_cells and has_empty_cells:
                # Find the color used in this row (should be uniform per row)
                row_color = None
                for cell in row:
                    if cell != 0:
                        row_color = cell
                        break
                
                if row_color is not None:
                    # Find the first empty cell in the row and fill it
                    empty_positions = np.where(row == 0)[0]
                    if len(empty_positions) > 0:
                        # Fill the first empty cell (leftmost empty cell)
                        first_empty_pos = empty_positions[0]
                        output_grid[row_idx, first_empty_pos] = row_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random grid size between 5 and 15
        grid_size = random.randint(5, 30)
        
        taskvars = {
            'grid_size': grid_size
        }
        
        # Generate training examples (3-5)
        num_train = random.randint(3, 5)
        
        def generate_example():
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            return {'input': input_grid, 'output': output_grid}
        
        # Ensure diversity - generate examples that demonstrate transformation
        train_examples = []
        for _ in range(num_train):
            example = retry(
                lambda: generate_example(),
                lambda ex: not np.array_equal(ex['input'], ex['output']),  # Ensure transformation occurs
                max_attempts=50
            )
            train_examples.append(example)
        
        # Generate test example
        test_example = retry(
            lambda: generate_example(),
            lambda ex: not np.array_equal(ex['input'], ex['output']),  # Ensure transformation occurs
            max_attempts=50
        )
        
        train_test_data = {
            'train': train_examples,
            'test': [test_example]
        }
        
        return taskvars, train_test_data


