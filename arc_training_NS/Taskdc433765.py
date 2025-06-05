from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects
from input_library import random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskde1cd16c(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each grid contains exactly two colored cells, one with color {color('color_1')} and the other with color {color('color_2')}.",
            "These two cells are aligned either along the same diagonal, the same row, or the same column, with at least one empty cell separating them in the corresponding direction."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The two single-colored cells are identified, and their relative positions determined; let the colors of these cells be denoted as {color('color_1')} and {color('color_2')} respectively.",
            "Depending on their spatial arrangement (whether they lie along the same diagonal, column, or row), the cell with color {color('color_1')} is moved one step closer to the cell with color {color('color_2')}. This movement occurs along the identified direction: diagonally, vertically, or horizontally, accordingly."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        """Create an input grid with exactly two aligned colored cells."""
        height = gridvars.get('height', random.randint(5, 15))
        width = gridvars.get('width', random.randint(5, 15))
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Choose alignment type: 0=row, 1=column, 2=diagonal
        alignment = gridvars.get('alignment', random.randint(0, 2))
        
        if alignment == 0:  # Same row
            row = random.randint(0, height - 1)
            # Choose two column positions with at least one cell gap
            col1 = random.randint(0, width - 3)
            col2 = random.randint(col1 + 2, width - 1)
            pos1 = (row, col1)
            pos2 = (row, col2)
            
        elif alignment == 1:  # Same column
            col = random.randint(0, width - 1)
            # Choose two row positions with at least one cell gap
            row1 = random.randint(0, height - 3)
            row2 = random.randint(row1 + 2, height - 1)
            pos1 = (row1, col)
            pos2 = (row2, col)
            
        else:  # Diagonal
            # Choose starting position ensuring we have room for diagonal with gap
            min_size = min(height, width)
            if min_size < 4:  # Need at least 4 for diagonal with gap
                # Fall back to row alignment
                row = random.randint(0, height - 1)
                col1 = random.randint(0, width - 3)
                col2 = random.randint(col1 + 2, width - 1)
                pos1 = (row, col1)
                pos2 = (row, col2)
            else:
                # Choose diagonal direction: 0=main diagonal, 1=anti-diagonal
                diag_dir = random.randint(0, 1)
                
                if diag_dir == 0:  # Main diagonal (top-left to bottom-right)
                    start_row = random.randint(0, height - 4)
                    start_col = random.randint(0, width - 4)
                    # Place cells with at least one gap
                    pos1 = (start_row, start_col)
                    pos2 = (start_row + random.randint(2, min(height - start_row - 1, width - start_col - 1)), 
                           start_col + random.randint(2, min(height - start_row - 1, width - start_col - 1)))
                else:  # Anti-diagonal (top-right to bottom-left)
                    start_row = random.randint(0, height - 4)
                    start_col = random.randint(3, width - 1)
                    gap = random.randint(2, min(height - start_row - 1, start_col))
                    pos1 = (start_row, start_col)
                    pos2 = (start_row + gap, start_col - gap)
        
        # Place the two colored cells
        grid[pos1] = color_1
        grid[pos2] = color_2
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        """Transform input by moving color_1 one step closer to color_2."""
        output_grid = grid.copy()
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        
        # Find positions of the two colors
        pos_1 = tuple(np.argwhere(grid == color_1)[0])
        pos_2 = tuple(np.argwhere(grid == color_2)[0])
        
        r1, c1 = pos_1
        r2, c2 = pos_2
        
        # Determine movement direction
        if r1 == r2:  # Same row - horizontal movement
            if c1 < c2:  # Move right
                new_pos = (r1, c1 + 1)
            else:  # Move left
                new_pos = (r1, c1 - 1)
        elif c1 == c2:  # Same column - vertical movement
            if r1 < r2:  # Move down
                new_pos = (r1 + 1, c1)
            else:  # Move up
                new_pos = (r1 - 1, c1)
        else:  # Diagonal movement
            # Determine diagonal direction
            dr = 1 if r1 < r2 else -1
            dc = 1 if c1 < c2 else -1
            new_pos = (r1 + dr, c1 + dc)
        
        # Move color_1 to new position
        output_grid[pos_1] = 0  # Clear old position
        output_grid[new_pos] = color_1  # Set new position
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        """Create task variables and train/test grids."""
        # Initialize task variables
        colors = list(range(1, 10))
        random.shuffle(colors)
        taskvars = {
            'color_1': colors[0],
            'color_2': colors[1]
        }
        
        # Create train examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Vary grid dimensions and alignment for diversity
            gridvars = {
                'height': random.randint(5, 20),
                'width': random.randint(5, 20),
                'alignment': random.randint(0, 2)  # 0=row, 1=col, 2=diag
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {
            'height': random.randint(6, 25),
            'width': random.randint(6, 25),
            'alignment': random.randint(0, 2)
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
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
