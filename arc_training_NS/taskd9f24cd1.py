from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObject, GridObjects
from input_library import retry, random_cell_coloring
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class taskd9f24cd1(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains a variable number of single-colored cells of color {color('color_1')}, located exclusively on the bottommost row.",
            "Each single-colored cell of color {color('color_1')} has at most one associated single-colored cell of color {color('color_2')}.",
            "The associated cell of color {color('color_2')} is positioned either in the same column as its corresponding {color('color_1')} cell, or in the column immediately to the left or right of it.",
            "There is a minimum spacing of one empty column between any two cells of color {color('color_1')}.",
            "Similarly, each associated {color('color_2')} cell is separated by at least one empty column from any other cell except its corresponding {color('color_1')} cell.",
            "The associated {color('color_2')} cells may be located at any row above the bottommost row;they are not located on the first or last column.",
            "All remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The cells on the bottommost row are identified, let them color be {color('color_1')}.",
            "For each cell on the bottommost row with color {color('color_1')}, it is checked whether there is any other cell above it in the same column.",
            "If no such cell exists, the entire column is filled with {color('color_1')}.",
            "If there is another cell above in the same column, let its row index be i. Then, all cells between the bottommost row and row i (excluding row i itself) in that column are filled with {color('color_1')}, and additionally, all cells from the top row down to row i in the column immediately to the right are also filled with {color('color_1')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        height = gridvars.get('height', random.randint(5, 30))
        width = gridvars.get('width', random.randint(5, 30))
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        
        def generate_valid_grid():
            grid = np.zeros((height, width), dtype=int)
            
            # Place bottom row cells with minimum spacing of 1 empty column
            bottom_positions = []
            col = 1  # Start from column 1 to avoid first column
            while col < width - 1:  # Avoid last column for associated cells
                if random.choice([True, False]) and len(bottom_positions) < (width // 3):
                    bottom_positions.append(col)
                    grid[height - 1, col] = color_1
                    col += 2  # Skip at least one column
                else:
                    col += 1
            
            if not bottom_positions:
                return None
            
            # Place associated cells for some bottom cells
            associated_positions = []
            has_same_column = False
            
            for pos in bottom_positions:
                if random.choice([True, False]):  # Not all bottom cells need associated cells
                    # Choose position: same column, left, or right
                    valid_cols = []
                    
                    # Same column
                    valid_cols.append(pos)
                    
                    # Left column (if valid and doesn't conflict)
                    if pos > 1:  # Not first column and has space
                        valid_cols.append(pos - 1)
                    
                    # Right column (if valid and doesn't conflict)
                    if pos < width - 2:  # Not last column and has space
                        valid_cols.append(pos + 1)
                    
                    if valid_cols:
                        chosen_col = random.choice(valid_cols)
                        
                        # Check spacing constraint for associated cells
                        can_place = True
                        for existing_col in associated_positions:
                            if abs(chosen_col - existing_col) < 2 and chosen_col != existing_col:
                                can_place = False
                                break
                        
                        if can_place:
                            # Place in a random row above bottom (but not top row for visibility)
                            row = random.randint(1, height - 2)
                            grid[row, chosen_col] = color_2
                            associated_positions.append(chosen_col)
                            
                            if chosen_col == pos:
                                has_same_column = True
            
            # Check constraints
            if not has_same_column:
                return None
                
            return grid
        
        return retry(generate_valid_grid, lambda x: x is not None)
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        height, width = grid.shape
        color_1 = taskvars['color_1']
        
        # Find all bottom row cells with color_1
        bottom_row = height - 1
        for col in range(width):
            if grid[bottom_row, col] == color_1:
                # Check if there's any cell above in the same column
                cell_above_row = None
                for row in range(bottom_row):
                    if grid[row, col] != 0:
                        cell_above_row = row
                        break
                
                if cell_above_row is None:
                    # Fill entire column with color_1
                    output[:, col] = color_1
                else:
                    # Fill between bottom and cell above (excluding cell above)
                    for row in range(cell_above_row + 1, bottom_row):
                        output[row, col] = color_1
                    
                    # Fill column to the right from top to cell above
                    if col + 1 < width:
                        for row in range(cell_above_row + 1):
                            output[row, col + 1] = color_1
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        available_colors = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        color_1 = random.choice(available_colors)
        available_colors.remove(color_1)
        color_2 = random.choice(available_colors)
        
        taskvars = {
            'color_1': color_1,
            'color_2': color_2
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Vary grid sizes within the specified range
            height = random.randint(5, 30)
            width = random.randint(5, 30)
            gridvars = {'height': height, 'width': width}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        height = random.randint(5, 30)
        width = random.randint(5, 30)
        gridvars = {'height': height, 'width': width}
        
        test_input = self.create_input(taskvars, gridvars)
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