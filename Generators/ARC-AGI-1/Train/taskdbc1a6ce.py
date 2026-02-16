from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskdbc1a6ceGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each grid contains a variable number of single-colored cells, all of the same color: {color('color_1')}.",
            "The single-colored cells are not adjacent to each other.",
            "The positions and count of these single-colored cells vary, in each input grid.",
            "The remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "The single-colored cells are identified.",
            "For any pair of single-colored cells in the same row or same column, all the cells between them in that row or column are filled color {color('color_2')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        height = gridvars.get('height', random.randint(5, 30))
        width = gridvars.get('width', random.randint(5, 30))
        color_1 = taskvars['color_1']
        
        def generate_valid_grid():
            grid = np.zeros((height, width), dtype=int)
            
            # Place scattered single-colored cells
            num_cells = random.randint(2, min(15, height * width // 3))
            placed_positions = set()
            
            # Try to place cells ensuring they're not adjacent and at least one pair can be filled
            attempts = 0
            while len(placed_positions) < num_cells and attempts < 100:
                r = random.randint(0, height - 1)
                c = random.randint(0, width - 1)
                
                # Check if position is valid (not adjacent to existing cells)
                valid = True
                for pr, pc in placed_positions:
                    if abs(r - pr) <= 1 and abs(c - pc) <= 1:
                        valid = False
                        break
                
                if valid:
                    placed_positions.add((r, c))
                    grid[r, c] = color_1
                
                attempts += 1
            
            return grid
        
        def has_fillable_cells(grid):
            # Check if there are at least two cells in the same row or column
            positions = [(r, c) for r in range(height) for c in range(width) if grid[r, c] == color_1]
            
            # Check for pairs in same row or column
            for i, (r1, c1) in enumerate(positions):
                for j, (r2, c2) in enumerate(positions[i+1:], i+1):
                    if r1 == r2 or c1 == c2:  # Same row or same column
                        return True
            return False
        
        return retry(generate_valid_grid, has_fillable_cells, max_attempts=50)

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output = grid.copy()
        color_1 = taskvars['color_1']
        color_2 = taskvars['color_2']
        
        # Find all positions with color_1
        positions = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) 
                    if grid[r, c] == color_1]
        
        # For each pair of positions, check if they're in the same row or column
        for i, (r1, c1) in enumerate(positions):
            for j, (r2, c2) in enumerate(positions[i+1:], i+1):
                if r1 == r2:  # Same row
                    # Fill cells between them in the row
                    start_col = min(c1, c2)
                    end_col = max(c1, c2)
                    for c in range(start_col + 1, end_col):
                        if output[r1, c] == 0:  # Only fill empty cells
                            output[r1, c] = color_2
                
                elif c1 == c2:  # Same column
                    # Fill cells between them in the column
                    start_row = min(r1, r2)
                    end_row = max(r1, r2)
                    for r in range(start_row + 1, end_row):
                        if output[r, c1] == 0:  # Only fill empty cells
                            output[r, c1] = color_2
        
        return output

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        taskvars = {
            'color_1': available_colors[0],
            'color_2': available_colors[1]
        }
        
        # Generate training examples
        num_train = random.randint(3, 6)
        train_examples = []
        
        for _ in range(num_train):
            # Generate diverse grid sizes
            height = random.randint(5, 20)
            width = random.randint(5, 20)
            gridvars = {'height': height, 'width': width}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_height = random.randint(8, 25)
        test_width = random.randint(8, 25)
        test_gridvars = {'height': test_height, 'width': test_width}
        
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
