from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import GridObject
from input_library import random_cell_coloring, retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskd9fac9be(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "Each grid contains exactly one 3x3 square-shaped figure.",
            "The square is composed of 8 cells of one color forming its border, and a single center cell of a different color.",
            "In addition to the square, each grid contains a random number of single-colored cells (at least one), which are colored either of the two colors used in the square and are located outside the 3x3 square.",
            "These two colors vary in each input grid.",
            "All the remaining cells are empty(0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid has a fixed size of 1×1.",
            "The color of the central cell within the 3×3 square in the input grid is identified, the single cell in the output grid is then assigned this identified color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        rows = taskvars['rows']
        cols = taskvars['cols']
        border_color = gridvars['border_color']
        center_color = gridvars['center_color']
        square_row = gridvars['square_row']
        square_col = gridvars['square_col']
        
        # Create empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Place the 3x3 square
        # Border cells (8 cells around the perimeter)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = square_row + dr, square_col + dc
                if dr == 0 and dc == 0:
                    # Center cell
                    grid[r, c] = center_color
                else:
                    # Border cell
                    grid[r, c] = border_color
        
        # Add random scattered cells outside the square
        # Create a mask for valid positions (outside the 3x3 square)
        valid_mask = np.ones((rows, cols), dtype=bool)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = square_row + dr, square_col + dc
                if 0 <= r < rows and 0 <= c < cols:
                    valid_mask[r, c] = False
        
        valid_positions = list(zip(*np.where(valid_mask)))
        
        # Add at least 1 and up to 8 scattered cells
        num_scattered = random.randint(1, min(8, len(valid_positions)))
        scattered_positions = random.sample(valid_positions, num_scattered)
        
        for r, c in scattered_positions:
            # Randomly choose between the two colors
            color = random.choice([border_color, center_color])
            grid[r, c] = color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        # Find the 3x3 square by looking for a pattern where we have 8 border cells around 1 center cell
        rows, cols = grid.shape
        
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                # Check if this could be the center of a 3x3 square
                center_color = grid[r, c]
                if center_color == 0:  # Skip empty cells
                    continue
                
                # Get all border cells
                border_cells = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue  # Skip center
                        border_cells.append(grid[r + dr, c + dc])
                
                # Check if all border cells have the same non-zero color (different from center)
                if len(set(border_cells)) == 1 and border_cells[0] != 0 and border_cells[0] != center_color:
                    # Found the square! Return the center color
                    return np.array([[center_color]])
        
        # Fallback (should not happen with valid input)
        return np.array([[0]])
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        num_train = random.randint(3, 6)
        
        taskvars = {
            'rows': random.randint(5, 30),  # Keep reasonable size for 3x3 square + scattered cells
            'cols': random.randint(5, 30)
        }
        
        def generate_example():
            def generate_valid_grid():
                # Choose two different colors (excluding background 0)
                available_colors = list(range(1, 10))
                border_color, center_color = random.sample(available_colors, 2)
                
                # Choose position for 3x3 square (ensure it fits in grid)
                square_row = random.randint(1, taskvars['rows'] - 2)
                square_col = random.randint(1, taskvars['cols'] - 2)
                
                gridvars = {
                    'border_color': border_color,
                    'center_color': center_color,
                    'square_row': square_row,
                    'square_col': square_col
                }
                
                input_grid = self.create_input(taskvars, gridvars)
                output_grid = self.transform_input(input_grid, taskvars)
                
                return input_grid, output_grid
            
            # Use retry to ensure valid grid generation
            return retry(
                generate_valid_grid,
                lambda grids: grids[1][0, 0] != 0  # Ensure we found a valid center
            )
        
        # Generate training examples
        train_examples = []
        for _ in range(num_train):
            input_grid, output_grid = generate_example()
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_input, test_output = generate_example()
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data

