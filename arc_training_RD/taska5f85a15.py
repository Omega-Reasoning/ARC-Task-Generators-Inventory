from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity, retry
from transformation_library import find_connected_objects, BorderBehavior

class Taska5f85a15Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are squares and can have different sizes.",
            "They contain one or more same-colored diagonal lines, with the remaining cells being empty (0).",
            "The diagonal lines are in the direction which is parallel to the main diagonal (top-left to bottom-right).",
            "The diagonal lines may not necessarily be on the main diagonal but parallel to it.",
            "Incase of more than one diagonal line, all diagonal lines must be completely separated from each other by having at least one empty (0) diagonal line between two consecutive colored diagonal lines.",
            "If there are more than one diagonal lines, the color of each line should be the same within the grid but must vary across grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying the colored diagonal lines.",
            "Once identified, for each diagonal line, start from the second cell and change its color to a different color.", 
            "Repeat this process by changing the color of all alternating cells along the diagonal line, continuing until no more cells can be recolored."
        ]
        
        # Call parent constructor
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Generate 3-4 training pairs
        num_train_pairs = random.randint(3, 4)
        
        # Ensure we have different grid sizes for each grid
        grid_sizes = random.sample(range(15, 31), num_train_pairs + 1)  # +1 for test grid
        
        # Generate different line colors for each grid (ensuring they're all different)
        available_colors = list(range(1, 10))
        line_colors = random.sample(available_colors, num_train_pairs + 1)
        
        # Generate different fill colors for each grid (must be different from corresponding line color)
        fill_colors = []
        for line_color in line_colors:
            available_fill_colors = [c for c in available_colors if c != line_color]
            fill_colors.append(random.choice(available_fill_colors))
        
        # Generate train grids
        train_pairs = []
        for i in range(num_train_pairs):
            gridvars = {
                "size": grid_sizes[i],
                "line_color": line_colors[i],
                "fill_color": fill_colors[i],
                "has_main_diag": random.choice([True, False]),
                "num_diags": random.randint(1, 3)
            }
            
            input_grid = self.create_input(gridvars)
            output_grid = self.transform_input(input_grid, gridvars)
            train_pairs.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test grid
        test_gridvars = {
            "size": grid_sizes[-1],
            "line_color": line_colors[-1],
            "fill_color": fill_colors[-1],
            "has_main_diag": random.choice([True, False]),
            "num_diags": random.randint(1, 3)
        }
        
        test_input = self.create_input(test_gridvars)
        test_output = self.transform_input(test_input, test_gridvars)
        
        # Return empty dictionary for task variables since we don't want any predefined
        return {}, TrainTestData(
            train=train_pairs,
            test=[GridPair(input=test_input, output=test_output)]
        )
    
    def create_input(self, gridvars: dict[str, any]) -> np.ndarray:
        # Extract variables
        grid_size = gridvars["size"]
        line_color = gridvars["line_color"]
        has_main_diag = gridvars["has_main_diag"]
        num_diags = gridvars["num_diags"]
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Find valid diagonal offsets that would give at least 3 cells on each diagonal
        # An offset of k means cells at positions (i, i+k) or (i-k, i)
        # Length of diagonal with offset k is (grid_size - abs(k))
        valid_offsets = [k for k in range(-grid_size+3, grid_size-2) if abs(k) <= grid_size-3]
        
        # If we need the main diagonal, make sure 0 is in the selected offsets
        if has_main_diag:
            offsets = [0]
            if 0 in valid_offsets:
                valid_offsets.remove(0)
            num_diags -= 1
        else:
            offsets = []
        
        # Ensure diagonal lines are separated by at least one empty diagonal
        if num_diags > 0:
            # If we already have the main diagonal, exclude adjacent diagonals
            if has_main_diag:
                if 1 in valid_offsets:
                    valid_offsets.remove(1)
                if -1 in valid_offsets:
                    valid_offsets.remove(-1)
            
            # Select remaining offsets ensuring separation
            while num_diags > 0 and valid_offsets:
                offset = random.choice(valid_offsets)
                offsets.append(offset)
                valid_offsets.remove(offset)
                
                # Remove adjacent diagonals to ensure separation
                if offset+1 in valid_offsets:
                    valid_offsets.remove(offset+1)
                if offset-1 in valid_offsets:
                    valid_offsets.remove(offset-1)
                
                num_diags -= 1
        
        # Draw the diagonal lines
        for offset in offsets:
            if offset >= 0:
                # Diagonals below or on the main diagonal
                for i in range(grid_size - offset):
                    grid[i, i + offset] = line_color
            else:
                # Diagonals above the main diagonal
                for i in range(grid_size + offset):
                    grid[i - offset, i] = line_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, gridvars: dict[str, any]) -> np.ndarray:
        # Create a copy of the input grid
        output = grid.copy()
        
        # Get the fill color for this specific grid
        fill_color = gridvars["fill_color"]
        
        # Find all diagonal lines
        # We'll use diagonal detection by iterating through all possible diagonals
        grid_size = grid.shape[0]
        
        # Process each diagonal (both main and parallel)
        for offset in range(-grid_size+1, grid_size):
            if offset >= 0:
                # Diagonals below or on the main diagonal (from top-left to bottom-right)
                cells = [(i, i+offset) for i in range(grid_size-offset)]
            else:
                # Diagonals above the main diagonal (from top-left to bottom-right)
                cells = [(i-offset, i) for i in range(grid_size+offset)]
            
            # Filter valid cells and check if this is a colored diagonal
            valid_cells = [(r, c) for r, c in cells if 0 <= r < grid_size and 0 <= c < grid_size]
            diagonal_values = [grid[r, c] for r, c in valid_cells]
            
            # Skip empty diagonals or diagonals with mixed colors
            if not diagonal_values or all(val == 0 for val in diagonal_values):
                continue
                
            # Identify non-zero cells and check if they all have the same color
            non_zero_cells = [(r, c) for r, c in valid_cells if grid[r, c] != 0]
            non_zero_colors = {grid[r, c] for r, c in non_zero_cells}
            
            if len(non_zero_colors) > 1:
                continue
                
            # Ensure the diagonal has at least 3 cells before applying transformation
            if len(non_zero_cells) >= 3:
                # Apply the alternating color pattern: change every second cell to fill_color
                for i in range(1, len(non_zero_cells), 2):
                    r, c = non_zero_cells[i]
                    output[r, c] = fill_color
                
        return output